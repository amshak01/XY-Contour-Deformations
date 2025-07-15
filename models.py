import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ShiftOptimizer(nn.Module):
    """Basic superclass for all vertical shift deformation schemes"""

    def __init__(self, lat_size, hamiltonian, obs):
        """
        Args:
            lat_size (int): lattice size
            hamiltonian (function): function for computing the hamiltonian
            obs (function): function that computes the observable on the lattice configurations
        """

        super().__init__()

        self.lat_size = lat_size
        self.hamiltonian = hamiltonian
        self.obs = obs
        self.shifts = 0

    def deformed(self, temps, lats, *obs_args):
        """Computes a deformed observable for a vertical shift deformation

        Args:
            temp (float): system temperature
            lats (Tensor): torch tensor of size (N,L,L) with N configs of lattice size L
            *obs_args: additional variables required to compute the observable

        Returns:
            Tensor: tensor of size (N,) containing the deformed observable for each configuration
        """

        shifted = lats + 1j * self.shifts
        betas = 1 / temps

        return self.obs(shifted, *obs_args) * torch.exp(
            -betas * (self.hamiltonian(shifted) - self.hamiltonian(lats))
        )


class Corr2PtConv(ShiftOptimizer):
    """Superclass for convolution-based vertical shift deformations of the two-point correlator"""

    def __init__(self, lat_size, hamiltonian):
        from lattice_utils import corr_2d

        super().__init__(lat_size, hamiltonian, corr_2d)

    def get_masks(self, lats, x_seps, y_seps):
        """Generate N masks for each of N input configurations

        Args:
            lats (Tensor): torch tensor of size (N,L,L) with N configs of lattice size L
            x_seps (Tensor): torch tensor of size (N,) containing x separations at which to compute correlations
            y_seps (Tensor): same as **x_seps** but for y coordinate

        Returns:
            Tensor: tensor of shape (N,1,L,L) containing the N 1-channel masks for input to convolutional layers
        """

        masks = torch.zeros(lats.size(), device=lats.device).unsqueeze(
            dim=1
        )  # an extra channel dimension added
        order = torch.arange(0, lats.size(0))
        masks[order, 0, 0, 0] = 1  # Set the origin to have a value of 1 (source)
        masks[order, 0, y_seps.long(), x_seps.long()] = (
            -1
        )  # Set the other site to be opposite (sink)

        return masks


class Corr2PtConv1Layer(Corr2PtConv):
    """Class for single-layer convolution"""

    def __init__(self, lat_size, hamiltonian, kernel_size=8):
        """
        Args:
            lat_size (int): lattice size
            hamiltonian (function): function for computing the hamiltonian
            kernel_size (int, optional): width of the square convolution kernel. Defaults to 8.
        """

        super().__init__(lat_size, hamiltonian)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="circular",
            bias=False,
        )

    def forward(self, lats, temps, x_seps, y_seps):
        """Forward pass of the 1-layer CNN

        Args:
            lats (Tensor): tensor of lattice configs with shape (N,L,L) for batch size N and lattice size L
            temp (float): system temperature
            x_sep (int): separation in x
            y_sep (int): separation in y

        Returns:
            Tensor: deformed 2-point correlator as tensor of shape (N,1,1)
        """

        masks = self.get_masks(lats, x_seps, y_seps)

        self.shifts = self.conv(masks).squeeze(1)
        return self.deformed(temps, lats, x_seps, y_seps)


class Corr2PtUNet(Corr2PtConv):
    """U-Net architecture for temperature generalization"""

    def __init__(self, lat_size, hamiltonian, min_size=16):
        """
        Args:
            lat_size (int): lattice size
            hamiltonian (function): function for computing the hamiltonian
            conv_width (int, optional): width of convolution kernel. Defaults to 4.
            pool_width (int, optional): width of max pooling kernel. Defaults to 4.
        """

        super().__init__(lat_size, hamiltonian)

        self.levels = int(np.log2(lat_size) - np.log2(min_size)) + 1

        self.encoder_expand_convs = nn.ModuleList()
        self.encoder_refine_convs = nn.ModuleList()
        self.decoder_reduce_convs = nn.ModuleList()
        self.decoder_upsample_convs = nn.ModuleList()
        self.temp_mlps = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)

        for i in range(self.levels):

            self.encoder_expand_convs.append(
                nn.Conv2d(
                    in_channels=2**i,
                    out_channels=2 ** (i + 1),
                    kernel_size=3,
                    padding="same",
                    padding_mode="circular",
                )
            )

            self.encoder_refine_convs.append(
                nn.Conv2d(
                    in_channels=2 ** (i + 1),
                    out_channels=2 ** (i + 1),
                    kernel_size=3,
                    padding="same",
                    padding_mode="circular",
                )
            )

            if i < self.levels - 1:

                self.decoder_upsample_convs.append(
                    nn.ConvTranspose2d(
                        in_channels=2 ** (self.levels - i),
                        out_channels=2 ** (self.levels - i - 1),
                        kernel_size=2,
                        stride=2,
                    )
                )

                self.temp_mlps.append(
                    nn.Sequential(
                        nn.Linear(1, 2 ** (self.levels - i - 1)),
                        nn.SiLU(),
                        nn.Linear(
                            2 ** (self.levels - i - 1), 2 ** (self.levels - i - 1)
                        ),
                        nn.SiLU(),
                    )
                )

            self.decoder_reduce_convs.append(
                nn.Conv2d(
                    in_channels=2 ** (self.levels - i),
                    out_channels=2 ** (self.levels - i - 1),
                    kernel_size=3,
                    padding="same",
                    padding_mode="circular",
                )
            )

    def forward(self, lats, temps, x_seps, y_seps):

        out = self.get_masks(lats, x_seps, y_seps)
        out_copies = []

        for i in range(self.levels):

            out = F.silu(
                self.encoder_refine_convs[i](F.silu(self.encoder_expand_convs[i](out)))
            )

            if i < self.levels - 1:
                out_copies.append(out.clone())
                out = self.pool(out)

        for i in range(self.levels - 1):

            temp_rescaling = self.temp_mlps[i](temps.unsqueeze(-1).float())
            skip_connection = torch.einsum(
                "bchw,bc->bchw", out_copies[self.levels - 2 - i], temp_rescaling
            )

            out = torch.cat(
                (self.decoder_upsample_convs[i](out), skip_connection),
                dim=-3,
            )  # dim -3 is channels
            out = F.silu(
                self.encoder_refine_convs[self.levels - 2 - i](
                    F.silu(self.decoder_reduce_convs[i](out))
                )
            )

        self.shifts = self.decoder_reduce_convs[-1](out).squeeze(1)

        return self.deformed(temps, lats, x_seps, y_seps)
