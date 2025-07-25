import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ConvNet(nn.Module):
    """Superclass for convolution-based vertical shift deformations of the two-point correlator"""

    def __init__(self, lat_size):
        super().__init__()

        self.lat_size = lat_size

    def get_masks(self, x_seps, y_seps):
        """Generate N masks for each of N input configurations

        Args:
            lats (Tensor): torch tensor of size (N,L,L) with N configs of lattice size L
            x_seps (Tensor): torch tensor of size (N,) containing x separations at which to compute correlations
            y_seps (Tensor): same as **x_seps** but for y coordinate

        Returns:
            Tensor: tensor of shape (N,1,L,L) containing the N 1-channel masks for input to convolutional layers
        """

        N = x_seps.size(0)
        size = (N, self.lat_size, self.lat_size)
        masks = torch.zeros(size, device=x_seps.device).unsqueeze(dim=1)  # an extra channel dimension added
        order = torch.arange(0, N)
        masks[order, 0, 0, 0] = 1  # Set the origin to have a value of 1 (source)
        masks[order, 0, y_seps.long(), x_seps.long()] = -1  # Set the other site to be opposite (sink)

        return masks


class Conv1Layer(ConvNet):
    """Class for single-layer convolution"""

    def __init__(self, lat_size, kernel_size=8):
        """
        Args:
            lat_size (int): lattice size
            kernel_size (int, optional): width of the square convolution kernel. Defaults to 8.
        """

        super().__init__(lat_size)

        # Single convolution layer
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="circular",
            bias=False,
        )

    def forward(self, temps, x_seps, y_seps):
        """Forward pass of the 1-layer CNN

        Args:
            temp (float): system temperature
            x_sep (int): separation in x
            y_sep (int): separation in y

        Returns:
            Tensor: computed shift field as size (N,L,L)
        """

        masks = self.get_masks(x_seps, y_seps)

        shifts = self.conv(masks).squeeze(1)
        return shifts


class UNet(ConvNet):
    """U-Net architecture for temperature generalization"""

    def __init__(self, lat_size, min_size=4):
        """
        Args:
            lat_size (int): lattice size
            min_size (int, optional): minimum size of the lattice at the coarsest level. Defaults to 4.
        """

        super().__init__(lat_size)

        self.levels = int(np.log2(lat_size) - np.log2(min_size)) + 1  # Number of levels in the U-Net
        assert self.levels > 0, "The lattice size must be larger than the minimum size."

        self.encoder_expand_convs = nn.ModuleList()  # Expands the number of channels

        self.encoder_refine_convs = nn.ModuleList()  # Refines the features in the expanded channels

        self.decoder_reduce_convs = nn.ModuleList()  # Reduces the number of channels after upsampling

        self.decoder_upsample_convs = nn.ModuleList()  # Upsamples the features to the next level

        self.temp_channel_rescaling = nn.ModuleList()  # MLPs to incorporate temperature in the skip connections

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)  # Max pooling layer

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

            # MLP accepts scalar and outputs per-channel scaling factors and biases
            self.temp_channel_rescaling.append(
                nn.Sequential(
                    nn.Linear(1, 2 ** (i + 1)), nn.SiLU(), nn.Linear(2 ** (i + 1), 2 ** (i + 2)), nn.SiLU()
                )  # Output is double the number of channels -- chunk into scale and bias
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

            self.decoder_reduce_convs.append(
                nn.Conv2d(
                    in_channels=2 ** (self.levels - i),
                    out_channels=2 ** (self.levels - i - 1),
                    kernel_size=3,
                    padding="same",
                    padding_mode="circular",
                )
            )

    def forward(self, temps, x_seps, y_seps):

        out = self.get_masks(x_seps, y_seps)
        out_copies = []

        # Encoding path
        for i in range(self.levels):

            out = F.silu(self.encoder_expand_convs[i](out))  # Expand the number of channels

            temp_rescaling, temp_bias = self.temp_channel_rescaling[i](
                temps.unsqueeze(-1).float()  # Unsqueeze to add singleton dimension for MLP input
            ).chunk(
                2, dim=-1
            )  # Split into scaling and bias, 2 ** (i + 1) each

            out = out * temp_rescaling[:, :, None, None] + temp_bias[:, :, None, None]  # Apply FiLM layer

            out = F.silu(self.encoder_refine_convs[i](out))  # Expecting 2 ** (i + 1) channels

            if i < self.levels - 1:
                out_copies.append(out.clone())
                out = self.pool(out)

        # Decoding path with skip connections
        for i in range(self.levels - 1):

            skip_connection = out_copies[self.levels - 2 - i]

            out = torch.cat(
                (self.decoder_upsample_convs[i](out), skip_connection),
                dim=-3,
            )  # dim -3 is channels -- concatenate along channels

            out = F.silu(self.encoder_refine_convs[self.levels - 2 - i](F.silu(self.decoder_reduce_convs[i](out))))

        # Final output layer
        shifts = self.decoder_reduce_convs[-1](out).squeeze(1)

        return shifts
