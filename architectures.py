import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torchinfo import summary

from livelossplot import PlotLosses

# import arviz


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

        self.shifts = self.conv(masks).squeeze()
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

            out = torch.cat(
                (self.decoder_upsample_convs[i](out), out_copies[self.levels - 2 - i]),
                dim=-3,
            )  # dim -3 is channels
            out = F.silu(
                self.encoder_refine_convs[self.levels - 2 - i](
                    F.silu(self.decoder_reduce_convs[i](out))
                )
            )

        self.shifts = self.decoder_reduce_convs[-1](out).squeeze()

        return self.deformed(temps, lats, x_seps, y_seps)


class Corr2PtDataset(torch.utils.data.Dataset):
    def __init__(self, lattices, temperatures, x_separations, y_separations):
        """
        Args:
            lattices (Tensor): Tensor of size (N,L,L) containing N LxL lattice configs
            temperatures (Tensor): array of length N containing temperature labels for each lattice config
            x_separations (Tensor): array of length N containing x separations at which to compute correlation
            y_separations (Tensor): same as **x_separations** but for y separations
        """

        super().__init__()

        self.temps = temperatures
        self.x_seps = x_separations
        self.y_seps = y_separations
        self.lats = lattices

    def __len__(self):
        return self.temps.size(0)

    def __getitem__(self, index):
        return (
            self.lats[index, ...],
            self.temps[index],
            self.x_seps[index],
            self.y_seps[index],
        )


def load_configs(L, T, dir="configs"):
    """Loads lattice configurations from files with consistent naming schemes

    Args:
        L (int): lattice size
        T (float): temperature
        dir (str): directory in which files reside

    Returns:
        Tensor: tensor of shape (N,L,L) with N configs of lattice size L
    """

    filename = (
        dir + "/L=" + str(L) + "_cluster_T=" + "{:.4f}".format(T) + "_configs.bin"
    )
    return torch.from_numpy(np.fromfile(filename).reshape(-1, L, L)).to(
        dtype=torch.cfloat
    )


def identity(vals):
    """Identity function for reweighting"""
    return np.ones_like(vals)


def reweight(temps, dist=identity):
    """_summary_

    Args:
        temps (ndarray): temperatures to reweight
        dist (function, optional): probability mass distribution. Defaults to uniform.

    Returns:
        ndarray: calculated weights for sampling
    """

    unique_temps = np.unique(temps)
    densities = np.zeros_like(temps)

    for unique_temp in unique_temps:
        bool_mask = temps == unique_temp
        densities[bool_mask] = bool_mask.size / np.sum(bool_mask)

    return densities * dist(temps)


def load_temp_range(temps, lat_size, dir="configs"):
    """Loads data for temperatures over a range

    Args:
        temps (ndarray): temperature range
        lat_size (int): lattice size
        dir (str, optional): folder in which files reside. Defaults to "configs".

    Returns:
        tuple(Tensor, Tensor): torch tensors containing lattice configs and temperature labels
    """

    lats = []
    temp_labels = []

    for i in range(temps.size):

        configs = load_configs(lat_size, temps[i], dir)

        n = configs.size(0)
        cut = configs[n // 20 :, ...]

        lats.append(cut)
        temp_labels.append(torch.ones(lats[i].size(0)) * temps[i])

    return (lats, temp_labels)


def xy_hamiltonian(lats):
    """Computes the XY hamiltonian for a batch of lattice configurations

    Args:
        lats (Tensor): tensor of shape (N,L,L) with N configs of lattice size L

    Returns:
        Tensor: tensor of shape (N,) containing the value of the XY hamiltonian for each config
    """

    return (
        (1 - torch.cos(lats - lats.roll(1, -2)))
        + (1 - torch.cos(lats - lats.roll(1, -1)))
    ).sum((-1, -2))


def corr_2d(lats, x_seps, y_seps):
    """Computes the contributions to the two-point correlator for a set of lattice configurations

    Args:
        lats (Tensor): tensor of shape (N,L,L) with N configs of lattice size L
        x_seps (Tensor): torch tensor of size (N,) containing x separations at which to compute correlations
        y_seps (Tensor): same as **x_seps** but for y coordinate

    Returns:
        Tensor: tensor of shape (N,) containing the contributions to the correlator
    """

    # print(lats, x_seps, y_seps)

    N = lats.size(0)
    order = torch.arange(0, N)

    return torch.exp(
        1j * (lats[order, 0, 0] - lats[order, y_seps.long(), x_seps.long()])
    )


def helicity_modulus(lats, T):
    """Computes the contributions to the spin stiffness for a set of lattice configurations

    Args:
        lats (Tensor): tensor of shape (N,L,L) with N configs of lattice size L
        T (float): Temperature at which the configurations were sampled

    Returns:
        ndarray: numpy array of floats with shape (N,) containing the contributions
    """

    L = lats.shape[-1]
    beta = 1 / T
    roll_x = lats.roll(1, -1)

    cosine_term = torch.cos(lats - roll_x).sum((-1, -2))
    sine_term = torch.square(torch.sin(lats - roll_x).sum((-1, -2)))

    return ((cosine_term / (L**2)) - (beta * (sine_term / (L**2)))).numpy().real


def mean_squared_mag(lats):
    """Computes the contributions to the mean-squared magnetization for a set of lattice configurations

    Args:
        lats (Tensor): tensor of shape (N,L,L) with N configs of lattice size L

    Returns:
        ndarray: numpy array of floats with shape (N,) containing the contributions
    """

    L = lats.shape[-1]
    N = L**2
    mag_squared = (torch.exp(1j * lats).sum((-2, -1))) * (
        torch.exp(-1j * lats).sum((-2, -1))
    )
    return mag_squared.numpy().real / (N**2)


def bin_bootstrap(data, stat, nboot, level):
    """Bootstrap correlated data using bins of size determined by ac length

    Args:
        data (ndarray): numpy array of measurements
        stat (function): function that computes the desired statistic from data
        nboot (int): number of times to resample
        level (float): desired confidence level on returned interval

    Returns:
        (float, float, float): centre along with lower and upper bounds of confidence interval, respectively
    """

    # Compute bin size from acf
    thresh = 2 / np.sqrt(data.size)
    binsize = np.argmax(np.abs(acf(data, length=25)) < thresh) + 1
    # print(binsize)

    # Cut off trailing data points so length divisible by bin size
    # (need to find a better way but this is fine for now)
    cut = data.size % binsize
    samples = data[: data.size - cut]

    binned = np.reshape(samples, shape=(-1, binsize))
    n_bins = binned.shape[0]
    boots = np.zeros(nboot)

    for i in range(nboot):
        resampled = binned[np.random.randint(0, n_bins, size=n_bins), :].flatten()
        boots[i] = stat(resampled)

    alpha = 1 - (level / 100)
    upper = np.quantile(boots, alpha / 2)
    lower = np.quantile(boots, 1 - alpha / 2)

    est = stat(samples)
    return (stat(data), est + (est - lower), est + (est - upper))


def acf(samples, length=20):
    """Compute the autocorrelation function for a list of samples
    Courtesy of https://stackoverflow.com/a/7981132

    Args:
        samples (ndarray): data on which to compute acf
        length (int, optional): maximum distance at which acf will be computed. Defaults to 20.

    Returns:
        ndarray: 1D array of 'floats' of size 'length'
    """

    return np.array(
        [1] + [np.corrcoef(samples[:-i], samples[i:])[0, 1] for i in range(1, length)]
    )


def train(model, device, loader, optimizer, epochs, scheduler=None):
    """_summary_

    Args:
        model (nn): torch neural network to train
        device (device): device on which to train (cuda or cpu)
        loader (DataLoader): dataloader that feeds samples during training
        epochs (int): number of epochs to train for
        learning_rate (float): learning rate

    Returns:
        list[float]: list containing loss at each epoch for plotting/diagnostic purposes
    """

    liveloss = PlotLosses()
    logs = {}
    # train_losses = []

    model.train()

    for i in range(epochs):
        total_loss = 0

        for batch, inputs in enumerate(loader):

            inputs = tuple(input.to(device) for input in inputs)

            optimizer.zero_grad()

            reQ = model(*inputs).real
            loss = torch.mean(reQ**2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(loader)
        # train_losses.append(avg_loss)
        logs["train"] = avg_loss

        liveloss.update(logs)
        liveloss.send()

    # return train_losses
