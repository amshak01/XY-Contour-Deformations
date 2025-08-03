import torch
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import os, re
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde


class BinFileDataset(Dataset):
    """Dataset for reading binary files containing lattice configurations"""

    def __init__(self, folder, fname, seps, random_seps=False, dtype=np.float64):
        """Args:
        folder (str): folder containing the binary file
        fname (str): name of the binary file
        seps (tuple): tuple of two integers specifying the minimum x and y separations for the masks
        random_seps (bool, optional): whether to randomly sample separations or use fixed ones. Defaults to False.
        dtype (type, optional): data type of the binary file. Defaults to np.float64.
        """

        assert fname.endswith(".bin"), "File must be binary data with extension '.bin'"

        self.filepath = os.path.join(folder, fname)
        self.L, self.T = parse_filename(fname)

        self.dtype = dtype

        self._memmap = np.memmap(self.filepath, dtype=self.dtype, mode="r")
        total_elements = self._memmap.size
        elements_per_array = self.L * self.L

        if total_elements % elements_per_array != 0:
            raise ValueError("File size is not divisible by LxL. Possibly wrong L or data corruption.")

        self.N = total_elements // elements_per_array

        self.get_seps = None
        self.seps = seps

        if random_seps:
            self.get_seps = lambda: (
                np.random.randint(low=self.seps[0], high=self.L - self.seps[0] + 1),
                np.random.randint(low=self.seps[1], high=self.L - self.seps[1] + 1),
            )
        else:
            self.get_seps = lambda: self.seps

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        start = idx * self.L * self.L
        end = (idx + 1) * self.L * self.L
        array = self._memmap[start:end].reshape(self.L, self.L)

        x, y = self.get_seps()

        return (
            torch.from_numpy(array.copy()).float(),
            torch.tensor(self.T, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )  # still casting to float32 for model input


class MultiFileDataset(Dataset):
    """Dataset for reading multiple binary files from a folder containing lattice configurations at different temperatures"""

    def __init__(self, folder, min_sep=1, dtype=np.float64):
        """Args:
        folder (str): folder containing the binary files
        min_sep (int, optional): minimum separation for the masks. Defaults to 1.
        dtype (type, optional): data type of the binary files. Defaults to np.float64.
        """

        self.datasets = []
        self.idx_map = []

        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".bin"):

                ds = BinFileDataset(folder, fname, seps=(min_sep, min_sep), random_seps=True, dtype=dtype)
                self.datasets.append(ds)

                for i in range(len(ds)):
                    self.idx_map.append((len(self.datasets) - 1, i))

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self.idx_map[idx]
        return self.datasets[file_idx][local_idx]


def parse_filename(filename):
    match = re.search(r"L=(\d+)_cluster_T=([\d.]+)", filename)
    if not match:
        raise ValueError(f"Filename format unexpected: {filename}")
    L = int(match.group(1))
    T = float(match.group(2))
    return L, T


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


def xy_hamiltonian(lats):
    """Computes the XY hamiltonian for a batch of lattice configurations

    Args:
        lats (Tensor): tensor of shape (N,L,L) with N configs of lattice size L

    Returns:
        Tensor: tensor of shape (N,) containing the value of the XY hamiltonian for each config
    """

    return ((1 - torch.cos(lats - lats.roll(1, -2))) + (1 - torch.cos(lats - lats.roll(1, -1)))).sum((-1, -2))


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

    return torch.exp(1j * (lats[order, 0, 0] - lats[order, y_seps.long(), x_seps.long()]))


def deformed_corr_2d(lats, temps, shifts, x_seps, y_seps, ham=xy_hamiltonian):
    """Computes the deformed two-point correlator for a set of lattice configurations

    Args:
        lats (Tensor): tensor of shape (N,L,L) with N configs of lattice size L
        shifts (Tensor): tensor of shape (N,) with vertical shifts applied to each config
        x_seps (Tensor): torch tensor of size (N,) containing x separations at which to compute correlations
        y_seps (Tensor): same as **x_seps** but for y coordinate

    Returns:
        Tensor: tensor of shape (N,) containing the deformed 2-point correlator
    """

    _, L1, L2 = lats.shape
    shifted = lats + 1j * shifts  # Apply the vertical shift deformation
    betas = 1 / temps
    reweighting = torch.exp(-betas * (ham(shifted) - ham(lats)))

    return corr_2d(shifted, x_seps, y_seps) * reweighting


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
    mag_squared = (torch.exp(1j * lats).sum((-2, -1))) * (torch.exp(-1j * lats).sum((-2, -1)))
    return mag_squared.numpy().real / (N**2)


def plot_hists(initial, deformed, fig=None):
    if hasattr(initial, "detach"):
        initial = initial.detach().cpu().numpy()
    if hasattr(deformed, "detach"):
        deformed = deformed.detach().cpu().numpy()

    xi, yi = initial.real, initial.imag
    xf, yf = deformed.real, deformed.imag

    xlim = (-1, 1)
    ylim = (-1, 1)

    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        axes = fig.subplots(1, 2)

    for i, (x, y, ax_main, xlabel, ylabel) in enumerate(
        [
            (xi, yi, axes[0], "Re($\\mathcal{O}$)", "Im($\\mathcal{O}$)"),
            (xf, yf, axes[1], "Re($\\mathcal{Q}$)", "Im($\\mathcal{Q}$)"),
        ]
    ):
        # --- Divider for marginals and colorbar ---
        divider = make_axes_locatable(ax_main)
        ax_x = divider.append_axes("top", size="20%", pad=0.05, sharex=ax_main)
        ax_y = divider.append_axes("right", size="20%", pad=0.05, sharey=ax_main)
        cbar_ax = divider.append_axes("bottom", size="5%", pad=0.3)

        # --- 2D Histogram ---
        h = ax_main.hist2d(x, y, bins=50, range=[xlim, ylim], cmap="viridis", norm=Normalize(vmin=0))

        # --- Marginal KDEs ---
        def plot_kde(ax, data, axis):
            kde = gaussian_kde(data)
            grid = np.linspace(-1, 1, 500)
            density = kde(grid)
            if axis == "x":
                ax.plot(grid, density, color="black")
            else:
                ax.plot(density, grid, color="black")

        plot_kde(ax_x, x, "x")
        plot_kde(ax_y, y, "y")

        # --- Cleanup ---
        ax_main.set_xlim(xlim)
        ax_main.set_ylim(ylim)
        ax_main.set_aspect("equal")
        ax_main.axvline(np.mean(x), color="white", linestyle="--")
        ax_main.axhline(np.mean(y), color="white", linestyle="--")

        ax_main.set_xlabel(xlabel)
        ax_main.set_ylabel(ylabel)

        for ax in [ax_x, ax_y]:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_facecolor("none")

        # --- Colorbar (now same width as plot) ---
        cb = fig.colorbar(h[3], cax=cbar_ax, orientation="horizontal")
        cb.set_label("Counts")

        # --- Stats box ---
        x_mean, y_mean = np.mean(x), np.mean(y)
        x_var, y_var = np.var(x), np.var(y)
        stat_txt = (
            f"$\\mu_x$ = {x_mean:.4f}\n"
            f"$\\mu_y$ = {y_mean:.4f}\n"
            f"$\\sigma^2_x$ = {x_var:.4f}\n"
            f"$\\sigma^2_y$ = {y_var:.4f}"
        )
        ax_main.text(
            0.02,
            0.02,
            stat_txt,
            transform=ax_main.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
        )

    fig.tight_layout(pad=0.5)  # Reduce internal padding
    return fig
