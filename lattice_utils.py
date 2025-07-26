import torch
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import os, re


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
    shifted = lats + 1j * shifts # Apply the vertical shift deformation
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


def plot_tensor_grid(tensor):
    assert tensor.ndim == 3, "Input tensor must have shape (N, L, L)"
    N, L1, L2 = tensor.shape
    assert L1 == L2, "Each slice must be square (L, L)"

    # Convert to numpy
    array = tensor.detach().cpu().numpy()

    # Determine grid size
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))

    # Create figure and subplots with no gaps
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2, nrows * 2),
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0.05, hspace=0.05),
    )

    axes = axes.flat

    # Common color scale
    vmin = array.min()
    vmax = array.max()

    im = None
    for i in range(N):
        im = axes[i].imshow(array[i], vmin=vmin, vmax=vmax, cmap="seismic")
        axes[i].axis("off")

    for i in range(N, len(axes)):
        axes[i].axis("off")

    # Horizontal colorbar
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.08)
    plt.show()


def plot_hists(initial, deformed):
    """Plot histograms of initial and deformed configurations

    Args:
        initial (Tensor): tensor of initial configurations
        deformed (Tensor): tensor of deformed configurations
    """

    xi, yi = initial.real, initial.imag
    xf, yf = deformed.real, deformed.imag

    x_means = (xi.mean(), xf.mean())
    y_means = (yi.mean(), yf.mean())

    xlim = (-1, 1)
    ylim = (-1, 1)
    # --- Create Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    # Plot histograms and store images
    h1 = axes[0].hist2d(xi, yi, bins=50, range=[xlim, ylim])
    h2 = axes[1].hist2d(xf, yf, bins=50, range=[xlim, ylim])

    # Titles and labels
    axes[0].set_title("Undeformed observable $\\mathcal{O}$")
    axes[1].set_title("Deformed Observable $\\mathcal{Q}$")

    axes[0].set_xlabel("$\\mathrm{Re}(\\mathcal{O})$")
    axes[0].set_ylabel("$\\mathrm{Im}(\\mathcal{O})$")

    axes[1].set_xlabel("$\\mathrm{Re}(\\mathcal{Q})$")
    axes[1].set_ylabel("$\\mathrm{Im}(\\mathcal{Q})$")

    for i in range(len(axes)):
        axes[i].set_xticks(np.linspace(-1, 1, 5))
        axes[i].set_yticks(np.linspace(-1, 1, 5))
        axes[i].vlines(x_means[i], ymin=-1, ymax=1, color="white", ls="--", lw=1)
        axes[i].hlines(y_means[i], xmin=-1, xmax=1, color="white", ls="--", lw=1)

    # --- Shared Colorbar ---
    # Use the mappable from one of the histograms (e.g. h1[3])
    cbar = fig.colorbar(h1[3], ax=axes.ravel().tolist(), orientation="vertical")
    cbar.set_label("Counts")

    plt.show()
