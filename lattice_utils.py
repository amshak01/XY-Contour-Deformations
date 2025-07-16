import torch
import numpy as np
from matplotlib import pyplot as plt


class Corr2PtDataset(torch.utils.data.Dataset):
    def __init__(self, lattices, temperatures, x_separations, y_separations):
        """Class for storing lattice configurations and their corresponding temperatures and separations
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
        figsize=(ncols, nrows),
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
        im = axes[i].imshow(array[i], vmin=vmin, vmax=vmax, cmap="viridis")
        axes[i].axis("off")

    for i in range(N, len(axes)):
        axes[i].axis("off")

    # Horizontal colorbar
    cbar_ax = fig.add_axes([0.2, 0.02, 0.6, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.08)
    plt.show()
