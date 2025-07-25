import numpy as np

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

    return np.array([1] + [np.corrcoef(samples[:-i], samples[i:])[0, 1] for i in range(1, length)])