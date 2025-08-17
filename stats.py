import numpy as np
from statsmodels.tsa.stattools import acf

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
    autocorr = acf(data, nlags=50)[1:]
    n = autocorr.size
    taper = 1 - np.arange(1, n + 1) / (n + 1)
    int_act = 1 + 2 * np.sum(taper * autocorr)
    blocksize = int(np.ceil(2 * int_act))
    # print(blocksize)

    blocks = [data[i:i+blocksize] for i in range(0, len(data) - blocksize + 1)]
    boots = np.zeros(nboot)

    # Apply overlapping block boostrap as in Kunsch et al.
    for i in range(nboot):
        resampled_blocks = np.random.choice(len(blocks), size=int(np.ceil(len(data) / blocksize)), replace = True)
        resampled = np.concatenate([blocks[i] for i in resampled_blocks])
        boots[i] = stat(resampled[:len(data)])

    alpha = 1 - (level / 100)
    upper = np.quantile(boots, 1 - alpha / 2)
    lower = np.quantile(boots, alpha / 2)
    centre = stat(data)

    return centre, 2*centre - upper, 2*centre - lower


def bin_bootstrap_2d(data, stat, dim, nboot, level):

    centres = np.zeros(data.shape[dim])
    lowers = np.zeros(data.shape[dim])
    uppers = np.zeros(data.shape[dim])

    for i in range(data.shape[dim]):
        c, l, u = bin_bootstrap(data.take(i, axis=dim), stat, nboot, level)
        centres[i] = c
        lowers[i] = l
        uppers[i] = u

    return centres, lowers, uppers