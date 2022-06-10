import numpy as np

from scipy.stats import sem


def get_fig_stats(data, axis=0):
    if data.any():  # if not empty
        mean = np.nanmean(data, axis=axis)
        err = sem(data, axis=axis)
        upper = mean + 2 * err
        lower = mean - 2 * err
    else:
        mean = []
        err = []
        upper = []
        lower = []

    stats_dict = dict(mean=mean,
                      err=err,
                      upper=upper,
                      lower=lower,
                      )

    return stats_dict


def get_descriptive_stats(data, axis=0):
    mean = np.nanmean(data, axis=axis)
    median = np.nanmedian(data, axis=axis)
    std = np.nanstd(data, axis=axis)
    err = sem(data, axis=axis)
    upper = mean + 2 * err  # 95% CI
    lower = mean - 2 * err  # 95% CI
    quantiles = np.quantile(data, [0, 0.25, 0.5, 0.75, 1])
    n = np.shape(data)[axis]

    stats_dict = dict(mean=mean,
                      median=median,
                      std=std,
                      err=err,
                      upper=upper,
                      lower=lower,
                      quantiles=quantiles,
                      n=n,
                      )

    return stats_dict