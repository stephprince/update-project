import numpy as np

from scipy.stats import sem, ranksums, kstest


def get_fig_stats(data, axis=0):
    if len(data):  # if not empty
        mean = np.nanmean(data, axis=axis)
        err = sem(data, axis=axis, nan_policy='omit')
        upper = mean + 2 * err
        lower = mean - 2 * err
        err_upper = mean + err
        err_lower = mean - err
    else:
        mean = []
        err = []
        upper = []
        lower = []
        err_upper = []
        err_lower = []
        # nan_arr = np.empty(np.delete(np.shape(data), axis))

    stats_dict = dict(mean=mean,
                      err=err,
                      upper=upper,
                      lower=lower,
                      err_upper=err_upper,
                      err_lower=err_lower
                      )

    return stats_dict


def get_descriptive_stats(data, name=None, axis=0):
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


def get_comparative_stats(x, y):
    rs_test_statistic, rs_p_value = ranksums(x, y)
    ks_test_statistic, ks_p_value = kstest(x, y)

    stats_dict = dict(ranksum=dict(test_statistic=rs_test_statistic,
                                   p_value=rs_p_value,),
                      kstest=dict(test_statistic=ks_test_statistic,
                                  p_value=ks_p_value,),
                      )

    return stats_dict


def get_stats_summary(data_dict, axis=0):
    plus_minus = '\u00B1'
    new_line = '\n'

    summary_list = []
    keys = []
    for key, value in data_dict.items():
        stats = get_descriptive_stats(value, axis)
        keys.append(key)
        summary_list.append(f"{key}: {stats['mean']} {plus_minus} {stats['err']}, n = {stats['n']}, "
                            f"quantiles = {stats['quantiles']}{new_line}")

    comparative_stats = get_comparative_stats(*data_dict.values())
    for key, value in comparative_stats.items():
        summary_list.append(f"{key}: p = {value['p_value']}, test-statistic = {value['test_statistic']}{new_line}")

    summary_text = ''.join(summary_list)

    return summary_text
