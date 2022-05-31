from scipy.stats import sem


def get_stats(data, axis=0):
    mean = np.nanmean(data, axis=axis)
    err = sem(data, axis=axis)

    return {'mean': mean,
            'err': err,
            'upper': mean + 2 * err,
            'lower': mean - 2 * err,
            }