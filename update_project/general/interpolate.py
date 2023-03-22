import numpy as np

from bisect import bisect
from scipy.interpolate import griddata, interp1d


def interp_timeseries(timeseries, trials, start_label, stop_label, start_window=0.0, stop_window=0.0,
                      step=0.01, time_offset=None):
    start_times_all = trials[start_label][:]
    stop_times_all = trials[stop_label][:]
    start_times = start_times_all[np.logical_and(~np.isnan(stop_times_all), ~np.isnan(start_times_all))]
    stop_times = stop_times_all[np.logical_and(~np.isnan(stop_times_all), ~np.isnan(start_times_all))]

    timestamps = timeseries.index.values
    data = timeseries.values
    interpolated_data = []
    new_times = []
    for start, stop in zip(start_times, stop_times):
        # extract data interval
        offset = time_offset or start  # center to start time by default
        idx_start = bisect(timestamps, start + start_window*2)  # extra time for interpolation's sake
        idx_stop = bisect(timestamps, stop + stop_window*2, lo=idx_start)
        data_aligned = np.array(data[idx_start:idx_stop])
        time_aligned = np.array(timestamps[idx_start:idx_stop]) - offset

        # interpolate data interval to make even sampling rate
        fxn = interp1d(time_aligned, data_aligned, kind='nearest')
        new_times = np.arange(start + start_window - offset, stop + stop_window - offset, step)  # resample to constant sampling rate
        interpolated_data.append(fxn(new_times))

    return interpolated_data, new_times

def interp1d_time_intervals(data, start_locs, stop_locs, new_times, time_offset):
    interpolated_position = []
    for start, stop, offset in zip(start_locs, stop_locs, time_offset):
        times = np.array(data.iloc[start:stop].index) - offset
        values = data.iloc[start:stop].values

        fxn = interp1d(times, values, kind='nearest', bounds_error=False)
        interpolated_position.append(fxn(new_times))

    return interpolated_position


def griddata_time_intervals(data, start_locs, stop_locs, nbins, time_offset=[0], method='nearest',
                            time_bins=None):
    grid_prob = []
    for start, stop, offset in zip(start_locs, stop_locs, time_offset):
        proby = data.iloc[start:stop].stack().reset_index().values
        proby[:, 0] = proby[:, 0] - offset

        if time_bins is None:
            x1 = np.linspace(min(proby[:, 0]), max(proby[:, 0]), nbins)  # time bins
        else:
            x1 = time_bins  # use user-specified time bins

        # # ok to grab position bins from max/min bc these come from the dataframe columns
        # y1 = np.linspace(min(proby[:, 1]), max(proby[:, 1]), len(data.columns))  # position bins
        # grid_x, grid_y = np.meshgrid(x1, y1)
        # grid_prob_y = griddata(proby[:, 0:2], proby[:, 2], (grid_x, grid_y), method=method, fill_value=np.nan)
        #
        raw_data = data.iloc[start:stop].to_numpy().T
        bin_inds = np.digitize(data.iloc[start:stop].index, x1 + offset, right=True)
        if bin_inds[-1] == len(x1):  # if bins run up until past limit don't include
            bin_inds = bin_inds[:-1]
            raw_data = raw_data[:, :-1]

        grid_prob_y = np.empty((np.shape(raw_data)[0], len(x1)))
        grid_prob_y[:] = np.nan
        grid_prob_y[:, bin_inds] = raw_data

        grid_prob.append(grid_prob_y)

    return grid_prob


def griddata_2d_time_intervals(data, binsxy, times, start_locs, stop_locs, time_offset, nbins):
    grid_prob = []
    for start, stop, offset in zip(start_locs, stop_locs, time_offset):
        times_around_update = times[start:stop] - offset
        data_subset = data[start:stop, :, :]

        prob_data = []
        for ind, t in enumerate(times_around_update):
            df = pd.DataFrame(data_subset[ind, :, :], index=binsxy[0], columns=binsxy[1])
            data_around_update = df.stack().reset_index().values
            data_from_timepoint = np.hstack((np.tile(t, len(data_around_update))[:, None], data_around_update))

            prob_data.append(data_from_timepoint)

        all_data = np.vstack(prob_data)
        t1 = np.linspace(min(times_around_update), max(times_around_update), nbins)  # time bins
        x1 = np.linspace(min(binsxy[0]), max(binsxy[0]), np.shape(data)[1])  # x_position bins
        y1 = np.linspace(min(binsxy[1]), max(binsxy[1]), np.shape(data)[2])  # y_position bins
        grid_x, grid_y, grid_t = np.meshgrid(x1, y1, t1)

        grid_prob_trial = griddata(all_data[:, 0:3], all_data[:, 3], (grid_t, grid_x, grid_y), method='nearest',
                                   fill_value=np.nan)
        grid_prob.append(grid_prob_trial)

    return grid_prob