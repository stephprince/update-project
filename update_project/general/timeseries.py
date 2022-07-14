import numpy as np
import pandas as pd

from bisect import bisect


def align_by_time_intervals(time_series, time_intervals, start_label='start_time', stop_label='stop_time',
                            start_window=0.0, stop_window=0.0, return_timestamps=False):
    start_times = time_intervals[start_label][:]
    stop_times = time_intervals[stop_label][:]

    if time_series.timestamps:
        timestamps = time_series.timestamps
    elif time_series.rate:
        timestamps = np.arange(0, len(time_series.data) / time_series.rate, 1 / time_series.rate)

    ts_by_trial = []
    timestamps_by_trial = []
    for start, stop in zip(start_times, stop_times):
        idx_start = bisect(timestamps, start+start_window)
        idx_stop = bisect(timestamps, stop+stop_window, lo=idx_start)
        if np.ndim(time_series.data) == 1:
            ts_by_trial.append(time_series.data[idx_start:idx_stop] * time_series.conversion)
        else:
            ts_by_trial.append(time_series.data[idx_start:idx_stop, :] * time_series.conversion)
        timestamps_by_trial.append(timestamps[idx_start:idx_stop])

    if return_timestamps:
        return ts_by_trial, timestamps_by_trial
    else:
        return ts_by_trial


def get_series_from_timeseries(timeseries):
    if timeseries.timestamps:
        timestamps = timeseries.timestamps
    elif timeseries.rate:
        timestamps = np.arange(0, len(timeseries.data) / timeseries.rate, 1 / timeseries.rate)

    series_list = []
    if timeseries.name == 'position':
        series_list.append(pd.Series(index=timestamps[:], data=timeseries.data[:, 0], name='x_position'))
        series_list.append(pd.Series(index=timestamps[:], data=timeseries.data[:, 1], name='y_position'))
    else:
        series_list.append(pd.Series(index=timestamps[:], data=timeseries.data[:], name=timeseries.name))

    return series_list
