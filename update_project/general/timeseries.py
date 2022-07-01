import numpy as np
import pandas as pd

from bisect import bisect


def align_by_time_intervals(time_series, time_intervals, start_label='start_time', stop_label='stop_time',
                            start_window=0.0, stop_window=0.0):
    start_times = time_intervals[start_label][:]  # TODO - may need to change to DF
    stop_times = time_intervals[stop_label][:]

    ts_by_trial = []
    for start, stop in zip(start_times, stop_times):
        idx_start = bisect(time_series.timestamps, start+start_window)
        idx_stop = bisect(time_series.timestamps, stop_window, lo=idx_start)
        ts_by_trial.append(time_series.data[idx_start:idx_stop, :])

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
