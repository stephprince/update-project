import pynwb
import numpy as np

def align_by_times(units: pynwb.misc.Units, index, starts, stops):
    """
    Args:
        units: pynwb.misc.Units
        index: int
        starts: array-like
        stops: array-like
    Returns:
        np.array
    """

    st = units["spike_times"]
    unit_spike_data = st[index]

    istarts = np.searchsorted(unit_spike_data, starts)
    istops = np.searchsorted(unit_spike_data, stops)
    for start, istart, istop in zip(starts, istarts, istops):
        yield unit_spike_data[istart:istop] - start


def align_by_time_intervals(
    units: pynwb.misc.Units,
    index,
    intervals,
    start_label="start_time",
    stop_label="stop_time",
    start=0.0,
    end=0.0,
    rows_select=(),
):
    """
    Args:
        units: time-aware neurodata_type
        index: int
        intervals: pynwb.epoch.TimeIntervals
        start_label: str
            default: 'start_time'
        stop_label: str
            default: 'stop_time'
        start: float
            Start time for calculation before or after (negative or positive) the reference point (aligned to).
        end: float
            End time for calculation before or after (negative or positive) the reference point (aligned to).
        rows_select: array_like, optional
            sub-selects specific rows
    Returns:
        np.array(shape=(n_trials, n_time, ...))
    """
    if stop_label is None:
        stop_label = start_label
    starts = np.squeeze(np.array(intervals[start_label][:])[rows_select] + start)
    stops = np.squeeze(np.array(intervals[stop_label][:])[rows_select] + end)

    out = []
    for i, x in enumerate(align_by_times(units, index, starts, stops)):
        out.append(x + start)

    return out
