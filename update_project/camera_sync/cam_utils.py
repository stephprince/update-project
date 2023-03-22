# import libraries
import numpy as np
from scipy.io import loadmat, matlab
from datetime import timedelta, datetime

# define general functions
def mat_obj_to_dict(mat_struct):
    """Recursive function to convert nested matlab struct objects to dictionaries."""
    dict_from_struct = {}
    for field_name in mat_struct.__dict__['_fieldnames']:
        dict_from_struct[field_name] = mat_struct.__dict__[field_name]
        if isinstance(dict_from_struct[field_name], matlab.mio5_params.mat_struct):
            dict_from_struct[field_name] = mat_obj_to_dict(dict_from_struct[field_name])
        elif isinstance(dict_from_struct[field_name], np.ndarray):
            try:
                dict_from_struct[field_name] = mat_obj_to_array(dict_from_struct[field_name])
            except TypeError:
                continue
    return dict_from_struct

def mat_obj_to_array(mat_struct_array):
    """Construct array from matlab cell arrays.
    Recursively converts array elements if they contain mat objects."""
    if has_struct(mat_struct_array):
        array_from_cell = [mat_obj_to_dict(mat_struct) for mat_struct in mat_struct_array]
        array_from_cell = np.array(array_from_cell)
    else:
        array_from_cell = mat_struct_array

    return array_from_cell


def has_struct(mat_struct_array):
    """Determines if a matlab cell array contains any mat objects."""
    return any(
        isinstance(mat_struct, matlab.mio5_params.mat_struct) for mat_struct in mat_struct_array)

def convert_mat_file_to_dict(mat_file_name):
    """
    Convert mat-file to dictionary object.
    It calls a recursive function to convert all entries
    that are still matlab objects to dictionaries.
    """
    data = loadmat(mat_file_name, struct_as_record=False, squeeze_me=True)
    for key in data:
        if isinstance(data[key], matlab.mio5_params.mat_struct):
            data[key] = mat_obj_to_dict(data[key])
    return data

def matlab_time_to_datetime(series):
    times = datetime.fromordinal(int(series)) + \
            timedelta(days=series % 1) - \
            timedelta(days=366)
    return times

def get_trial_intervals(virmen_df, camera_trig, trial_type="first_last", event='trial', n_trials=None,
                        time_window=2.5, turn_type=1, sort=True):
    # get trial start and end indices
    task_state_names = (
        'trial_start', 'initial_cue', 'update_cue', 'delay_cue', 'choice_made', 'reward', 'trial_end', 'inter_trial')
    task_state_dict = dict(zip(task_state_names, range(1, 9)))

    trial_starts = virmen_df.index[virmen_df.taskState == task_state_dict['trial_start']].to_numpy() + 1
    trial_ends = virmen_df.index[virmen_df.taskState == task_state_dict['trial_end']].to_numpy()
    if trial_starts[0] > trial_ends[0]:
        trial_ends = trial_ends[1:]
    if trial_ends[-1] < trial_starts[-1]:
        trial_starts = trial_starts[:-1]

    # only use trials of one turn type
    turn_types = virmen_df['trialType'][trial_starts].to_numpy()
    trial_starts = trial_starts[turn_types == turn_type]
    trial_ends = trial_ends[turn_types == turn_type]

    # only use trials during which camera was on
    trial_starts = trial_starts[np.logical_and(trial_starts >= camera_trig[0], trial_starts <= camera_trig[-1])]
    trial_ends = trial_ends[np.logical_and(trial_ends >= camera_trig[0], trial_ends <= camera_trig[-1])]
    trial_intervals_all = list(zip(trial_starts, trial_ends))

    # get event times
    virmen_time = virmen_df["time"].apply(lambda x: matlab_time_to_datetime(x))
    virmen_time_elapsed = [(t - virmen_time[0]).total_seconds() for t in virmen_time]
    updates = np.where(np.diff(virmen_df['updateOccurred'], prepend=0) == 1)[0]
    delays = np.where(np.diff(virmen_df['delayUpdateOccurred'], prepend=0) == 1)[0]
    event_starts, event_ends = [], []
    t_delay_from_update = []
    for start, end in trial_intervals_all:
        temp = updates[np.logical_and(updates > start, updates < end)]
        temp_delay = delays[np.logical_and(delays > start, delays < end)]
        if 0 < len(temp):
            i_event = temp[0]
            t_event = virmen_time_elapsed[i_event]
            event_starts.append(np.searchsorted(virmen_time_elapsed, t_event - time_window) - 1)
            event_ends.append(np.searchsorted(virmen_time_elapsed, t_event + time_window))

            t_delay_from_update.append(virmen_time_elapsed[temp_delay[0]] - t_event)

    event_intervals_all = list(zip(event_starts, event_ends))

    if event == 'trial':
        intervals = trial_intervals_all
    elif event == 'update':
        intervals = event_intervals_all

    # select examples trials
    if trial_type == "first_last":
        # first/last
        n_trials = 3
        ts_first = trial_starts[0:n_trials]
        te_first = trial_ends[trial_ends > trial_starts[0]][0:n_trials]
        ts_last = trial_starts[trial_starts < trial_ends[-1]][-n_trials - 1:-1]
        te_last = trial_ends[-n_trials - 1:-1]

        trial_intervals = list(zip(np.concatenate((ts_first, ts_last)), np.concatenate((te_first, te_last))))
    elif trial_type == "switch":
        # update only
        update = virmen_df.index[virmen_df.trialTypeUpdate == 2].to_list()
        trial_intervals, delays = [], []
        for i, interval in enumerate(intervals):
            if any(interval[0] <= n <= interval[1] for n in update):
                trial_intervals.append(interval)
                delays.append(t_delay_from_update[i])
    elif trial_type == "stay":
        # stay only
        stay = virmen_df.index[virmen_df.trialTypeUpdate == 2].to_list()
        trial_intervals = []
        for i, interval in enumerate(intervals):
            if any(interval[0] <= n <= interval[1] for n in stay):
                trial_intervals.append(interval)

    if sort:
        trial_intervals = [interval for _, interval in sorted(zip(delays, trial_intervals))]

    if n_trials:
        trial_intervals = trial_intervals[:n_trials] #get only the first X trials, default 5

    return trial_intervals

