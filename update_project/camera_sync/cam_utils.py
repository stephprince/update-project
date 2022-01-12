# import libraries
import numpy as np
from scipy.io import loadmat, matlab

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


def get_trial_intervals(virmen_df, camera_trig, trial_type="first_last", num_trials=5):
    # get trial start and end indices
    task_state_names = (
        'trial_start', 'initial_cue', 'update_cue', 'delay_cue', 'choice_made', 'reward', 'trial_end', 'inter_trial')
    task_state_dict = dict(zip(task_state_names, range(1, 9)))

    trial_starts = virmen_df.index[virmen_df.taskState == task_state_dict['trial_start']].to_numpy()
    trial_ends = virmen_df.index[virmen_df.taskState == task_state_dict['trial_end']].to_numpy()

    # only use trials during which camera was on
    trial_starts = trial_starts[np.logical_and(trial_starts >= camera_trig[0], trial_starts <= camera_trig[-1])]
    trial_ends = trial_ends[np.logical_and(trial_ends >= camera_trig[0], trial_ends <= camera_trig[-1])]
    trial_intervals_all = list(zip(trial_starts, trial_ends))

    # select examples trials
    if trial_type == "first_last":
        # first/last
        n_trials = 3
        ts_first = trial_starts[0:n_trials]
        te_first = trial_ends[trial_ends > trial_starts[0]][0:n_trials]
        ts_last = trial_starts[trial_starts < trial_ends[-1]][-n_trials - 1:-1]
        te_last = trial_ends[-n_trials - 1:-1]

        trial_intervals = list(zip(np.concatenate((ts_first, ts_last)), np.concatenate((te_first, te_last))))
    elif trial_type == "update":
        # update only
        update = virmen_df.index[virmen_df.trialTypeUpdate == 2].to_list()
        trial_intervals = []
        for i, interval in enumerate(trial_intervals_all):
            if any(interval[0] <= n <= interval[1] for n in update):
                trial_intervals.append(interval)
    elif trial_type == "stay":
        # stay only
        stay = virmen_df.index[virmen_df.trialTypeUpdate == 2].to_list()
        trial_intervals = []
        for i, interval in enumerate(trial_intervals_all):
            if any(interval[0] <= n <= interval[1] for n in stay):
                trial_intervals.append(interval)

    trial_intervals = trial_intervals[:num_trials] #get only the first X trials, default 5

    return trial_intervals