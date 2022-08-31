import numpy as np
import pandas as pd

from update_project.virtual_track import UpdateTrack
from update_project.general.timeseries import align_by_time_intervals


def get_trials_dataframe(nwbfile, with_pseudoupdate=False):
    if with_pseudoupdate:
        return get_trials_with_pseudoupdate(nwbfile)
    else:
        return nwbfile.trials.to_dataframe()


def get_trials_with_pseudoupdate(nwbfile):
    # get track info
    virtual_track = UpdateTrack(linearization=False)

    # get trial data
    trials = nwbfile.trials.to_dataframe()
    mask = pd.concat([trials[k].isin(v) for k, v in dict(update_type=[1], maze_id=[4]).items()], axis=1).all(axis=1)
    update_mask = pd.concat([trials[k].isin(v) for k, v in dict(update_type=[2], maze_id=[4]).items()], axis=1).all(axis=1)
    trial_inds = np.array(trials.index[mask])
    trial_inds_update = np.array(trials.index[update_mask])
    position_data, vr_times = align_by_time_intervals(nwbfile.processing['behavior']['position']['position'],
                                                      nwbfile.intervals['trials'][trial_inds],
                                                      return_timestamps=True)
    position_data_update, vr_times_update = align_by_time_intervals(nwbfile.processing['behavior']['position']['position'],
                                                      nwbfile.intervals['trials'][trial_inds_update],
                                                      return_timestamps=True)

    # get the earliest update location occurring in this track to use as a threshold
    update_loc_default = virtual_track.cue_start_locations['y_position']['update cue']
    delay_loc_default = virtual_track.cue_start_locations['y_position']['delay2 cue']
    update_locs, delay_locs = [], []
    for update_pos, update_time, trial_ind in zip(position_data_update, vr_times_update, trial_inds_update):
        t_update = trials['t_update'].iloc[trial_ind]
        t_delay = trials['t_delay2'].iloc[trial_ind]
        update_locs.append(update_pos[np.searchsorted(update_time, t_update), 1])
        delay_locs.append(update_pos[np.searchsorted(update_time, t_delay), 1])
    if np.size(update_locs):
        update_loc = np.max([np.min(update_locs), update_loc_default])  # don't let it go lower than default value
        delay_loc = np.max([np.min(delay_locs), delay_loc_default])  # don't let it go lower than default value
    else:
        update_loc = update_loc_default
        delay_loc = delay_loc_default

    # get pseudo update and delay times
    t_update, t_delay2 = [], []
    for pos, time in zip(position_data, vr_times):
        update_ind_predict = np.argwhere(pos[:, 1] >= update_loc)[0][0]
        delay_ind_predict = np.argwhere(pos[:, 1] >= delay_loc)[0][0]

        t_update.append(time[update_ind_predict])
        t_delay2.append(time[delay_ind_predict])

    # test that grabbing the right times with known t_update (won't be 100% accurate but should be pretty accurate)
    predict_update_time, actual_update_time = [], []
    for update_pos, update_time, trial_ind in zip(position_data_update, vr_times_update, trial_inds_update):
        update_ind_predict = np.argwhere(update_pos[:, 1] >= update_loc)[0][0]
        predict_update_time.append(update_time[update_ind_predict])
        actual_update_time.append(trials['t_update'].iloc[trial_ind])

    assert sum(np.array(predict_update_time) == np.array(actual_update_time)) > 0.5*len(np.array(predict_update_time))

    # fill in trial data
    assert np.sum(trials[mask]['t_update'].isna()) == len(t_update)  # check that filling in right number of na values
    trials['t_update'].iloc[trial_inds] = t_update
    trials['t_delay2'].iloc[trial_inds] = t_delay2

    return trials
