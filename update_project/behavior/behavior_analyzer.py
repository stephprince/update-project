import numpy as np
import pickle
import pandas as pd
import warnings

from pathlib import Path
from pynwb import NWBFile

from update_project.results_io import ResultsIO
from update_project.virtual_track import UpdateTrack
from update_project.general.timeseries import get_series_from_timeseries
from update_project.general.trials import get_trials_dataframe
from update_project.decoding.interpolate import interp_timeseries


class BehaviorAnalyzer:
    def __init__(self, nwbfile: NWBFile, session_id: str, params: dict = {}):
        # setup params
        self.virtual_track = UpdateTrack()
        self.virmen_vars = params.get('behavioral_vars', ['position', 'view_angle', 'rewards'])
        self.analog_vars = params.get('analog_vars', ['rotational_velocity', 'translational_velocity', 'licks'])
        self.maze_ids = params.get('maze_ids', [4])  # which virtual environments to use for analysis
        self.position_bins = params.get('position_bins', 50)  # number of bins to use for virtual track
        self.trial_window = params.get('trial_window', 20)  # number of trials to use for rolling calculations
        self.align_window_start = params.get('align_window_start', -3)  # seconds to add before/after aligning window
        self.align_window_stop = params.get('align_window_stop', 3)  # seconds to add before/after aligning window
        self.align_times = params.get('align_times', ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made',
                                                      'stop_time'])

        # setup data
        self.trials = self._setup_trials(nwbfile)
        self.data = self._setup_data(nwbfile)

        # setup I/O
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path().absolute().stem)
        self.data_files = dict(behavior_output=dict(vars=['proportion_correct', 'aligned_data', 'trajectories'],
                                                    format='pkl'))

    def run_analysis(self, overwrite=False):
        print(f'Analyzing behavioral data for {self.results_io.session_id}...')

        if overwrite:
            self._get_proportion_correct()
            self._get_trajectories()
            self._align_data()
            self._get_event_durations()
            self._export_data()  # save output data
        else:
            if self.results_io.data_exists(self.data_files):
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run_analysis(overwrite=True)

    def _setup_trials(self, nwbfile):
        all_trials = get_trials_dataframe(nwbfile, with_pseudoupdate=True)
        trials = all_trials[all_trials['maze_id'].isin(self.maze_ids)]

        return trials

    def _setup_data(self, nwbfile):
        data = dict()
        for an in self.analog_vars:
            data[an] = nwbfile.acquisition[an]

        for vr in self.virmen_vars:
            if nwbfile.processing['behavior'][vr].neurodata_type in ['BehavioralEvents']:
                data[vr] = nwbfile.processing['behavior'][vr].get_timeseries(vr)
            else:
                data[vr] = nwbfile.processing['behavior'][vr].get_spatial_series(vr)

        return data

    def _get_proportion_correct(self):
        proportion_correct = []
        groups = self.trials.groupby(['update_type'])
        for name, group_data in groups:
            data = group_data.reset_index(drop=True)  # TODO - determine if I want to have min bin length to use data
            rolling = data['correct'].rolling(self.trial_window, min_periods=self.trial_window).mean()
            binned = data['correct'].groupby(data['correct'].index // self.trial_window).mean()
            if len(data['correct']) % self.trial_window:  # if any leftover trials getting included in last bin
                binned = binned[:-1]

            proportion_correct.append(dict(prop_correct=rolling.values,
                                           type='rolling',
                                           update_type=self.virtual_track.mappings['update_type'][str(name)]))
            proportion_correct.append(dict(prop_correct=binned.values,
                                           type='binned',
                                           update_type=self.virtual_track.mappings['update_type'][str(name)]))

        self.proportion_correct = proportion_correct

    def _get_trajectories(self):
        trajectory_df = pd.DataFrame(index=self.data['position'].timestamps, data=self.data['position'].data[:, 1],
                                     columns=['y_position'])
        y_limits = self.virtual_track.get_limits('y_position')
        position_bins = np.linspace(y_limits[0], y_limits[1], self.position_bins + 1)

        # get dataframe of data
        for key, value in self.data.items():
            # get only times during trials
            series_list = get_series_from_timeseries(value)
            for s in series_list:
                if value.timestamps:  # if the data is from virmen with timestamps/uneven sampling
                    trajectory_df[s.name] = s
                elif value.rate:  # if the data is from analog
                    # match nearest analog sample to virmen sample
                    trajectory_df = pd.merge_asof(trajectory_df, s, left_index=True, right_index=True)

        # get average for each position bin for each trial
        trajectories = []
        groups = self.trials.groupby(['update_type', 'turn_type'])
        for name, group_data in groups:
            for ind, trial in group_data.iterrows():
                start_loc = trajectory_df.index.searchsorted(trial['start_time'])
                stop_loc = trajectory_df.index.searchsorted(trial['stop_time'])
                trial_df = trajectory_df.iloc[start_loc:stop_loc, :]

                bins = pd.cut(trial_df['y_position'], position_bins)
                agg_data = trial_df.groupby(bins).agg([lambda x: np.nanmean(x)])
                agg_data.columns = trial_df.columns.values
                trajectories.append(dict(**dict(trial_id=ind,
                                                update_type=self.virtual_track.mappings['update_type'][str(name[0])],
                                                turn_type=self.virtual_track.mappings['turn_type'][str(name[1])]),
                                         **agg_data.to_dict()))

        self.trajectories = trajectories

    def _align_data(self):
        # make line plot of position throughout the trial
        aligned_data = []
        for key, value in self.data.items():
            for ind, label in enumerate(self.align_times[:-1]):  # skip last align times so only until stop of trial
                start_label = label
                # stop_label = self.align_times[ind + 1]
                series_list = get_series_from_timeseries(value)
                for s in series_list:
                    groups = self.trials.groupby(['update_type', 'turn_type'])
                    for name, group_data in groups:
                        align_window_stop = self.align_window_stop
                        if start_label == 't_choice_made':
                            align_window_stop = 0  # if choice made, don't grab data past bc could be end of trial
                        data, times = interp_timeseries(s, group_data, start_label=start_label, stop_label=start_label,
                                                        start_window=self.align_window_start, stop_window=align_window_stop)
                        aligned_data.append(dict(var=s.name,
                                                 start_label=start_label,
                                                 stop_label=start_label,
                                                 aligned_data=data,
                                                 aligned_times=times,
                                                 update_type=self.virtual_track.mappings['update_type'][str(name[0])],
                                                 turn_type=self.virtual_track.mappings['turn_type'][str(name[1])]))

        self.aligned_data = aligned_data

    def _get_event_durations(self):
        # get event durations from trials table
        initial_cue = self.trials['t_delay'] - self.trials['start_time']
        delay1 = self.trials['t_update'] - self.trials['t_delay']
        update = self.trials['t_delay2'] - self.trials['t_update']
        delay2 = self.trials['t_choice_made'] - self.trials['t_delay2']
        total_trial = self.trials['t_choice_made'] - self.trials['start_time']  # go to choice_made to not include 3s freeze after reward on correct trials

        durations = pd.concat([initial_cue, delay1, update, delay2, total_trial], axis=1)
        durations.columns = ['initial_cue', 'delay1', 'update', 'delay2', 'total_trial']

        # replace delay times with full delay length for non-update trials since actually no update cue
        non_update_bool = (self.trials['update_type'] == 1)
        durations['delay1'][non_update_bool] = delay1 + update + delay2
        durations['update'][non_update_bool] = np.nan
        durations['delay2'][non_update_bool] = np.nan

        self.event_durations = pd.concat([self.trials[['maze_id', 'turn_type', 'update_type', 'correct']], durations],
                                         axis=1)

    def _load_data(self):
        print(f'Loading existing data for session {self.results_io.session_id}...')

        # load npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            import_data = self.results_io.load_pickled_data(fname)
            for v, data in zip(file_info['vars'], import_data):
                setattr(self, v, data)

        return self

    def _export_data(self):
        print(f'Exporting data for session {self.results_io.session_id}...')

        # save npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            with open(fname, 'wb') as f:
                [pickle.dump(getattr(self, v), f) for v in file_info['vars']]
