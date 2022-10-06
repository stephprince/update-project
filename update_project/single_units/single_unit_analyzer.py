import numpy as np
import pickle
import pandas as pd
import pynapple as nap
import warnings

from bisect import bisect, bisect_left
from pathlib import Path
from pynwb import NWBFile

from update_project.results_io import ResultsIO
from update_project.virtual_track import UpdateTrack
from update_project.general.lfp import get_theta
from update_project.general.acquisition import get_velocity
from update_project.general.trials import get_trials_dataframe
from update_project.general.units import align_by_time_intervals as align_by_time_intervals_units
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts


class SingleUnitAnalyzer:
    def __init__(self, nwbfile: NWBFile, session_id: str, feature: str, params=dict()):
        # setup parameters
        self.units_types = params.get('units_types',
                                      dict(region=['CA1', 'PFC'],  # dict of filters to apply to units table
                                           cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))
        self.speed_threshold = params.get('speed_threshold', 1000)  # minimum virtual speed to subselect epochs
        self.firing_threshold = params.get('firing_threshold', 0)  # Hz, minimum peak firing rate of place cells to use
        self.encoder_trial_types = params.get('encoder_trial_types', dict(update_type=[1], correct=[0, 1]))  # trials
        self.encoder_bin_num = params.get('encoder_bin_num', 40)  # number of bins to build encoder
        self.linearized_features = params.get('linearized_features', ['y_position'])  # which features to linearize
        self.virtual_track = UpdateTrack(linearization=bool(self.linearized_features))
        self.limits = self.virtual_track.get_limits(feature)  # TODO - add limits for other features
        self.align_times = ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made']
        self.window = 2.5  # sec to look at aligned psth
        self.align_nbins = np.round((self.window * 2) / 25)  # hardcoded to match binsize of decoder too
        self.downsample_factor = 10  # downsample signal from 2000Hz to 200Hz

        # setup decoding/encoding functions based on dimensions
        self.encoder = nap.compute_1d_tuning_curves

        # setup file paths for io
        trial_types = [str(t) for t in self.encoder_trial_types['correct']]
        self.results_tags = f"{'_'.join(feature)}_regions_{'_'.join(self.units_types['region'])}_" \
                            f"enc_bins{self.encoder_bin_num}_speed_thresh" \
                            f"{self.speed_threshold}_trial_types{'_'.join(trial_types)}"
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path().absolute().stem,
                                    tags=self.results_tags)
        self.data_files = dict(single_unit_output=dict(vars=['spikes', 'tuning_curves', 'unit_selectivity',
                                                             'aligned_data', 'bins'],
                                                            format='pkl'),
                               params=dict(vars=['speed_threshold', 'firing_threshold', 'units_types',
                                                 'encoder_trial_types', 'encoder_bin_num', 'feature_name',
                                                 'linearized_features', 'downsample_factor', 'window'],
                                           format='npz'))

        # setup data
        self.feature_name = feature
        self.trials = get_trials_dataframe(nwbfile, with_pseudoupdate=True)
        self.units = nwbfile.units.to_dataframe()
        self.nwb_units = nwbfile.units
        self.data = self._setup_data(nwbfile)
        self.velocity = get_velocity(nwbfile)
        self.theta = get_theta(nwbfile, adjust_reference=True, session_id=session_id)

    def run(self, overwrite=False):
        print(f'Analyzing single unit data for session {self.results_io.session_id}...')

        if overwrite:
            self._preprocess()._encode()  # get tuning curves
            self._get_unit_selectivity()  # get selectivity index
            self._get_aligned_spikes()  # get spiking aligned to task events
            self._export_data()  # save output data
        else:
            if self._data_exists() and self._params_match():
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run(overwrite=True)

        return self

    def _setup_data(self, nwbfile):
        data_dict = dict()
        if self.feature_name in ['x_position', 'y_position']:
            time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
            if self.feature_name in self.linearized_features:
                raw_data = time_series.data[:]
                data = self.virtual_track.linearize_track_position(raw_data)
            else:
                column = [ind for ind, val in enumerate(['x_position', 'y_position']) if val == self.feature_name][0]
                data = time_series.data[:, column]
        elif self.feature_name in ['view_angle']:
            time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')
            data = time_series.data[:]
        elif self.feature_name in ['choice', 'turn_type']:
            choice_mapping = {'0': np.nan, '1': -1, '2': 1}  # convert to negative/non-negative for flipping
            time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

            data = np.empty(np.shape(time_series.data))
            data[:] = np.nan
            for n, trial in self.trials.iterrows():
                idx_start = bisect_left(time_series.timestamps, trial['start_time'])
                idx_stop = bisect(time_series.timestamps, trial['stop_time'], lo=idx_start)
                data[idx_start:idx_stop] = choice_mapping[
                    str(int(trial[self.feature_name]))]  # fill in values with the choice value

                # choice is the animals final decision but turn_type indicates what side is correct/incorrect
                # turn type will flip around the update
                if self.feature_name in ['turn_type'] and ~np.isnan(trial['t_update']) and trial['update_type'] == 2:
                    idx_switch = bisect_left(time_series.timestamps, trial['t_update'])
                    data[idx_switch:idx_stop] = -1 * choice_mapping[str(int(trial[feat]))]
        elif self.feature_name in ['dynamic_choice', 'cue_bias']:
            time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

            # load dynamic choice from saved output
            data_mapping = dict(dynamic_choice='choice', cue_bias='turn_type')
            fname_tag = data_mapping[self.feature_name]
            choice_path = Path().absolute().parent.parent / 'results' / 'dynamic_choice'
            fname = self.results_io.get_data_filename(filename=f'dynamic_choice_output_{fname_tag}',
                                                      results_type='session', format='pkl',
                                                      diff_base_path=choice_path)
            import_data = self.results_io.load_pickled_data(fname)
            for v, i_data in zip(['output_data', 'agg_data', 'decoder_data', 'params'], import_data):
                if v == 'decoder_data':
                    choice_data = i_data
            data = choice_data - 0.5  # convert to -0.5 and +0.5 for flipping between trials
        else:
            raise RuntimeError(f'{self.feature_name} feature is not currently supported')

        data_dict[self.feature_name] = pd.Series(index=time_series.timestamps[:], data=data)

        return pd.DataFrame.from_dict(data_dict)

    def _get_time_intervals(self, trial_starts, trial_stops):
        movement = self.velocity['combined'] > self.speed_threshold

        new_starts = []
        new_stops = []
        for start, stop in zip(trial_starts, trial_stops):
            # pull out trial specific time periods
            start_ind = bisect(movement.index, start)
            stop_ind = bisect_left(movement.index, stop)
            movement_by_trial = movement.iloc[start_ind:stop_ind]

            # get intervals where movement crosses speed threshold
            breaks = np.where(np.hstack((True, np.diff(movement_by_trial), True)))[0]
            breaks[-1] = breaks[-1] - 1  # subtract one index to get closest to final trial stop time
            index = pd.IntervalIndex.from_breaks(movement_by_trial.index[breaks])
            movements_df = pd.DataFrame(movement_by_trial[index.left].to_numpy(), index=index, columns=['moving'])

            # append new start/stop times to data struct
            new_times = movements_df[movements_df['moving'] == True].index
            new_starts.append(new_times.left.values)
            new_stops.append(new_times.right.values)

        times = nap.IntervalSet(start=np.hstack(new_starts), end=np.hstack(new_stops), time_units='s')

        return times

    def _params_match(self):
        params_path = self.results_io.get_data_filename(f'params', results_type='session', format='npz')
        params_cached = np.load(params_path, allow_pickle=True)
        params_matched = []
        for k, v in params_cached.items():
            params_matched.append(getattr(self, k) == v)

        return all(params_matched)

    def _data_exists(self):
        files_exist = []
        for name, file_info in self.data_files.items():
            path = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])
            files_exist.append(path.is_file())

        return all(files_exist)

    def _preprocess(self):
        # get spikes
        units_mask = pd.concat([self.units[k].isin(v) for k, v in self.units_types.items()], axis=1).all(axis=1)
        units_subset = self.units[units_mask]
        spikes_dict = {n: nap.Ts(t=units_subset.loc[n, 'spike_times'], time_units='s') for n in units_subset.index}
        if spikes_dict:
            self.spikes = nap.TsGroup(spikes_dict, time_units='s')
        else:
            self.spikes = []  # if no units match criteria, leave spikes empty

        # get data based on trial times
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.encoder_trial_types.items()], axis=1).all(axis=1)
        encoder_trials = self.trials[mask]
        self.train_df = encoder_trials.sort_index()

        self.encoder_times = self._get_time_intervals(self.train_df['start_time'], self.train_df['stop_time'])
        self.features_train = nap.TsdFrame(self.data, time_units='s', time_support=self.encoder_times)
        self.theta = nap.TsdFrame(self.theta.iloc[::self.downsample_factor, :], time_units='s')
        self.velocity = nap.TsdFrame(self.velocity.iloc[::self.downsample_factor, :], time_units='s')

        return self

    def _encode(self):
        if self.spikes:  # if there were units and spikes to use for encoding
            feat_input = self.features_train[self.feature_name]
            self.bins = np.linspace(self.limits[0], self.limits[-1], self.encoder_bin_num + 1)
            self.tuning_curves = self.encoder(group=self.spikes, feature=feat_input, nb_bins=self.encoder_bin_num,
                                              ep=self.encoder_times)
        else:  # if there were no units/spikes to use for encoding, create empty dataframe
            self.model = pd.DataFrame()
            self.bins = []

        return self

    def _get_unit_selectivity(self):
        # get goal selective cells (cells with a place field in at least one of the choice locations)
        place_field_thresholds = self.tuning_curves.apply(lambda x: x > (np.mean(x) + np.std(x)))
        place_fields_2_bins = place_field_thresholds.rolling(window=2).mean() > 0.5
        bins_shifted = place_fields_2_bins.shift(periods=-1, axis=0,)
        bins_shifted.iloc[-1, :] = place_fields_2_bins.iloc[-1, :]
        place_fields = np.logical_or(place_fields_2_bins, bins_shifted).astype(bool)

        bounds = list(self.virtual_track.choice_boundaries.get(self.feature_name).values())
        goal_selective_bool = place_fields.apply(lambda x: x.loc[bounds[0][0]:bounds[1][1]].any())  # get all goal
        goal_selective_cells = self.tuning_curves.loc[:, goal_selective_bool]

        # get left/right selective (goal selective + how much prefer one goal location to the other)
        selectivity_index = goal_selective_cells.apply(lambda x: self._calc_selectivity_index(x, bounds)).mean()
        selectivity_index.name = 'selectivity_index'

        self.unit_selectivity = pd.merge(self.units[['region', 'cell_type']], selectivity_index,
                                         left_index=True, right_index=True, how='left')
        self.unit_selectivity['unit_id'] = self.unit_selectivity.index
        self.unit_selectivity['place_field_threshold'] = self.tuning_curves.apply(lambda x: np.mean(x) + np.std(x))

        return self

    @staticmethod
    def _calc_selectivity_index(x, bounds):
        mean_fr = [np.nanmean(x.loc[b[0]:b[1]]) for b in bounds]
        max_fr = [np.nanmax(x.loc[b[0]:b[1]]) for b in bounds]

        mean_selectivity_index = (mean_fr[0] - mean_fr[1]) / (mean_fr[0] + mean_fr[1])
        max_selectivity_index = (max_fr[0] - max_fr[1]) / (max_fr[0] + max_fr[1])

        return pd.Series([mean_selectivity_index, max_selectivity_index],
                         index=['mean_selectivity_index', 'max_selectivity_index'])

    def _get_aligned_spikes(self):
        units_aligned = []
        ts_aligned = []
        for time_label in self.align_times:  # skip last align times so only until stop of trial
            window_start, window_stop = self._get_event_window(time_label)
            new_times = np.linspace(window_start, window_stop,
                                    int(np.round(self.theta['phase'].rate * (window_stop - window_start))))

            for unit_index in range(len(self.nwb_units)):
                units_aligned.append(dict(trial_ids=self.trials.index.to_numpy(),
                                          time_label=time_label,
                                          unit_id=self.nwb_units.id[unit_index],
                                          spikes=align_by_time_intervals_units(self.nwb_units, unit_index, self.trials,
                                                                               start_label=time_label,
                                                                               stop_label=time_label,
                                                                               start=window_start, end=window_stop),
                                          ))

            vars = dict(theta_phase=self.theta['phase'], theta_amplitude=self.theta['amplitude'],
                        rotational_velocity=self.velocity['rotational'])
            dig_data = dict()
            for v_name, v_data in vars.items():
                dig_data[v_name], timestamps = align_by_time_intervals_ts(v_data, self.trials, return_timestamps=True,
                                                                     start_label=time_label, stop_label=time_label,
                                                                     start_window=window_start, stop_window=window_stop)
            ts_aligned.append(dict(**dig_data, timestamps=timestamps, new_times=new_times, time_label=time_label,
                                   turn_type=self.trials['turn_type'].to_numpy(),
                                   outcomes=self.trials['correct'].to_numpy()))

        aligned_data = (pd.merge(pd.DataFrame(units_aligned), pd.DataFrame(ts_aligned), on='time_label')
                        .explode(['trial_ids', 'spikes',  *list(vars.keys()), 'timestamps', 'turn_type', 'outcomes'])
                        .reset_index(drop=True))

        self.aligned_data = pd.merge(aligned_data, self.unit_selectivity, on='unit_id')

    def _get_event_window(self, time_label):
        window_start, window_stop = self.window, self.window
        if time_label == 't_choice_made':
            window_stop = 0  # if choice made, don't grab data past bc could be end of trial
        elif time_label == 'start_time':
            window_start = 0

        return -window_start, window_stop

    def _load_data(self):
        print(f'Loading existing data for session {self.results_io.session_id}...')

        # load npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            if file_info['format'] == 'npz':
                import_data = np.load(fname, allow_pickle=True)
                for v in file_info['vars']:
                    setattr(self, v, import_data[v])
            elif file_info['format'] == 'pkl':
                import_data = self.results_io.load_pickled_data(fname)
                for v, data in zip(file_info['vars'], import_data):
                    setattr(self, v, data)
            else:
                raise RuntimeError(f'{file_info["format"]} format is not currently supported for loading data')

        return self

    def _export_data(self):
        print(f'Exporting data for session {self.results_io.session_id}...')

        # save npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            if file_info['format'] == 'npz':
                kwargs = {v: getattr(self, v) for v in file_info['vars']}
                np.savez(fname, **kwargs)
            elif file_info['format'] == 'pkl':
                with open(fname, 'wb') as f:
                    [pickle.dump(getattr(self, v), f) for v in file_info['vars']]
            else:
                raise RuntimeError(f'{file_info["format"]} format is not currently supported for exporting data')
