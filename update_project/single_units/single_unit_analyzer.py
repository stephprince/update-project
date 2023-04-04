import numpy as np
import pandas as pd
import pynapple as nap
import warnings

from bisect import bisect, bisect_left
from pathlib import Path
from pynwb import NWBFile
from scipy import signal
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate

from update_project.general.results_io import ResultsIO
from update_project.general.virtual_track import UpdateTrack
from update_project.general.lfp import get_theta
from update_project.general.acquisition import get_velocity
from update_project.general.place_cells import get_place_fields, get_largest_field_loc
from update_project.general.trials import get_trials_dataframe
from update_project.general.units import align_by_time_intervals as align_by_time_intervals_units
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts
from update_project.choice.commitment_quartiles import get_commitment_quartiles
from update_project.base_analysis_class import BaseAnalysisClass


class SingleUnitAnalyzer(BaseAnalysisClass):
    def __init__(self, nwbfile: NWBFile, session_id: str, feature: str, params=dict()):
        # setup parameters
        self.units_types = params.get('units_types',
                                      dict(region=['CA1', 'PFC'],  # dict of filters to apply to units table
                                           cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))
        self.speed_threshold = params.get('speed_threshold', 1000)  # minimum virtual speed to subselect epochs
        self.firing_threshold = params.get('firing_threshold', 0)  # Hz, minimum peak firing rate of place cells to use
        self.encoder_trial_types = params.get('encoder_trial_types', dict(maze_id=[4], update_type=[1], correct=[0, 1]))  # trials
        self.switch_trial_types = params.get('switch_trial_types', dict(maze_id=[4], update_type=[2], correct=[0, 1]))  # trials
        self.stay_trial_types = params.get('stay_trial_types', dict(maze_id=[4], update_type=[3], correct=[0, 1]))  # trials
        self.encoder_bin_num = params.get('encoder_bin_num', 50)  # number of bins to build encoder
        self.linearized_features = params.get('linearized_features', ['y_position'])  # which features to linearize
        self.virtual_track = UpdateTrack(linearization=bool(self.linearized_features))
        self.align_times = params.get('align_times', ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made'])
        self.window = params.get('align_window', 2.5)  # number of bins to build encoder  # sec to look at aligned psth
        self.align_nbins = np.round((self.window * 2) / 25)  # hardcoded to match binsize of decoder too
        self.downsample_factor = 40  # downsample signal from 2000Hz to 50Hz
        self.goal_selectivity_strictness = 'goal_field'  # options are 'only_goal_field' (no fields in rest of track)
                                                         # or 'goal_field' (can have other fields in rest of track)

        # setup decoding/encoding functions based on dimensions
        self.encoder = nap.compute_1d_tuning_curves

        # setup file paths for io
        trial_types = [str(t) for t in self.encoder_trial_types['correct']]
        self.results_tags = f"{'_'.join(feature)}_regions_{'_'.join(self.units_types['region'])}_" \
                            f"enc_bins{self.encoder_bin_num}_speed_thresh" \
                            f"{self.speed_threshold}_trial_types{'_'.join(trial_types)}"
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path(__file__).parent.stem,
                                    tags=self.results_tags)
        self.data_files = dict(single_unit_output=dict(vars=['spikes', 'tuning_curves', 'cycle_skipping',
                                                             'goal_selectivity', 'update_selectivity', 'aligned_data',
                                                             'bins','trials', 'train_df'],
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
        self.commitment = get_commitment_quartiles(session_id=session_id)
        self.theta = get_theta(nwbfile, adjust_reference=True, session_id=session_id)
        self.bounds = list(self.virtual_track.choice_boundaries.get(self.feature_name).values())
        self.limits = self.virtual_track.get_limits(feature)

    def run_analysis(self, overwrite=False, export_data=True):
        print(f'Analyzing single unit data for session {self.results_io.session_id}...')

        if overwrite:
            self._preprocess()._encode()  # get tuning curves
            self._get_theta_cycle_skipping()  # get theta cycle skipping index
            self._get_goal_selectivity()  # get selectivity index
            self._get_aligned_spikes()  # get spiking aligned to task events
            self._get_update_selectivity()  # get units that respond
            if export_data:
                self._export_data()  # save output data
        else:
            if self._data_exists() and self._params_match():
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run_analysis(overwrite=True, export_data=export_data)

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
        elif self.feature_name in ['choice', 'cue_bias']:
            time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

            # load dynamic choice from saved output
            data_mapping = dict(dynamic_choice='choice', cue_bias='turn_type')
            fname_tag = data_mapping[self.feature_name]
            choice_path = Path().absolute().parent.parent / 'results' / 'choice'
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

    def _get_time_intervals(self, trial_starts, trial_stops, min_duration=None):
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

        if np.size(new_starts):
            start = np.hstack(new_starts)
            end = np.hstack(new_stops)
        else:
            start, end = [], []

        if min_duration and np.size(start):
            start, end = zip(*[(s, e) for s, e in zip(start, end) if (e - s) > min_duration])

        times = nap.IntervalSet(start=start, end=end, time_units='s')

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
        self.units_subset = self.units[units_mask]
        spikes_dict = {n: nap.Ts(t=self.units_subset.loc[n, 'spike_times'], time_units='s') for n in self.units_subset.index}
        if spikes_dict:
            self.spikes = nap.TsGroup(spikes_dict, time_units='s')
        else:
            self.spikes = []  # if no units match criteria, leave spikes empty

        # get data based on trial times
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.encoder_trial_types.items()], axis=1).all(axis=1)
        encoder_trials = self.trials[mask]
        self.train_df = encoder_trials.sort_index()

        # update trials
        trials_with_delay = self.trials.dropna(subset=['t_delay2', 't_update'])
        self.encoder_times = self._get_time_intervals(self.train_df['start_time'], self.train_df['stop_time'])
        self.delay_trial_times = self._get_time_intervals(trials_with_delay.query('update_type == 1')['t_update'],
                                                          trials_with_delay.query('update_type == 1')['t_delay2'] + 2.5,
                                                          min_duration=1.5)
        self.switch_trial_times = self._get_time_intervals(trials_with_delay.query('update_type == 2')['t_update'],
                                                           trials_with_delay.query('update_type == 2')['t_delay2'] + 2.5,
                                                           min_duration=1.5)
        self.stay_trial_times = self._get_time_intervals(trials_with_delay.query('update_type == 3')['t_update'],
                                                         trials_with_delay.query('update_type == 3')['t_delay2'] + 2.5,
                                                         min_duration=1.5)
        self.commitment_trial_times = dict()
        for q_name, q_trials in self.commitment.groupby('view_angle_quantile'):
            trial_times = self._get_time_intervals(q_trials.query('update_type == 2')['t_update'],
                                                   q_trials.query('update_type == 2')['t_delay2'],
                                                   min_duration=1.5)
            self.commitment_trial_times[q_name] = trial_times

        self.features_train = nap.TsdFrame(self.data, time_units='s', time_support=self.encoder_times)
        self.theta = nap.TsdFrame(self.theta.iloc[::self.downsample_factor, :], time_units='s')
        self.velocity = nap.TsdFrame(self.velocity.iloc[::self.downsample_factor, :], time_units='s')

        return self

    def _encode(self):
        if self.spikes:  # if there were units and spikes to use for encoding
            feat_input = self.features_train[self.feature_name]
            self.bins = np.linspace(self.limits[0], self.limits[-1], self.encoder_bin_num + 1)
            self.tuning_curves = self.encoder(group=self.spikes, feature=feat_input, nb_bins=self.encoder_bin_num,
                                              ep=self.encoder_times, minmax=self.limits)
        else:  # if there were no units/spikes to use for encoding, create empty dataframe
            self.tuning_curves = pd.DataFrame()
            self.bins = []

        return self

    def _get_goal_selectivity(self):
        if np.size(self.tuning_curves):
            place_fields = get_place_fields(tuning_curves=self.tuning_curves)

            # get cells with place fields in goal arms but no fields in other parts of the track
            bounds_bool_1 = place_fields.apply(
                lambda x: x.loc[self.bounds[0][0]:self.bounds[0][1]].any())  # get all goal
            bounds_bool_2 = place_fields.apply(
                lambda x: x.loc[self.bounds[1][0]:self.bounds[1][1]].any())  # get all goal

            if self.goal_selectivity_strictness == 'only_goal_field':
                all_bounds = np.hstack([place_fields.loc[self.bounds[0][0]:self.bounds[0][1]].index,
                                        place_fields.loc[self.bounds[1][0]:self.bounds[1][1]].index])
                out_of_bounds_bool = place_fields.apply(lambda x: x[~x.index.isin(all_bounds)].any())
                goal_selective_bool = np.logical_and(np.logical_or(bounds_bool_1, bounds_bool_2),
                                                     ~out_of_bounds_bool).astype(bool)
            else:
                goal_selective_bool = np.logical_or(bounds_bool_1, bounds_bool_2).astype(bool)

            goal_selective_cells = self.tuning_curves.loc[:, goal_selective_bool]

            # get left/right selective (goal selective + how much prefer one goal location to the other)
            selectivity_index = goal_selective_cells.apply(lambda x: self._calc_selectivity_index(x, self.bounds))
            selectivity_index = selectivity_index.transpose()

            self.goal_selectivity = pd.merge(self.units_subset[['region', 'cell_type']], selectivity_index,
                                             left_index=True, right_index=True, how='left')
            self.goal_selectivity['unit_id'] = self.goal_selectivity.index
            self.goal_selectivity['place_field_threshold'] = self.tuning_curves.apply(lambda x: np.mean(x) + np.std(x))
            self.goal_selectivity['place_field_peak_ind'] = place_fields.apply(lambda x: get_largest_field_loc(x),
                                                                               axis=0)
        else:
            self.goal_selectivity = pd.DataFrame()

        return self

    @staticmethod
    def _calc_selectivity_index(x, bounds):
        # based on Kay, Frank Neuron 2020 paper for goal-selective cells (but more stringent bc must have a place field
        # only within the goal zone
        mean_fr = [np.nanmean(x.loc[b[0]:b[1]]) if np.size(x.loc[b[0]:b[1]]) else np.nan for b in bounds]
        max_fr = [np.nanmax(x.loc[b[0]:b[1]]) if np.size(x.loc[b[0]:b[1]]) else np.nan for b in bounds]

        mean_selectivity_index = (mean_fr[0] - mean_fr[1]) / (mean_fr[0] + mean_fr[1])
        max_selectivity_index = (max_fr[0] - max_fr[1]) / (max_fr[0] + max_fr[1])

        return pd.Series([mean_selectivity_index, max_selectivity_index],
                         name='selectivity_index',
                         index=['mean_selectivity_index', 'max_selectivity_index']).transpose()

    def _get_aligned_spikes(self):
        units_aligned = []
        ts_aligned = []
        for time_label in self.align_times:  # skip last align times so only until stop of trial
            window_start, window_stop = self._get_event_window(time_label)
            new_times = np.linspace(window_start, window_stop,
                                    int(np.round(self.theta['phase'].rate * (window_stop - window_start))))

            for unit_index in range(len(self.nwb_units)):
                if self.nwb_units.id[unit_index] in self.units_subset.index.to_list():
                    units_aligned.append(dict(trial_ids=self.trials.index.to_numpy(),
                                              time_label=time_label,
                                              unit_id=self.nwb_units.id[unit_index],
                                              spikes=align_by_time_intervals_units(self.nwb_units, unit_index, self.trials,
                                                                                   start_label=time_label,
                                                                                   stop_label=time_label,
                                                                                   start=window_start, end=window_stop),
                                              ))

            vars = dict(theta_phase=self.theta['phase'], theta_amplitude=self.theta['amplitude'],
                        rotational_velocity=self.velocity['rotational'], translational_velocity=self.velocity['translational'])
            dig_data = dict()
            for v_name, v_data in vars.items():
                dig_data[v_name], timestamps = align_by_time_intervals_ts(v_data, self.trials, return_timestamps=True,
                                                                     start_label=time_label, stop_label=time_label,
                                                                     start_window=window_start, stop_window=window_stop)
            ts_aligned.append(dict(**dig_data, timestamps=timestamps, new_times=new_times, time_label=time_label,
                                   turn_type=self.trials['turn_type'].to_numpy(),
                                   correct=self.trials['correct'].to_numpy(),
                                   update_type=self.trials['update_type'].to_numpy()))

        if np.size(self.units_subset):
            aligned_data = (pd.merge(pd.DataFrame(units_aligned), pd.DataFrame(ts_aligned), on='time_label')
                            .explode(['trial_ids', 'spikes',  *list(vars.keys()), 'timestamps', 'turn_type', 'correct',
                                      'update_type'])
                            .reset_index(drop=True))
            aligned_data['update_type'] = aligned_data['update_type'].map({1: 'non_update', 2: 'switch', 3: 'stay'})

            self.aligned_data = pd.merge(aligned_data, self.goal_selectivity, on='unit_id')
        else:
            self.aligned_data = pd.DataFrame()

    def _get_event_window(self, time_label):
        window_start, window_stop = self.window, self.window
        if time_label == 't_choice_made':
            window_stop = 0  # if choice made, don't grab data past bc could be end of trial
        elif time_label == 'start_time':
            window_start = 0

        return -window_start, window_stop

    def _get_update_selectivity(self):
        # based on the Finkelstein, Svoboda, Nat Neurosci 2021 selectivity indices
        # get trial averaged spike rates for update and non-update trials
        if np.size(self.aligned_data):
            update_aligned = (self.aligned_data
                                .query('time_label == "t_update" & correct == 1.0')
                                .groupby(['unit_id', 'update_type', 'region', 'cell_type'])
                                .apply(lambda x: self._calc_trial_averaged_psth(x['spikes'], x['new_times']))
                                .reset_index())
            psth_data = pd.json_normalize(update_aligned[0])
            update_aligned = pd.concat([update_aligned.drop(labels=[0, 'region', 'cell_type'], axis=1), psth_data], axis=1)

            # subtract update - non-update trials at all timepoints
            default_diff = np.empty(np.shape(update_aligned['psth_times'].to_numpy()[0]))
            default_diff[:] = np.nan
            update_pivot = update_aligned.pivot(index=['unit_id'], columns=['update_type'], values=['psth_mean', 'psth_err'])
            update_pivot['psth_diff_switch_non_update'] = [default_diff] * len(update_pivot.index)
            update_pivot['psth_diff_switch_stay'] = [default_diff] * len(update_pivot.index)
            trial_types = [c for c in update_pivot['psth_mean'].columns if c != 'switch' ]
            for c in trial_types:
                update_pivot[f'psth_diff_switch_{c}'] = (update_pivot[('psth_mean', 'switch')] -
                                                         update_pivot[('psth_mean', c)])
            update_pivot['psth_times'] = update_aligned.query('update_type == "non_update"')['psth_times'].to_numpy()
            update_pivot.reset_index(inplace=True)

            # average time after update and test if significant? confused how they did that
            update_selective = update_pivot.apply(lambda x: self._get_update_selectivity_index(x['psth_mean'],
                                                                                               x[('psth_times', '')],),
                                                  axis=1)
            update_pivot.columns = ['_'.join(c) if c[1] != '' else c[0] for c in update_pivot.columns.to_flat_index()]
            self.update_selectivity = pd.concat([update_pivot, pd.json_normalize(update_selective)], axis=1)
        else:
            self.update_selectivity = pd.DataFrame()

    @staticmethod
    def _get_update_selectivity_index(psth_mean, psth_times, window_size=2):
        window_start = np.where(psth_times > 0)[0][0]
        window_end = np.where(psth_times > window_size)[0][0]

        means = dict(switch=np.nan, non_update=np.nan, stay=np.nan)
        for t in ['switch', 'non_update', 'stay']:
            if t in psth_mean.index and ~np.isnan(psth_mean[t]).all():
                means[t] = np.nanmean(psth_mean[t][window_start:window_end])

        # _, switch_vs_non_update_p_value = ranksums(psth_switch, psth_non_update) from Finkelstien Svoboda 2021
        # _, switch_vs_stay_p_value = ranksums(psth_switch, psth_stay)

        switch_vs_non_update_mod = (means['switch'] - means['non_update']) / (means['switch'] + means['non_update'])
        switch_vs_stay_mod = (means['switch'] - means['stay']) / (means['switch'] + means['stay'])

        return dict(switch_vs_non_update_index=switch_vs_non_update_mod,
                    switch_vs_stay_index=switch_vs_stay_mod)

    @staticmethod
    def _calc_trial_averaged_psth(data, times, ntt=None):
        sigma_in_secs = 0.05  # 50 ms smoothing
        ntt = ntt or 250  # default value
        tt = np.linspace(times.values[0].min(), times.values[0].max(), ntt)

        all_data = np.hstack(data)
        if len(all_data):  # if any spikes
            firing_rate = np.array([compute_smoothed_firing_rate(x, tt, sigma_in_secs) for x in data])
        else:
            # firing_rate = np.empty((np.shape(data)[0], ntt))
            # firing_rate[:] = np.nan
            firing_rate = np.nan  # TODO - test how these nans propogate to include cells or not

        # get average across trials
        mean = np.nanmean(firing_rate, axis=0)
        err = sem(firing_rate, axis=0)  # just filled with nan values

        return dict(psth_mean=mean, psth_err=err, psth_times=tt)

    def correct_acg(self, raw, total_dur, sd):
        # correct raw autocorr for triangular shape from finite duration data A
        # ACG (t) = ACG_raw(time_lag) / (1 - |time_lag| / total_spike_train_duration)
        triangle_corrected = []
        for r, t in zip(raw.to_numpy(), raw.index):
            triangle_corrected.append(r / ((1 - np.abs(t)) / total_dur))

        # smooth corrected ACG (gaussian kernel, SD 20ms) and peak-normalize (2 bc each bin = 10 ms)
        smoothed = gaussian_filter1d(triangle_corrected, sigma=sd, mode='nearest')

        # peak normalize
        normalized = smoothed / np.max(smoothed)

        return normalized

    def get_theta_modulation_power(self, acg, bin_size):
        theta_band = (6, 10)
        full_band = (1, 50)

        # get power spectra with FFT and relative theta power by dividing theta band (6-10Hz) by total in (1-50Hz)
        freqs, psd = signal.welch(acg, 1 / bin_size)
        theta_power = np.sum(psd[np.logical_and(freqs >= theta_band[0], freqs <= theta_band[1])])
        full_power = np.sum(psd[np.logical_and(freqs >= full_band[0], freqs <= full_band[1])])

        # # using fft (very very similar so will use simpler version for now)
        # freqs = np.fft.fftfreq(len(acg), d=bin_size)
        # ps = np.abs(np.fft.fft(acg))**2
        # idx = np.argsort(freqs)
        # full_power_test = sum(ps[idx][np.logical_and(freqs[idx] > full_band[0], freqs[idx] < full_band[1])])
        # theta_power_test = sum(ps[idx][np.logical_and(freqs[idx] > theta_band[0], freqs[idx] < theta_band[1])])
        # fft_output = theta_power_test / full_power_test

        return theta_power / full_power

    def get_cycle_skipping_index(self, acg, times, bin_size):
        # bandpass filter theta-modulated ACGs between 1-10Hz  (lowpass for now, TODO - check this)
        # order = 4
        # fs = 1 / bin_size
        # nyq = 0.5 * fs
        # bands = [1 / nyq, 10 / nyq]
        # b, a = signal.butter(order, bands, 'bandpass')
        # acg_filtered = signal.filtfilt(b, a, acg)
        # bandpass filtering leads to negative values which result in a CSI not necessarily between -1 to 1
        acg_filtered = acg.to_numpy()  # not filtering for now bc I don't think the kay paper does

        # find first peak near t=0 in 90-200ms window
        first_window = np.logical_and(times >= 0.089, times <= 0.2)
        first_peak_ind, _ = signal.find_peaks(acg_filtered[first_window])
        if first_peak_ind.any():
            first_peak = acg_filtered[first_window][first_peak_ind[0]]
        else:
            first_peak = np.max(acg_filtered[first_window])

        # find second peak near t=0 in 200-400ms window
        second_window = np.logical_and(times >= 0.199, times <= 0.4)
        second_peak_ind, _ = signal.find_peaks(acg_filtered[second_window])
        if second_peak_ind.any():
            second_peak = acg_filtered[second_window][second_peak_ind[0]]
        else:
            second_peak = np.min(acg_filtered[second_window])

        # CSI = p2 - p1 / max(p1, p2)  -> higher indicate more skipping
        cycle_skipping_index = (second_peak - first_peak) / np.max([first_peak, second_peak])

        return cycle_skipping_index

    def _get_theta_cycle_skipping(self):
        # note that these trial times only go up until 2.5 seconds after the delay2 onset
        epochs = dict(delay=self.delay_trial_times, switch=self.switch_trial_times, stay=self.stay_trial_times,
                      **self.commitment_trial_times)

        df_list = []
        for ep_name, ep in epochs.items():
            # calculate ACG over 400ms interval with 10m time lags (raw autocorr)
            bin_size = 0.01
            window_size = 0.4
            min_spikes = 100
            if np.size(ep):  # if any time periods of that trial type
                corr_raw = nap.compute_autocorrelogram(group=self.spikes, ep=ep, binsize=bin_size,
                                                       windowsize=window_size, norm=False)

                # exclude acgs with less than X spikes contributing
                spike_rates = self.spikes.restrict(ep).get_info('freq').to_numpy()
                duration = np.sum(ep['end'] - ep['start'])
                n_spikes_in_correlogram = np.sum(corr_raw * duration * spike_rates * bin_size, axis=0).to_numpy()
                fr_inclusion_bool = n_spikes_in_correlogram > min_spikes  # at least X spikes contributing to CCG
                corr_raw = corr_raw.iloc[:, fr_inclusion_bool]

                # triangle-correct, smooth, and peak-normalize ACGs
                total_dur = np.sum(self.encoder_times['end'] - self.encoder_times['start'])
                sd = 0.02 / bin_size  # for gaussian convolution to smooth, want 10 ms smoothing (tang paper used 20 ms)
                corr_corrected = corr_raw.apply(lambda x: self.correct_acg(raw=x, total_dur=total_dur, sd=sd), axis=0)

                # get theta modulated cells
                theta_modulation = corr_corrected.apply(lambda x: self.get_theta_modulation_power(x, bin_size=bin_size), axis=0)
                theta_modulated_bool = theta_modulation > 0.15  # considered theta modulated if relative power > 0.15
                corr_corrected_theta_only = corr_corrected.iloc[:, theta_modulated_bool.to_numpy()]

                # get theta cycling index
                times = corr_raw.index.to_numpy()
                theta_cycling_index = corr_corrected_theta_only.apply(lambda x: self.get_cycle_skipping_index(x,
                                                                                                       times=times,
                                                                                                       bin_size=bin_size),
                                                                      axis=0)

                # make final output data structure
                units_sorted = self.units.sort_index()
                cycle_skipping_data = pd.concat([units_sorted['region'][fr_inclusion_bool][theta_modulated_bool],
                                                 units_sorted['cell_type'][fr_inclusion_bool][theta_modulated_bool]],
                                                axis=1)
                cycle_skipping_data['cycle_skipping_index'] = theta_cycling_index
                cycle_skipping_data['theta_modulation'] = theta_modulation[theta_modulated_bool]
                cycle_skipping_data['epoch'] = ep_name
                cycle_skipping_data['acg_corrected'] = list(corr_corrected_theta_only.T.to_numpy())
                cycle_skipping_data['acg_lags'] = [corr_corrected_theta_only.index.to_numpy()] * len(theta_cycling_index)
                df_list.append(cycle_skipping_data)

        self.cycle_skipping = pd.concat(df_list, axis=0)
