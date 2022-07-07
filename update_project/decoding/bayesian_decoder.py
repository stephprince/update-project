import numpy as np
import pickle
import pandas as pd
import pynapple as nap
import warnings

from bisect import bisect, bisect_left
from math import sqrt
from pathlib import Path
from pynwb import NWBFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from update_project.results_io import ResultsIO
from update_project.decoding.interpolate import interp1d_time_intervals, griddata_2d_time_intervals, \
    griddata_time_intervals
from update_project.statistics import get_fig_stats
from update_project.virtual_track import UpdateTrack


class BayesianDecoder:
    def __init__(self, nwbfile: NWBFile, session_id: str, features: list, params: dict):
        # setup parameters
        self.units_types = params.get('units_types',
                                      dict(region=['CA1', 'PFC'],  # dict of filters to apply to units table
                                           cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))
        self.speed_threshold = params.get('speed_threshold', 1000)  # minimum virtual speed to subselect epochs
        self.firing_threshold = params.get('firing_threshold', 0)  # Hz, minimum peak firing rate of place cells to use
        self.decoder_test_size = params.get('decoder_test_size', 0.25)  # prop of trials for testing on train/test split
        self.encoder_trial_types = params.get('encoder_trial_types', dict(update_type=[1],
                                                                          correct=[0, 1]))  # trial filters
        self.encoder_bin_num = params.get('encoder_bin_num', 30)  # number of bins to build encoder
        self.decoder_trial_types = params.get('decoder_trial_types', dict(update_type=[1, 2, 3],
                                                                          correct=[0, 1]))  # trial filters
        self.decoder_bin_type = params.get('decoder_bin_type', 'time')  # time or theta phase to use for decoder
        self.decoder_bin_size = params.get('decoder_bin_size', 0.25)  # time/fraction of theta phase to use for decoder
        self.linearized_features = params.get('linearized_features', ['y_position'])  # which features to linearize
        self.prior = params.get('prior', 'uniform')  # whether to use uniform or history-dependent prior
        self.virtual_track = UpdateTrack(linearization=bool(self.linearized_features))

        # setup data
        self.feature_names = features
        self.trials = nwbfile.trials.to_dataframe()
        self.units = nwbfile.units.to_dataframe()
        self.data = self._setup_data(nwbfile)
        self.velocity = self._get_velocity(nwbfile)

        # setup feature specific settings
        self.convert_to_binary = params.get('convert_to_binary', False)  # convert decoded outputs to binary (e.g., L/R)
        self.flip_trials_by_turn = False  # default false
        if self.feature_names[0] in ['choice', 'turn_type']:  # TODO - make this logic better so it's less confusing
            self.convert_to_binary = True  # always convert choice to binary
            self.encoder_bin_num = 2
        if self.feature_names[0] in ['x_position', 'view_angle', 'choice', 'turn_type']:
            self.flip_trials_by_turn = True  # flip data by turn type for averaging

        # setup decoding/encoding functions based on dimensions
        self.dim_num = params.get('dim_num', 1)  # 1D decoding default
        self.encoder, self.decoder = self._setup_decoding_functions()

        # setup file paths for io
        trial_types = [str(t) for t in self.encoder_trial_types['correct']]
        self.results_tags = f"{'_'.join(self.feature_names)}_regions_{'_'.join(self.units_types['region'])}_" \
                            f"enc_bins{self.encoder_bin_num}_dec_bins{self.decoder_bin_size}_speed_thresh" \
                            f"{self.speed_threshold}_trial_types{'_'.join(trial_types)}"
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path().absolute().stem,
                                    tags=self.results_tags)
        self.data_files = dict(bayesian_decoder_output=dict(vars=['encoder_times', 'decoder_times', 'spikes',
                                                                  'features_test', 'features_train', 'train_df',
                                                                  'test_df', 'model', 'bins', 'decoded_values',
                                                                  'decoded_probs', 'aligned_data',
                                                                  'aligned_data_window', 'aligned_data_nbins',
                                                                  'summary_df'],
                                                            format='pkl'),
                               params=dict(vars=['speed_threshold', 'firing_threshold', 'units_types',
                                                 'encoder_trial_types', 'encoder_bin_num', 'decoder_trial_types',
                                                 'decoder_bin_type', 'decoder_bin_size', 'decoder_test_size', 'dim_num',
                                                 'feature_names', 'linearized_features' ],
                                           format='npz'))

    def run_decoding(self, overwrite=False):
        print(f'Decoding data for session {self.results_io.session_id}...')

        if overwrite:
            self._preprocess()._encode()._decode()   # build model
            self._align_by_times()                   # align data by times
            self._summarize()                        # generate summary data
            self._export_data()                      # save output data
        else:
            if self._data_exists() and self._params_match():
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run_decoding(overwrite=True)

        return self

    def _setup_data(self, nwbfile):
        data_dict = dict()
        for feat in self.feature_names:
            if feat in ['x_position', 'y_position']:
                time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
                if feat in self.linearized_features:
                    raw_data = time_series.data[:]
                    data = self.virtual_track.linearize_track_position(raw_data)
                else:
                    column = [ind for ind, val in enumerate(['x_position', 'y_position']) if val == feat][0]
                    data = time_series.data[:, column]
            elif feat in ['view_angle']:
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')
                data = time_series.data[:]
            elif feat in ['choice', 'turn_type']:
                choice_mapping = {'0': np.nan, '1': -1, '2': 1}  # convert to negative/non-negative for flipping
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

                data = np.empty(np.shape(time_series.data))
                data[:] = np.nan
                for n, trial in self.trials.iterrows():
                    idx_start = bisect_left(time_series.timestamps, trial['start_time'])
                    idx_stop = bisect(time_series.timestamps, trial['stop_time'], lo=idx_start)
                    data[idx_start:idx_stop] = choice_mapping[
                        str(int(trial[feat]))]  # fill in values with the choice value

                    # choice is the animals final decision but turn_type indicates what side is correct/incorrect
                    # turn type will flip around the update
                    if feat in ['turn_type'] and ~np.isnan(trial['t_update']) and trial['update_type'] == 2:
                        idx_switch = bisect_left(time_series.timestamps, trial['t_update'])
                        data[idx_switch:idx_stop] = -1 * choice_mapping[str(int(trial[feat]))]
            else:
                raise RuntimeError(f'{feat} feature is not currently supported')

            data_dict[feat] = pd.Series(index=time_series.timestamps[:], data=data)

        return pd.DataFrame.from_dict(data_dict)

    @staticmethod
    def _get_velocity(nwbfile):
        rotational_velocity = nwbfile.acquisition['rotational_velocity'].data
        translational_velocity = nwbfile.acquisition['translational_velocity'].data
        velocity = np.abs(rotational_velocity[:]) + np.abs(translational_velocity[:])
        rate = nwbfile.acquisition['translational_velocity'].rate
        timestamps = np.arange(0, len(velocity)/rate, 1/rate)
        velocity = pd.Series(index=timestamps[:], data=velocity)

        return velocity

    def _get_time_intervals(self, trial_starts, trial_stops):
        movement = self.velocity > self.speed_threshold

        new_starts = []
        new_stops = []
        for start, stop in zip(trial_starts, trial_stops):
            # pull out trial specific time periods
            start_ind = bisect(movement.index, start)
            stop_ind = bisect_left(movement.index, stop)
            movement_by_trial = movement.iloc[start_ind:stop_ind]

            # get intervals where movement crosses speed threshold
            breaks = np.where(np.hstack((True, np.diff(movement_by_trial), True)))[0]
            breaks[-1] = breaks[-1] - 1 # subtract one index to get closest to final trial stop time
            index = pd.IntervalIndex.from_breaks(movement_by_trial.index[breaks])
            movements_df = pd.DataFrame(movement_by_trial[index.left].to_numpy(), index=index, columns=['moving'])

            # append new start/stop times to data struct
            new_times = movements_df[movements_df['moving'] == True].index
            new_starts.append(new_times.left.values)
            new_stops.append(new_times.right.values)

        times = nap.IntervalSet(start=np.hstack(new_starts), end=np.hstack(new_stops), time_units='s')

        return times

    def _setup_decoding_functions(self):
        if self.dim_num == 1:
            encoder = nap.compute_1d_tuning_curves
            decoder = nap.decode_1d
        elif self.dim_num == 2:  # TODO - test 2D data
            encoder = nap.compute_2d_tuning_curves
            decoder = nap.decode_2d
        elif self.dim_num not in [1, 2]:
            raise RuntimeError('Invalid decoding dimension')

        return encoder, decoder

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

    def _train_test_split_ok(self, train_data, test_data):
        train_values = np.sort(train_data[self.feature_names[0]].unique())
        test_values = np.sort(test_data[self.feature_names[0]].unique())

        train_values = np.delete(train_values, train_values == 0)  # remove failed trials bc n/a to train/test check
        test_values = np.delete(test_values, test_values == 0)

        return all(train_values == test_values)

    def _train_test_split(self, random_state=21):
        # test data includes update/non-update trials, would focus only on non-update trials for error summary so the
        # test size refers only to the non-update trials in the decoding vs encoding phase

        # get encoder/training data times
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.encoder_trial_types.items()], axis=1).all(axis=1)
        encoder_trials = self.trials[mask]
        train_data, test_data = train_test_split(encoder_trials, test_size=self.decoder_test_size,
                                                 random_state=random_state)
        train_df = train_data.sort_index()

        # get decoder/testing data times (decoder conditions + remove any training data)
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.decoder_trial_types.items()], axis=1).all(axis=1)
        decoder_trials = self.trials[mask]
        test_df = decoder_trials[~decoder_trials.index.isin(train_df.index)]  # remove any training data

        # check that split is ok for binarized data, run different random states until it is
        if self.convert_to_binary and not self._train_test_split_ok(train_df, test_df):
            train_df, test_df = self._train_test_split(random_state=random_state+1)  # use new random state

        return train_df, test_df

    def _preprocess(self):
        # get spikes
        units_mask = pd.concat([self.units[k].isin(v) for k, v in self.units_types.items()], axis=1).all(axis=1)
        units_subset = self.units[units_mask]
        spikes_dict = {n: nap.Ts(t=units_subset.loc[n, 'spike_times'], time_units='s') for n in units_subset.index}
        if spikes_dict:
            self.spikes = nap.TsGroup(spikes_dict, time_units='s')
        else:
            self.spikes = []  # if no units match criteria, leave spikes empty

        # split data into training/encoding and testing/decoding trials
        self.train_df, self.test_df = self._train_test_split()

        # get start/stop times of train and test trials
        self.encoder_times = self._get_time_intervals(self.train_df['start_time'], self.train_df['stop_time'])
        self.decoder_times = self._get_time_intervals(self.test_df['start_time'], self.test_df['stop_time'])

        # select feature
        self.features_train = nap.TsdFrame(self.data, time_units='s', time_support=self.encoder_times)
        self.features_test = nap.TsdFrame(self.data, time_units='s', time_support=self.decoder_times)

        return self

    def _encode(self):
        if self.spikes:  # if there were units and spikes to use for encoding
            if self.dim_num == 1:
                feat_input = self.features_train[self.feature_names[0]]
                self.bins = np.linspace(np.min(feat_input), np.max(feat_input), self.encoder_bin_num + 1)
                self.model = self.encoder(group=self.spikes, feature=feat_input, nb_bins=self.encoder_bin_num,
                                          ep=self.encoder_times)
            elif self.dim_num == 2:
                self.model, self.bins = self.encoder(group=self.spikes, feature=self.features_train,
                                                     nb_bins=self.encoder_bin_num,
                                                     ep=self.encoder_times)
        else:  # if there were no units/spikes to use for encoding, create empty dataframe
            self.model = pd.DataFrame()
            self.bins = []

        return self

    def _decode(self):
        # get arguments for decoder
        kwargs = dict(tuning_curves=self.model, group=self.spikes, ep=self.decoder_times,
                      bin_size=self.decoder_bin_size)
        if self.dim_num == 2:
            kwargs.update(binsxy=self.bins)

        if self.prior == 'history-dependent':
            if self.dim_num == 1:
                kwargs.update(feature=self.features_train)
            elif self.dim_num == 2:
                kwargs.update(features=self.features_train)

        # run decoding
        if self.model.any().any():
            self.decoded_values, self.decoded_probs = self.decoder(**kwargs)
        else:
            self.decoded_values = pd.DataFrame()
            self.decoded_probs = pd.DataFrame()

        # convert to binary if needed
        if self.convert_to_binary:
            self.decoded_values.values[self.decoded_values > 0] = int(1)
            self.decoded_values.values[self.decoded_values < 0] = int(-1)

        return self

    def _align_by_times(self, trial_types=['switch', 'stay'], times='t_update', nbins=50, window=5):
        print(f'Aligning data for session {self.results_io.session_id}...')

        trial_type_dict = dict(nonupdate=1, switch=2, stay=3)
        output = dict()
        for trial_name in trial_types:
            trials_to_agg = self.test_df[self.test_df['update_type'] == trial_type_dict[trial_name]]

            mid_times = trials_to_agg[times]
            new_times = np.linspace(-window, window, num=nbins)
            if self.flip_trials_by_turn:
                trials_to_flip = trials_to_agg['turn_type'] == 1  # left trials, flip so all values the same way
            else:
                trials_to_flip = trials_to_agg['turn_type'] == 100  # set all to false

            # add extra index step to stop locs for interpolation and go one index earlier for start locs
            feat_start_locs = self.features_test[self.feature_names[0]].index.searchsorted(mid_times - window) - 1
            decoding_start_locs = self.decoded_values.index.searchsorted(mid_times - window) - 1
            feat_stop_locs = self.features_test[self.feature_names[0]].index.searchsorted(mid_times + window) + 1
            decoding_stop_locs = self.decoded_values.index.searchsorted(mid_times + window) + 1

            feat_interp = dict()
            decoding_interp = dict()
            decoding_error = dict()
            probability = dict()
            for name in self.feature_names:
                if self.decoded_values.any().any():
                    feat_interp[name] = interp1d_time_intervals(self.features_test[name],
                                                                feat_start_locs, feat_stop_locs,
                                                                new_times, mid_times, trials_to_flip)
                    decoding_interp[name] = interp1d_time_intervals(self.decoded_values, decoding_start_locs,
                                                                    decoding_stop_locs,
                                                                    new_times, mid_times, trials_to_flip)
                    decoding_error[name] = [abs(dec_feat - true_feat) for true_feat, dec_feat in
                                            zip(feat_interp[name], decoding_interp[name])]

                    if self.dim_num == 1:
                        probability[name] = griddata_time_intervals(self.decoded_probs, decoding_start_locs,
                                                                    decoding_stop_locs,
                                                                    nbins, trials_to_flip, mid_times)
                    elif self.dim_num == 2:
                        probability[name] = griddata_2d_time_intervals(self.decoded_probs, self.bins,
                                                                       self.decoding_values.index.values,
                                                                       decoding_start_locs, decoding_stop_locs,
                                                                       mid_times, nbins,
                                                                       trials_to_flip)
                else:
                    feat_interp[name] = []
                    decoding_interp[name] = []
                    decoding_error[name] = []
                    probability[name] = []

            # get means and sem
            data = dict()
            for name in self.feature_names:
                data[name] = dict(feature=np.array(feat_interp[name]).T,
                                  decoding=np.array(decoding_interp[name]).T,
                                  error=np.array(decoding_error[name]).T, )
                data[name].update(stats={k: get_fig_stats(v, axis=1) for k, v in data[name].items()})
                data[name].update(probability=probability[name])

            output.update({trial_name: data})

        self.aligned_data = output
        self.aligned_data_window = window
        self.aligned_data_nbins = nbins

    def _summarize(self):
        print(f'Summarizing data for session {self.results_io.session_id}...')

        # get decoding error
        time_index = []
        feature_mean = []
        for index, trial in self.decoder_times.iterrows():
            trial_bins = np.arange(trial['start'], trial['end'] + self.decoder_bin_size, self.decoder_bin_size)
            bins = pd.cut(self.features_test[self.feature_names[0]].index, trial_bins)
            feature_mean.append(self.features_test[self.feature_names[0]].groupby(bins).mean())
            time_index.append(trial_bins[0:-1] + np.diff(trial_bins) / 2)
        time_index = np.hstack(time_index)
        feature_means = np.hstack(feature_mean)

        actual_series = pd.Series(feature_means, index=np.round(time_index, 4), name='actual_feature')
        if self.decoded_values.any().any():
            decoded_series = self.decoded_values.as_series()  # TODO - figure out why decoded/actual series are diff lengths
        else:
            decoded_series = pd.Series()
        df_decode_results = pd.merge(decoded_series.rename('decoded_feature'), actual_series, how='left',
                                     left_index=True, right_index=True)
        df_decode_results['decoding_error'] = abs(
            df_decode_results['decoded_feature'] - df_decode_results['actual_feature'])
        df_decode_results['decoding_error_rolling'] = df_decode_results['decoding_error'].rolling(20,
                                                                                                  min_periods=20).mean()
        if self.decoded_probs.any().any():
            df_decode_results['prob_dist'] = [x for x in self.decoded_probs.as_dataframe().to_numpy()]
            df_positions = df_decode_results[['actual_feature', 'decoded_feature']].dropna(how='any')
            rmse = sqrt(mean_squared_error(df_positions['actual_feature'], df_positions['decoded_feature']))
        else:
            df_decode_results['prob_dist'] = [x for x in self.decoded_probs.to_numpy()]
            rmse = np.nan

        # add summary data
        df_decode_results['session_rmse'] = rmse
        df_decode_results['animal'] = self.results_io.animal
        df_decode_results['session'] = self.results_io.session_id

        self.summary_df = df_decode_results

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
