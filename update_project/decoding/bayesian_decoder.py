import numpy as np
import pickle
import pandas as pd
import pynapple as nap
import warnings

from bisect import bisect, bisect_left
from pathlib import Path
from pynwb import NWBFile
from sklearn.model_selection import train_test_split

from update_project.results_io import ResultsIO
from update_project.virtual_track import UpdateTrack
from update_project.general.lfp import get_theta
from update_project.general.acquisition import get_velocity
from update_project.general.trials import get_trials_dataframe


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

        # setup decoding/encoding functions based on dimensions
        self.dim_num = params.get('dim_num', 1)  # 1D decoding default
        self.encoder, self.decoder = self._setup_decoding_functions()

        # setup file paths for io
        trial_types = [str(t) for t in self.encoder_trial_types['correct']]
        self.results_tags = f"{'_'.join(features)}_regions_{'_'.join(self.units_types['region'])}_" \
                            f"enc_bins{self.encoder_bin_num}_dec_bins{self.decoder_bin_size}_speed_thresh" \
                            f"{self.speed_threshold}_trial_types{'_'.join(trial_types)}"
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path().absolute().stem,
                                    tags=self.results_tags)
        self.data_files = dict(bayesian_decoder_output=dict(vars=['encoder_times', 'decoder_times', 'spikes',
                                                                  'features_test', 'features_train', 'train_df',
                                                                  'test_df', 'model', 'bins', 'decoded_values',
                                                                  'decoded_probs', 'theta'],
                                                            format='pkl'),
                               params=dict(vars=['speed_threshold', 'firing_threshold', 'units_types',
                                                 'encoder_trial_types', 'encoder_bin_num', 'decoder_trial_types',
                                                 'decoder_bin_type', 'decoder_bin_size', 'decoder_test_size', 'dim_num',
                                                 'feature_names', 'linearized_features', ],
                                           format='npz'))

        # setup data
        self.feature_names = features
        self.trials = get_trials_dataframe(nwbfile, with_pseudoupdate=True)
        self.units = nwbfile.units.to_dataframe()
        self.data = self._setup_data(nwbfile)
        self.velocity = get_velocity(nwbfile)
        self.theta = get_theta(nwbfile, adjust_reference=True, session_id=session_id)

        # setup feature specific settings
        self.convert_to_binary = params.get('convert_to_binary', False)  # convert decoded outputs to binary (e.g., L/R)
        if self.feature_names[0] in ['choice', 'turn_type']:  # TODO - make this logic better so it's less confusing
            self.convert_to_binary = True  # always convert choice to binary
            self.encoder_bin_num = 2

    def run_decoding(self, overwrite=False):
        print(f'Decoding data for session {self.results_io.session_id}...')

        if overwrite:
            self._preprocess()._encode()._decode()  # build model
            self._export_data()  # save output data
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
            elif feat in ['dynamic_choice', 'cue_bias']:
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

                # load dynamic choice from saved output
                data_mapping = dict(dynamic_choice='choice', cue_bias='turn_type')
                fname_tag = data_mapping[self.feature_names[0]]
                choice_path = Path().absolute().parent.parent / 'results' / 'dynamic_choice'
                fname = self.results_io.get_data_filename(filename=f'dynamic_choice_output_{fname_tag}',
                                                          results_type='session', format='pkl',
                                                          diff_base_path=choice_path)
                import_data = self.results_io.load_pickled_data(fname)  #  TODO - make this more robust
                for v, i_data in zip(['output_data', 'agg_data', 'decoder_data', 'params'], import_data):
                    if v == 'decoder_data':
                        choice_data = i_data
                data = choice_data - 0.5  # convert to -0.5 and +0.5 for flipping between trials
            else:
                raise RuntimeError(f'{feat} feature is not currently supported')

            data_dict[feat] = pd.Series(index=time_series.timestamps[:], data=data)

        return pd.DataFrame.from_dict(data_dict)

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
            breaks[-1] = breaks[-1] - 1  # subtract one index to get closest to final trial stop time
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
            train_df, test_df = self._train_test_split(random_state=random_state + 1)  # use new random state

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

        # select additional data for post-processing
        self.theta = nap.TsdFrame(self.theta, time_units='s', time_support=self.decoder_times)

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
