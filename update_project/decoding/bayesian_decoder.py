import numpy as np
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


class BayesianDecoder:
    def __init__(self, nwbfile: NWBFile, session_id: str, features: list, params: dict, overwrite=False):
        # setup data
        self.feature_names = features
        self.trials = nwbfile.trials.to_dataframe()
        self.units = nwbfile.units.to_dataframe()
        self.data = self._setup_data(nwbfile)

        # setup parameters
        self.units_types = params.get('units_types',
                                      dict(region=['CA1', 'PFC'],  # dict of filters to apply to units table
                                           cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))
        self.units_threshold = params.get('units_threshold', 0)  # minimum number of units to use for a session
        self.speed_threshold = params.get('speed_threshold', 0)  # minimum virtual speed to subselect epochs
        self.firing_threshold = params.get('firing_threshold', 0)  # Hz, minimum peak firing rate of place cells to use
        self.decoder_test_size = params.get('decoder_test_size',
                                            0.25)  # proportion of trials to use for testing on train/test split
        self.encoder_trial_types = params.get('encoder_trial_types', dict(update_type=[1],
                                                                          correct=[0,
                                                                                   1]))  # dict of filters to apply to trials table
        self.encoder_bin_num = params.get('encoder_bin_num', 30)  # number of bins to build encoder
        self.decoder_trial_types = params.get('decoder_trial_types', dict(update_type=[1, 2, 3],
                                                                          correct=[0,
                                                                                   1]))  # dict of filters to apply to trials table
        self.decoder_bin_type = params.get('decoder_bin_type', 'time')  # time or theta phase to use for decoder
        self.decoder_bin_size = params.get('decoder_bin_size', 0.25)  # time/fraction of theta phase to use for decoder
        self.linearize_feature = params.get('linearize_feature', False)  # whether to linearize y-position/feature
        self.prior = params.get('prior', 'uniform')  # whether to use uniform or history-dependent prior
        self.excluded_session = False  # initialize to False, will be set to True if does not pass session requirements
        self.convert_to_binary = params.get('convert_to_binary', False)  # convert decoded outputs to binary (e.g., L/R)
        if self.feature_names[0] in ['choice', 'turn_type']:
            self.convert_to_binary = True  # always convert choice to binary
            self.encoder_bin_num = 3

        # setup decoding/encoding functions based on dimensions
        self.dim_num = params.get('dim_num', 1)  # 1D decoding default
        self.encoder, self.decoder = self._setup_decoding_functions()

        # setup file paths for io
        self.results_tags = f"{'_'.join(self.feature_names)}_regions_{'_'.join(self.units_types['region'])}"
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path().absolute().stem,
                                    tags=self.results_tags)
        self.data_files = dict(preprocess_output=['encoder_times', 'decoder_times', 'spikes', 'features_test',
                                                  'features_train', 'test_data', 'train_data'],
                               encode_output=['model', 'bins'],
                               decode_output=['decoded_values', 'prob_densities'],
                               aligned_output=['aligned_data', 'aligned_data_window', 'aligned_data_nbins'],
                               summary_output=['decoding_summary'],
                               params=['units_threshold', 'speed_threshold', 'firing_threshold', 'units_types',
                                       'encoder_trial_types', 'encoder_bin_num', 'decoder_trial_types',
                                       'decoder_bin_type',
                                       'decoder_bin_size', 'decoder_test_size', 'dim_num', 'feature_names',
                                       'linearize_feature', 'excluded_session', ])

    def run_decoding(self, overwrite=False):
        print(f'Decoding data for session {self.results_io.session_id}...')

        if overwrite:
            self._preprocess()._encode()._decode()  # build model
            self._align_by_times()                   # align data by times
            self._summarize()                        # generate summary data
            self._export_data()                      # save output data
        else:
            if self._check_params_match():
                self._load_data()  # load existing data structure if existing params match
            else:
                warnings.warn('Saved data did not match input parameters, overwriting existing data')
                self.run_decoding(overwrite=True)

        return self

    def _setup_data(self, nwbfile):
        data_dict = dict()
        for feat in self.feature_names:
            if feat in ['x_position', 'y_position']:
                time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
                column = [ind for ind, val in enumerate(['x_position', 'y_position']) if val == feat][0]
                data = time_series.data[:, column]
            elif feat in ['view_angle']:
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')
                data = time_series.data[:]
            elif feat in ['choice', 'turn_type']:
                choice_mapping = {'1': -1, '2': 1}  # convert to negative/non-negative for flipping
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

    def _check_params_match(self):
        if not self.overwrite:  # if loading from cached file, check params match, otherwise set overwrite to true
            params_path = self.results_io.get_data_filename(f'params', results_type='session', format='npz')
            params_cached = np.load(params_path, allow_pickle=True)
            params_matched = []
            for k, v in params_cached.items():
                params_matched.append(getattr(self, k) == v)

        return all(params_matched)

    def _preprocess(self):
        # get spikes
        units_mask = pd.concat([self.units[k].isin(v) for k, v in self.units_types.items()], axis=1).all(axis=1)
        units_subset = self.units[units_mask]
        spikes_dict = {n: nap.Ts(t=units_subset.loc[n, 'spike_times'], time_units='s') for n in units_subset.index}
        self.spikes = nap.TsGroup(spikes_dict, time_units='s')
        if len(units_subset) < self.units_threshold:
            self.excluded_session = True
            warnings.warn(f'Session {self.results_io.session_id} does not meet requirements '
                          f'"Number of units >= {self.units_threshold}')

        # get encoder/training data times
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.encoder_trial_types.items()], axis=1).all(axis=1)
        encoder_trials = self.trials[mask]
        train_data, test_data = train_test_split(encoder_trials, test_size=self.decoder_test_size, random_state=21)
        self.train_data = train_data.sort_index()
        # test data includes update/non-update trials, would focus only on non-update trials for error summary so the
        # test size refers only to the non-update trials in the decoding vs encoding phase

        # get decoder/testing data times (decoder conditions + remove any training data)
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.decoder_trial_types.items()], axis=1).all(axis=1)
        decoder_trials = self.trials[mask]
        self.test_data = decoder_trials[~decoder_trials.index.isin(self.train_data.index)]  # remove any training data

        # apply speed threshold
        if self.speed_threshold > 0:
            get_movement_times()  # TODO - make this function
        self.encoder_times = nap.IntervalSet(start=self.train_data['start_time'], end=self.train_data['stop_time'],
                                             time_units='s')
        self.decoder_times = nap.IntervalSet(start=self.test_data['start_time'], end=self.test_data['stop_time'],
                                             time_units='s')

        # select feature, TODO - linearize y-position if needed
        if self.linearize_feature:
            new_position = linearize_y_position()  # TODO - make this function
        self.features_train = nap.TsdFrame(self.data, time_units='s', time_support=self.encoder_times)
        self.features_test = nap.TsdFrame(self.data, time_units='s', time_support=self.decoder_times)

        return self

    def _encode(self):
        if self.dim_num == 1:
            feat_input = self.features_train[self.feature_names[0]]
            self.model = self.encoder(group=self.spikes, feature=feat_input, nb_bins=self.encoder_bin_num,
                                      ep=self.encoder_times)
            self.bins = np.linspace(np.min(feat_input), np.max(feat_input), self.encoder_bin_num + 1)
        elif self.dim_num == 2:
            self.model, self.bins = self.encoder(group=self.spikes, feature=self.features_train,
                                                 nb_bins=self.encoder_bin_num,
                                                 ep=self.encoder_times)

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
        self.decoded_values, self.prob_densities = self.decoder(**kwargs)

        # convert to binary if needed
        if self.convert_to_binary:
            self.decoded_values.values[self.decoded_values > 0] = int(1)
            self.decoded_values.values[self.decoded_values < 0] = int(-1)

        return self

    def _align_by_times(self, trial_types=['switch', 'stay'], times='t_update', nbins=50, window=5, flip=False):
        print(f'Aligning data for session {self.results_io.session_id}...')

        if self.overwrite:
            trial_type_dict = dict(nonupdate=1, switch=2, stay=3)
            output = dict()
            for trial_name in trial_types:
                trials_to_agg = self.test_data[self.test_data['update_type'] == trial_type_dict[trial_name]]

                mid_times = trials_to_agg[times]
                new_times = np.linspace(-window, window, num=nbins)
                if flip:
                    trials_to_flip = trials_to_agg[
                                         'turn_type'] == 1  # left trials, flip values so all face the same way
                else:
                    trials_to_flip = trials_to_agg['turn_type'] == 100  # set all to false

                feat_start_locs = self.features_test[self.feature_names[0]].index.searchsorted(
                    mid_times - window - 1)  # a little extra just in case
                feat_stop_locs = self.features_test[self.feature_names[0]].index.searchsorted(mid_times + window + 1)
                decoding_start_locs = self.decoded_values.index.searchsorted(mid_times - window - 1)
                decoding_stop_locs = self.decoded_values.index.searchsorted(mid_times + window + 1)

                feat_interp = dict()
                decoding_interp = dict()
                decoding_error = dict()
                probability = dict()
                for name in self.feature_names:
                    feat_interp[name] = interp1d_time_intervals(self.features_test[name],
                                                                feat_start_locs, feat_stop_locs,
                                                                new_times, mid_times, trials_to_flip)
                    decoding_interp[name] = interp1d_time_intervals(self.decoded_values, decoding_start_locs,
                                                                    decoding_stop_locs,
                                                                    new_times, mid_times, trials_to_flip)
                    decoding_error[name] = [abs(dec_feat - true_feat) for true_feat, dec_feat in
                                            zip(feat_interp[name], decoding_interp[name])]

                    if self.dim_num == 1:
                        probability[name] = griddata_time_intervals(self.prob_densities, decoding_start_locs,
                                                                    decoding_stop_locs,
                                                                    nbins, trials_to_flip, mid_times)
                    elif self.dim_num == 2:
                        probability[name] = griddata_2d_time_intervals(self.prob_densities, self.bins,
                                                                       self.decoding_values.index.values,
                                                                       decoding_start_locs, decoding_stop_locs,
                                                                       mid_times, nbins,
                                                                       trials_to_flip)

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

        else:
            data_path = self.results_io.get_data_filename('aligned_data', results_type='session', format='npz')
            aligned_data = np.load(data_path, allow_pickle=True)

            self.aligned_data = aligned_data['aligned_data']
            self.aligned_data_window = aligned_data['aligned_data_window']
            self.aligned_data_nbins = aligned_data['aligned_data_nbins']

    def _summarize(self):
        print(f'Summarizing data for session {self.results_io.session_id}...')

        if self.overwrite:
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

            actual_series = pd.Series(feature_means, index=time_index, name='actual_feature')
            decoded_series = self.decoded_values.as_series()  # TODO - figure out why decoded and actual series are diff lengths
            df_decode_results = pd.merge(decoded_series.rename('decoded_feature'), actual_series, how='left',
                                         left_index=True, right_index=True)
            df_decode_results['decoding_error'] = abs(
                df_decode_results['decoded_feature'] - df_decode_results['actual_feature'])
            df_decode_results['decoding_error_rolling'] = df_decode_results['decoding_error'].rolling(20,
                                                                                                      min_periods=20).mean()
            df_decode_results['prob_dist'] = [x for x in self.prob_densities.as_dataframe().to_numpy()]

            # add summary data
            df_positions = df_decode_results[['actual_feature', 'decoded_feature']].dropna(how='any')
            rmse = sqrt(mean_squared_error(df_positions['actual_feature'], df_positions['decoded_feature']))
            df_decode_results['session_rmse'] = rmse
            df_decode_results['animal'] = self.results_io.animal
            df_decode_results['session'] = self.results_io.session_id
        else:
            data_path = self.results_io.get_data_filename('decoding_summary', results_type='session', format='csv')
            df_decode_results = pd.read_csv(data_path, index_col=0, na_values=['NaN', 'nan'], keep_default_na=True)

        self.summary_df = df_decode_results

    def _load_data(self):
        print(f'Loading existing data for session {self.results_io.session_id}...')

        for name, vars in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format='npz')
            import_data = np.load(fname, allow_pickle=True)

            for v in vars:
                setattr(self, v, import_data[v])

        return self

    def _export_data(self):
        print(f'Exporting data for session {self.results_io.session_id}...')

        for name, vars in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format='npz')
            kwargs = {v: getattr(self, v) for v in vars}
            np.savez(fname, **kwargs)

