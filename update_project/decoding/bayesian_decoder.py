import numpy as np
import pandas as pd
import pynapple as nap

from math import sqrt
from pynwb import NWBFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from update_project.results_io import ResultsIO
from bayesian_decoder_visualizer import BayesianDecoderVisualizer
from update_project.statistics import get_stats


class BayesianDecoder:
    def __init__(self, nwbfile: NWBFile, session_id: str, overwrite: bool, features: list, params: dict):
        # setup data
        self.data = self._setup_data(nwbfile, features)
        self.features = features
        self.trials = nwbfile.intervals['trials'].to_dataframe()
        self.units = nwbfile.units.to_dataframe()

        # setup file paths for io
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id)
        self.overwrite = overwrite

        # setup parameters
        self.units_threshold = params.get('units_threshold', 0) # minimum number of units to use for a session
        self.speed_threshold = params.get('speed_threshold', 0) # minimum virtual speed to subselect epochs
        self.firing_threshold = params.get('firing_threshold', 0) # Hz, minimum peak firing rate of place cells to use
        self.encoder_trial_types = params.get('encoder_trial_types', dict(update_type=[1])) # dict of filters to apply
        self.encoder_bin_num = params.get('encoder_bin_num', 30) # number of bins to build encoder
        self.decoder_trial_types = params.get('decoder_trial_types', dict(update_type=[1, 2, 3]))  # dict of filters to apply
        self.decoder_bin_type = params.get('decoder_bin_type', 'time') # time or theta phase to use for decoder
        self.decoder_bin_size = params.get('decoder_bin_size', 0.25) # time or fraction of theta phase to use for decoder

        # setup decoding/encoding functions based on dimensions
        self.dim_num, self.encoder, self.decoder = self._setup_dim_functions(params.get('dim_num'))

    @staticmethod
    def _setup_dim_functions(dim_num):
        if dim_num is None:
            dim_num = 1
            warnings.warn(f'No decoding dimension provided, defaulting to {self.dim_num}D')
        elif dim_num == 1:
            encoder = nap.compute_1d_tuning_curves
            decoder = nap.decode_1d
        elif dim_num == 2:
            encoder = nap.compute_2d_tuning_curves
            decoder = nap.decode_2d
        elif dim_num not in [1, 2]:
            raise RuntimeError('Invalid decoding dimension')

        return dim_num, encoder, decoder

    @staticmethod
    def _setup_data(nwbfile, features):
        data_dict = dict()
        for feat in features:
            if feat in ['x_position', 'y_position']:
                time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
                column = np.argwhere(feat, ['x_position', 'y_position'])[0]
                data = time_series.data[:,column]
            elif feat in ['view_angle']:
                time_series = nwbfile.processing['behavior']['view_angle'].get_time_series('view_angle')
                data = time_series.data[:]
            else:
                raise RuntimeError(f'{feat} feature is not currently supported')

            data_dict.update(feat=pd.Series(index=time_series.timestamps[:], data=data))

        return pd.DataFrame.from_dict(data_dict)

    def run_decoding(self):
        if self.overwrite:
            self.preprocess().encode().decode()
        else:
            data_path = self.results_io.get_data_path(results_type='session')
            preprocess_output = np.load(data_path / 'preprocess_output.npz', allow_pickle=True)
            self.encoder_times = preprocess_output['encoder_times']
            self.decoder_times = preprocess_output['decoder_times']
            self.spikes = preprocess_output['spikes']
            self.features = preprocess_output['features']

            encode_output = np.load(data_path / 'encode_output.npz', allow_pickle=True)
            self.model = encode_output['model']
            self.bins = encode_output['bins']

            decode_output = np.load(data_path / 'decode_output.npz', allow_pickle=True)
            self.decoded_values = decode_output['decoded_values']
            self.prob_densities = decode_output['prob_densities']

    def preprocess(self):
        # get times
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.encoder_trial_types.items()], axis=1).all(axis=1)
        encoder_times = self.trials[mask]
        self.encoder_times = nap.IntervalSet(start=encoder_times['start_time'], end=encoder_times['stop_time'], time_units='s')

        mask = pd.concat([self.trials[k].isin(v) for k, v in self.decoder_trial_types.items()], axis=1).all(axis=1)
        decoder_times = self.trials[mask]
        self.decoder_times = nap.IntervalSet(start=decoder_times['start_time'], end=decoder_times['stop_time'], time_units='s')

        # get spikes
        spikes_dict = {n: nap.Ts(t=units.loc[n, 'spike_times'], time_units='s') for n in units.index}
        self.spikes = nap.TsGroup(spikes_dict, time_units='s')

        # subselect feature
        self.features = nap.TsdFrame(self.data, time_units='s', time_support=self.encoder_times)
        # TODO - linearize y-position

        return self

    def encode(self):
        self.model, self.bins = self.encoder(group=self.spikes, feature=self.features, nb_bins=self.encoder_bin_num,
                                                 ep=self.encoder_times)
        if not self.bins:
            self.bins = np.linspace(np.min(self.features), np.max(self.features), self.encoder_bin_num + 1)

        return self

    def decode(self):
        # get arguments for decoder
        kwargs = dict(tuning_curves=self.model,
                    group=self.spikes,
                    ep=self.decoder_times,
                    bin_size=self.decoder_bin_size)
        if self.dim_num == 2:
            kwargs.update(binsxy=self.bins)

        if self.prior_type == 'uniform':
            pass
        elif self.prior_type == 'history-dependent':
            if self.dim_num == 1:
                kwargs.update(feature=self.features)
            elif self.dim_num == 2:
                kwargs.update(features=self.features)

        # run decoding
        self.decoded_values, self.prob_densities = self.decoder(**kwargs)

        return self

    def aggregate(self, trial_types, times='t_update', nbins=50, window=5, flip=False):
        trial_type_dict = dict(switch=2, stay=3)
        data_dict = dict()
        for trial_name in trial_types:
            trials_to_agg = self.decoder_times[self.decoder_times['update_type'] == trial_type_dict[trial_name]]

            if self.dim_num == 1:
                data = self.get_decoding_around_update(trials_to_agg, times=times, nbins=nbins, window=window, flip=flip)
            elif self.dim_num == 2:
                data = self.get_2d_decoding_around_update(trials_to_agg, times=times, nbins=nbins, window=window,
                                                          flip=flip)
            data_dict.update({trial_name: data})

        return data_dict

    def get_decoding_around_update(self, trials, times, nbins, window, flip):
        mid_times = trials[times]
        new_times = np.linspace(-window, window, num=nbins)
        if flip:
            trials_to_flip = trials['turn_type'] == 1  # left trials, flip values so all face the same way
        else:
            trials_to_flip = trials['turn_type'] == 100  # set all to false

        start_locs = self.features.index.searchsorted(mid_times - window - 1)  # a little extra just in case
        stop_locs = self.features.index.searchsorted(mid_times + window + 1)
        feat_interp = interp1d_time_intervals(self.features, start_locs, stop_locs, new_times, mid_times, trials_to_flip)
        feat_out = np.array(feat_interp).T

        decoding_start_locs = self.decoded_values.index.searchsorted(mid_times - window - 1)
        decoding_stop_locs = self.decoded_values.index.searchsorted(mid_times + window + 1)
        decoding_interp = interp1d_time_intervals(self.decoded_values, decoding_start_locs, decoding_stop_locs,
                                                  new_times, mid_times, trials_to_flip)
        decoding_out = np.array(decoding_interp).T

        prob_out = griddata_time_intervals(self.prob_densities, decoding_start_locs, decoding_stop_locs, mid_times, nbins,
                                           trials_to_flip)

        decoding_error = [abs(dec_feat - true_feat) for true_feat, dec_feat in zip(feat_interp, decoding_interp)]
        error_out = np.array(decoding_error).T

        # get means and sem
        stats = {'position': get_stats(pos_interp),
                 'decoding': get_stats(decoding_interp),
                 'probability': get_stats(prob_out),
                 'error': get_stats(decoding_error)}

        return {'position': pos_out,
                'decoding': decoding_out,
                'probability': prob_out,
                'decoding_error': error_out,
                'stats': stats}

    def get_2d_decoding_around_update(self, decoded_data, prob_feature, binsxy, trials, nbins=50, window=5):
        update_times = trials['t_update']
        new_times = np.linspace(-window, window, num=nbins)
        trials_to_flip = pd.concat([trials['turn_type'] == 1, trials['turn_type'] == 100], axis=1)
        trials_to_flip.columns = ['x', 'y']

        pos_start_locs = position.index.searchsorted(update_times - window - 1)  # a little extra just in case
        pos_stop_locs = position.index.searchsorted(update_times + window + 1)
        posx_interp = interp1d_time_intervals(position['x'], pos_start_locs, pos_stop_locs, new_times, update_times,
                                              trials_to_flip['x'])
        posy_interp = interp1d_time_intervals(position['y'], pos_start_locs, pos_stop_locs, new_times, update_times,
                                              trials_to_flip['y'])

        decoding_start_locs = decoded_data.index.searchsorted(update_times - window - 1)
        decoding_stop_locs = decoded_data.index.searchsorted(update_times + window + 1)
        decodingx_interp = interp1d_time_intervals(decoded_data['x'], decoding_start_locs, decoding_stop_locs,
                                                   new_times,
                                                   update_times, trials_to_flip['x'])
        decodingy_interp = interp1d_time_intervals(decoded_data['y'], decoding_start_locs, decoding_stop_locs,
                                                   new_times,
                                                   update_times, trials_to_flip['y'])

        prob_out = griddata_2d_time_intervals(prob_feature, binsxy, decoded_data.index.values, decoding_start_locs,
                                              decoding_stop_locs, update_times, nbins,
                                              trials_to_flip)

        decodingx_error = [abs(dec_pos - true_pos) for true_pos, dec_pos in zip(posx_interp, decodingx_interp)]
        decodingy_error = [abs(dec_pos - true_pos) for true_pos, dec_pos in zip(posy_interp, decodingy_interp)]

        # get means and sem
        stats = {'position_x': get_stats(posx_interp),
                 'position_y': get_stats(posy_interp),
                 'decoding_x': get_stats(decodingx_interp),
                 'decoding_y': get_stats(decodingy_interp),
                 'probability': get_stats(prob_out),
                 'error_x': get_stats(decodingx_error),
                 'error_y': get_stats(decodingy_error)}

        return {'position_x': posx_interp,
                'position_y': posy_interp,
                'decoding_x': decodingx_interp,
                'decoding_y': decodingy_interp,
                'probability': prob_out,
                'decoding_error_x': decodingx_error,
                'decoding_error_y': decodingy_error,
                'stats': stats}

    def get_decoding_error(self):
        # get decoding error
        time_index = []
        position_mean = []
        for index, trial in time_support_all_trials.iterrows():
            trial_bins = np.arange(trial['start'], trial['end'] + bin_size, bin_size)
            bins = pd.cut(position['y'].index, trial_bins)
            position_mean.append(position['y'].groupby(bins).mean())
            time_index.append(trial_bins[0:-1] + np.diff(trial_bins) / 2)
        time_index = np.hstack(time_index)
        position_means = np.hstack(position_mean)

        actual_series = pd.Series(position_means, index=time_index, name='actual_position')
        decoded_series = decoded.as_series()  # TODO - figure out why decoded and actual series are diff lengths
        df_decode_results = pd.merge(decoded_series.rename('decoded_position'), actual_series, how='left',
                                     left_index=True, right_index=True)
        df_decode_results['decoding_error'] = abs(
            df_decode_results['decoded_position'] - df_decode_results['actual_position'])
        df_decode_results['decoding_error_rolling'] = df_decode_results['decoding_error'].rolling(20,
                                                                                                  min_periods=20).mean()
        df_decode_results['prob_dist'] = [x for x in proby_feature.as_dataframe().to_numpy()]

        # get decoding matrices
        bins = pd.cut(df_decode_results['actual_position'], position_bins)
        decoding_matrix = df_decode_results['decoded_position'].groupby(bins).apply(
            lambda x: np.histogram(x, position_bins, density=True)[0]).values
        decoding_matrix_prob = df_decode_results['prob_dist'].groupby(bins).apply(
            lambda x: np.nanmean(np.vstack(x.values), axis=0)).values

        df_decode_results.to_csv(intermediate_data_path /'decoding_results.csv')
        df_decode_results = pd.read_csv(intermediate_data_path /'decoding_results.csv', index_col=0)

    def plot(self):
        visualizer = BayesianDecoderVisualizer()
        visualizer.plot_decoding_around_update(data_around_switch, dim=self.dim_num)
        visualizer.plot_decoding_around_update(data_around_stay, dim=self.dim_num)

        # plot decoding summary/accuracy
        visualizer.plot_decoding_accuracy()
        visualizer.plot_decoding_summary()

    def export_data(self):
        # export decoding intermediate data
        fname = self.results_io.get_data_filename(filename='preprocess_output', results_type='session', format='npz')
        np.savez(fname, encoder_times=encoder_times, decoder_times=decoder_times, spikes=spikes, features=features)

        fname = self.results_io.get_data_filename(filename='encode_output', results_type='session', format='npz')
        np.savez(fname, model=self.model, bins=self.bins)

        fname = self.results_io.get_data_filename(filename='decode_output', results_type='session', format='npz')
        np.savez(fname, decoded_values=self.decoded_values, prob_densities=self.prob_densities)

        params = ['units_threshold', 'speed_threshold', 'firing_threshold', 'encoder_trial_types', 'encoder_bin_num',
                  'decoder_trial_types', 'decoder_bin_type', 'decoder_bin_size', 'dim_num', 'features']
        kwargs = {p: getattr(self, p) for p in params}
        fname = self.results_io.get_data_filename(filename='params', results_type='session', format='npz')
        np.savez(fname, **kwargs)

    def get_decoding_error_summary(base_path, dimension, n_bins, size_bins, sessions):
        decode_df_list = []
        for name, session in sessions:

            # load file
            session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
            filename = base_path / f'{session_id}.nwb'
            io = NWBHDF5IO(str(filename), 'r')
            nwbfile = io.read()

            # get info for figure saving and labelling
            print(f"Getting error data for {session_id}")
            figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / f'{session_id}'
            Path(figure_path).mkdir(parents=True, exist_ok=True)
            intermediate_data_path = figure_path / 'intermediate_data'

            # split data into train and test
            trial_epochs = nwbfile.intervals['trials'].to_dataframe()
            non_update_epochs = trial_epochs[trial_epochs['update_type'] == 1]
            train_data, test_data = train_test_split(non_update_epochs, test_size=0.25, random_state=21)
            time_support_all = nap.IntervalSet(start=non_update_epochs['start_time'],
                                               end=non_update_epochs['stop_time'],
                                               time_units='s')
            time_support_train = nap.IntervalSet(start=train_data['start_time'], end=train_data['stop_time'],
                                                 time_units='s')
            time_support_test = nap.IntervalSet(start=test_data['start_time'], end=test_data['stop_time'],
                                                time_units='s')  # TODO - figure out why timestamps are not sorted

            # load position structure
            position_ss = nwbfile.processing['behavior']['position'].get_spatial_series('position')
            position = {'x': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 0]),
                        'y': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 1])}
            position = pd.DataFrame.from_dict(position)
            position_tsg = nap.TsdFrame(position, time_units='s', time_support=time_support_all)

            # load units structure
            units = nwbfile.units.to_dataframe()
            spikes_dict = {n: nap.Ts(t=units.loc[n, 'spike_times'], time_units='s') for n in units.index}
            spikes = nap.TsGroup(spikes_dict, time_units='s')

            # decode 1d data
            tuning_curves1d = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg[dimension],
                                                           nb_bins=n_bins,
                                                           ep=time_support_train)
            decoded, proby_feature = nap.decode_1d(tuning_curves=tuning_curves1d,
                                                   group=spikes,
                                                   ep=time_support_test,
                                                   bin_size=size_bins,  # second
                                                   feature=position_tsg[dimension],
                                                   )

            # get decoding error
            time_index = []
            position_mean = []
            for index, trial in time_support_test.iterrows():
                trial_bins = np.arange(trial['start'], trial['end'] + size_bins, size_bins)
                bins = pd.cut(position[dimension].index, trial_bins)
                position_mean.append(position[dimension].groupby(bins).mean())
                time_index.append(trial_bins[0:-1] + np.diff(trial_bins) / 2)
            time_index = np.hstack(time_index)
            position_means = np.hstack(position_mean)

            actual_series = pd.Series(position_means, index=time_index, name='actual_position')
            decoded_series = decoded.as_series()  # TODO - figure out why decoded and actual series are diff lengths
            df_decode_results = pd.merge(decoded_series.rename('decoded_position'), actual_series, how='left',
                                         left_index=True, right_index=True)
            df_decode_results['decoding_error'] = abs(
                df_decode_results['decoded_position'] - df_decode_results['actual_position'])
            df_decode_results['prob_dist'] = [x for x in proby_feature.as_dataframe().to_numpy()]

            # add summary data
            df_positions = df_decode_results[['actual_position', 'decoded_position']].dropna(how='any')
            mse = mean_squared_error(df_positions['actual_position'], df_positions['decoded_position'])
            rmse = sqrt(mse)
            df_decode_results['session_rmse'] = rmse
            df_decode_results['animal'] = name[1]
            df_decode_results['session'] = name[2]

            # append to the list
            decode_df_list.append(df_decode_results)

        # get data from all sessions
        all_decoding_data = pd.concat(decode_df_list, axis=0, ignore_index=True)

        return all_decoding_data