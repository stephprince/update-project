import numpy as np
import pickle
import pandas as pd
import warnings

from math import sqrt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import sem

from update_project.decoding.interpolate import interp1d_time_intervals, griddata_2d_time_intervals, \
    griddata_time_intervals
from update_project.results_io import ResultsIO
from update_project.statistics import get_fig_stats, get_comparative_stats


class BayesianDecoderAggregator:

    def __init__(self, exclusion_criteria=None):
        self.exclusion_criteria = exclusion_criteria
        self.align_times = ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made', 'stop_time']
        self.flip_trials_by_turn = True  # default false
        self.turn_to_flip = 2
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)
        self.data_files = dict(bayesian_aggregator_output=dict(vars=['group_df', 'group_aligned_df'],  format='pkl'),
                               params=dict(vars=['exclusion_criteria', 'align_times', 'flip_trials_by_turn'], format='npz'))

        # times, locs = self._get_example_period()
        # self.start_time, self.end_time = times
        # self.start_loc, self.end_loc = locs
        # self.prob_density_grid = self._get_prob_density_grid()  # prob density plot for example period

    def run_df_aggregation(self, data, overwrite=False, window=5):
        if overwrite:
            # aggregate session data
            for sess_dict in data:
                # get aggregate data and add to session dictionary
                bins = [[-1, 0, 1] if sess_dict['decoder'].convert_to_binary else sess_dict['decoder'].bins][0]
                summary_df = self._summarize(sess_dict['decoder'])
                session_error = self._get_session_error(sess_dict['decoder'], summary_df)
                session_aggregate_dict = dict(aligned_data=self._align_by_times(sess_dict['decoder'], window=window),
                                              summary_df=summary_df,
                                              confusion_matrix=self._get_confusion_matrix(summary_df, bins),
                                              confusion_matrix_sum=session_error['confusion_matrix_sum'],
                                              rmse=session_error['rmse'],
                                              raw_error=session_error['raw_error_median'],
                                              num_units=len(sess_dict['decoder'].spikes),
                                              num_trials=len(sess_dict['decoder'].train_df),
                                              excluded_session=self._meets_exclusion_criteria(sess_dict['decoder']),)
                metadata_keys = ['encoder_bin_num', 'bins', 'virtual_track', 'model', 'results_io', 'results_tags',
                                 'convert_to_binary',]
                metadata_dict = {k: getattr(sess_dict['decoder'], k) for k in metadata_keys}
                sess_dict.update({**session_aggregate_dict, **metadata_dict})

            # get group dataframe
            group_df_raw = pd.DataFrame(data)
            self.group_df = group_df_raw[~group_df_raw['excluded_session']]  # only keep non-excluded sessions
            self.group_df.drop('decoder', axis='columns', inplace=True)  # remove decoding section bc can't pickle h5py

            # get aligned dataframe:
            self.group_aligned_df = self._get_aligned_data(self.group_df)
            if self.flip_trials_by_turn:
                self._flip_aligned_trials()

            self._export_data()
        else:
            if self.results_io.data_exists(self.data_files) and self._params_match():
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those group parameters does not exist, setting overwrite to True')
                self.run_df_aggregation(data, overwrite=True)

    def _params_match(self):
        params_path = self.results_io.get_data_filename(f'params', format='npz')
        params_cached = np.load(params_path, allow_pickle=True)
        params_matched = []
        for k, v in params_cached.items():
            params_matched.append(getattr(self, k) == v)

        return all(params_matched)

    def get_group_confusion_matrices(self, param_name, param_data):
        # get parameter specific data (confusion matrix, track, bins)
        group_summary_df = self._get_group_combined_df(param_data)
        if param_data['convert_to_binary'].values[0]:
            bins = [-1, 0, 1]
        else:
            bins = param_data['bins'].values[0]
        vmax = 2
        virtual_track = param_data['virtual_track'].values[0]
        confusion_matrix = self._get_confusion_matrix(group_summary_df, bins)
        confusion_matrix_sum = self._get_confusion_matrix_sum(confusion_matrix)
        locations = virtual_track.cue_end_locations.get(param_data['feature'].values[0], dict())

        return dict(confusion_matrix=confusion_matrix, confusion_matrix_sum=confusion_matrix_sum, locations=locations,
                    bins=bins, vmax=vmax, param_values=param_name)

    def _flip_aligned_trials(self):
        trials_to_flip = self.group_aligned_df[self.group_aligned_df['turn_type'] == self.turn_to_flip]

        if np.size(trials_to_flip):
            assert len(self.group_aligned_df['feature_name'].unique()) == 1, 'More than one feature detected - not supported'
            feat = self.group_aligned_df['feature_name'].values[0]
            if feat == 'y_position':
                feat_bins = self.group_aligned_df['bins'].values[0]
                virtual_track = self.group_aligned_df['virtual_track'].values[0]
                bounds = virtual_track.choice_boundaries.get(feat, dict())

                trials_to_flip = trials_to_flip.apply(
                    lambda x: self._flip_y_position(x, bounds) if x.name in ['feature', 'decoding'] else x)
                trials_to_flip = trials_to_flip.apply(
                    lambda x: self._flip_y_position(x, bounds, feat_bins) if x.name in ['probability'] else x)
                self.group_aligned_df.loc[self.group_aligned_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
            else:
                feat_before_flip = trials_to_flip['feature'].values[0][0]
                prob_before_flip = trials_to_flip['probability'].values[0][0][0]
                trials_to_flip = trials_to_flip.apply(lambda x: x * -1 if x.name in ['feature', 'decoding'] else x)
                trials_to_flip['probability'] = trials_to_flip['probability'].apply(lambda x: np.flipud(x))
                self.group_aligned_df.loc[self.group_aligned_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
                if ~np.isnan(feat_before_flip):
                    assert feat_before_flip == -trials_to_flip['feature'].values[0][0], 'Data not correctly flipped'
                if ~np.isnan(prob_before_flip):
                    assert prob_before_flip == trials_to_flip['probability'].values[0][-1][
                        0], 'Data not correctly flipped'


    def select_group_aligned_data(self, param_data, filter_dict, ret_df=False):
        # filter for specific features
        mask = pd.concat([param_data[k].isin(v) for k, v in filter_dict.items()], axis=1).all(axis=1)
        data_subset = param_data[mask]

        if np.size(data_subset):
            times = data_subset['times'].values[0]

            field_names = ['feature', 'decoding', 'error']
            group_data = {n: np.vstack(data_subset[n]) for n in field_names}
            group_data.update(probability=np.stack(data_subset['probability']))
            group_data.update(times=times)
            group_data.update(stats={k: get_fig_stats(v, axis=0) for k, v in group_data.items()})
        else:
            group_data = None
            data_subset = None

        if ret_df:
            return data_subset
        else:
            return group_data

    def calc_theta_phase_data(self, param_data, filter_dict, time_bins=3):
        # get aligned df and apply filters
        mask = pd.concat([param_data[k].isin(v) for k, v in filter_dict.items()], axis=1).all(axis=1)
        data_subset = param_data[mask].rename_axis('trial').reset_index()
        data_subset['probability'] = data_subset['probability'].apply(lambda x: x.T)

        # break down so each row has a single theta phase/amplitude value
        data_to_explode = ['feature', 'decoding', 'error', 'probability', 'theta_phase', 'theta_amplitude', 'times']
        theta_phase_df = data_subset.explode(data_to_explode).reset_index(drop=True)
        theta_phase_df.dropna(axis='rows', inplace=True)  # TODO - get theta phase values aligned to center not start

        # get integrated probability densities for each brain region
        virtual_track = param_data['virtual_track'].values[0]
        prob_map_bins = param_data['bins'].values[0]
        choice_bounds = virtual_track.choice_boundaries.get(param_data['feature_name'].values[0], dict())
        home_bounds = virtual_track.home_boundaries.get(param_data['feature_name'].values[0], dict())
        bounds = dict(**choice_bounds, home=home_bounds)
        bound_mapping = dict(left='initial_stay', right='switch', home='home')
        for b_name, b_value in bounds.items():
            theta_phase_df[bound_mapping[b_name]] = theta_phase_df['probability'].apply(
                lambda x: self._integrate_prob_density(x, prob_map_bins, b_value))

        # get histogram, ratio, and mean probability values for different theta phases
        theta_bins = dict(full=np.linspace(-np.pi, np.pi, 12), half=np.linspace(-np.pi, np.pi, 3))
        time_bins = np.linspace(-3, 3, time_bins)  # TODO - change to grab from window periods
        data_to_average = [*list(bound_mapping.values()), 'theta_amplitude']
        theta_phase_list = []
        for t_name, t_bins in theta_bins.items():
            theta_df_bins = pd.cut(theta_phase_df['theta_phase'], t_bins)
            time_df_bins = pd.cut(theta_phase_df['times'], time_bins, right=False)
            if not theta_phase_df.empty:
                mean = theta_phase_df[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(lambda x: np.mean(x))
                err_upper = theta_phase_df[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(
                    lambda x: np.mean(x) + sem(x.astype(float)))
                err_lower = theta_phase_df[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(
                    lambda x: np.mean(x) - sem(x.astype(float)))
                t_df = err_lower.join(err_upper, lsuffix='_err_lower', rsuffix='_err_upper')
                t_df = mean.join(t_df).reset_index()
                t_df['phase_mid'] = t_df['theta_phase'].astype('interval').apply(lambda x: x.mid)
                t_df['time_mid'] = t_df['times'].astype('interval').apply(lambda x: x.mid)

                for g_name, g_value in t_df.groupby('times'):
                    if g_value['time_mid'].values[0] < 0:
                        times_label = 'pre'
                    elif g_value['time_mid'].values[0] > 0:
                        times_label = 'post'
                    g_value['bin_name'] = t_name
                    g_value['times'] = times_label
                    g_value['time_bins'] = [time_bins] * len(g_value)
                    theta_phase_list.append(g_value)

        if len(theta_phase_list):
            theta_phase_output = pd.concat(theta_phase_list, axis=0, ignore_index=True)
        else:
            theta_phase_output = None

        return theta_phase_output

    def _get_example_period(self, window=500):
        time_window = self.data.decoder_bin_size * window
        times = [self.data.summary_df['decoding_error_rolling'].idxmin()]  # get times/locs with minimum error
        locs = [self.data.summary_df.index.searchsorted(times[0])]
        if (times[0] + time_window) < self.data.summary_df.index.max():
            locs.append(self.data.summary_df.index.searchsorted(times[0] + time_window))
        else:
            locs.append(self.data.summary_df.index.searchsorted(times[0] - time_window))
        times.append(self.data.summary_df.iloc[locs[1]].name)
        times.sort()
        locs.sort()

        if np.max(locs) - np.min(locs) <= 1:  # try with a larger window if the bins fall in a weird place
            times, locs = self._get_example_period(window=1000)

        return times, locs

    def _get_prob_density_grid(self):
        # something still off - the big jumps are taking up too any bins, everything is a little slow
        # check that end time is getting us all the way
        nbins = int((self.end_time - self.start_time) / self.data.decoder_bin_size)
        trials_to_flip = self.data.test_df['turn_type'] == 100  # set all to false
        time_bins = np.linspace(self.start_time, self.end_time, nbins)  # time bins
        grid_prob = griddata_time_intervals(self.data.decoded_probs, [self.start_loc], [self.end_loc], nbins,
                                            trials_to_flip, method='nearest', time_bins=time_bins)

        return np.squeeze(grid_prob)

    @staticmethod
    def _integrate_prob_density(prob_density, prob_density_bins, bounds, axis=0):
        prob_map_bins = (prob_density_bins[1:] + prob_density_bins[:-1]) / 2
        start_bin = np.searchsorted(prob_map_bins, bounds[0])
        stop_bin = np.searchsorted(prob_map_bins, bounds[1])
        mean_prob = np.nansum(prob_density[start_bin:stop_bin], axis=axis)  # (trials, feature bins, window)
        # TODO - switch this to np.nanmean to get prob / chance values when I multiply all by total # of bins
        return mean_prob

    @staticmethod
    def quantify_aligned_data(param_data, aligned_data, ret_df=False):
        if np.size(aligned_data) and aligned_data is not None:
            # get bounds to use to quantify choices
            bins = param_data['bins'].values[0]
            virtual_track = param_data['virtual_track'].values[0]
            bounds = virtual_track.choice_boundaries.get(param_data['feature_name'].values[0], dict())

            if isinstance(aligned_data, dict):
                prob_map = aligned_data['probability']
            else:
                prob_map = np.stack(aligned_data['probability'])
            prob_choice = dict()
            num_bins = dict()
            output_list = []
            for bound_name, bound_values in bounds.items():  # loop through left/right bounds
                prob_map_bins = (bins[1:] + bins[:-1]) / 2
                start_bin = np.searchsorted(prob_map_bins, bound_values[0])
                stop_bin = np.searchsorted(prob_map_bins, bound_values[1])

                threshold = 0.1  # total probability density to call a left/right choice
                integrated_prob = np.nansum(prob_map[:, start_bin:stop_bin, :], axis=1)  # (trials, feature bins, window)
                num_bins[bound_name] = len(prob_map[0, start_bin:stop_bin, 0])
                assert len(bins) - 1 == np.shape(prob_map)[1], 'Bound bins not being sorted on right dimension'

                bound_quantification = dict(prob_sum=integrated_prob,  # (trials x window_bins)
                                            thresh_crossing=integrated_prob > threshold)  # (trials x window_bins)
                bound_quantification.update(stats={k: get_fig_stats(v, axis=0) for k, v in bound_quantification.items()})
                bound_quantification.update(bound_values=bound_values,
                                            threshold=threshold)
                prob_choice[bound_name] = bound_quantification  # choice calculating probabilities for

                if ret_df:  # if returning dataframe compile data accordingly
                    bound_quantification.update(dict(choice=bound_name, trial_index=aligned_data.index.to_numpy()))
                    output_list.append(bound_quantification)

            assert len(np.unique(list(num_bins.values()))) == 1, 'Number of bins for different bounds are not equal'

            if ret_df:
                list_df = pd.DataFrame(output_list)[['trial_index', 'prob_sum', 'bound_values', 'choice']]
                list_df = list_df.explode(['prob_sum', 'trial_index']).reset_index(drop=True).set_index('trial_index')
                output_df = pd.merge(aligned_data[['session_id', 'animal', 'time_label', 'times']], list_df, left_index=True,
                                     right_index=True)
                output_df['choice'] = output_df['choice'].map(dict(left='initial_stay', right='switch'))
                return output_df
            else:
                return prob_choice
        else:
            warnings.warn('No aligned data found')
            return None

    def calc_trial_by_trial_quant_data(self, param_data, plot_groups):
        group_aligned_data = self.select_group_aligned_data(param_data, plot_groups, ret_df=True)
        quant_df = self.quantify_aligned_data(param_data, group_aligned_data, ret_df=True)

        if np.size(quant_df):
            # get diff from baseline
            prob_sum_mat = np.vstack(quant_df['prob_sum'])
            align_time = np.argwhere(quant_df['times'].values[0] == 0)[0][0]
            prob_sum_diff = prob_sum_mat.T - prob_sum_mat[:, align_time]
            quant_df['diff_baseline'] = list(prob_sum_diff.T)

            # get diff from left vs. right bounds
            quant_df['trial_index'] = quant_df.index
            quant_df = quant_df.explode(['times', 'prob_sum', 'diff_baseline']).reset_index()
            quant_df['times_binned'] = pd.cut(quant_df['times'], np.linspace(0, 2.5, 3)).apply(lambda x: x.mid)
            quant_df = pd.DataFrame(quant_df.to_dict())  # fix to avoid object dtype errors in seaborn

            choice_df = quant_df.pivot(index=['session_id', 'animal', 'time_label', 'times', 'times_binned', 'trial_index'],
                                       columns=['choice'],
                                       values=['prob_sum', 'diff_baseline']).reset_index()  # had times here before
            choice_df['diff_switch_stay'] = choice_df[('prob_sum', 'switch')] - choice_df[('prob_sum', 'initial_stay')]
            choice_df.columns = ['_'.join(c) if c[1] != '' else c[0] for c in choice_df.columns.to_flat_index()]

            return quant_df, choice_df
        else:
            return None, None

    @staticmethod
    def get_tuning_data(param_data):
        tuning_curve_list = []
        for _, sess_data in param_data.iterrows():
            for key, unit_tuning in sess_data['model'].items():
                tuning_curve_list.append(dict(animal=sess_data['results_io'].animal,
                                              session=sess_data['results_io'].session_id,
                                              unit=key,
                                              tuning_curve=unit_tuning,
                                              bins=sess_data['bins']))

        tuning_curve_df = pd.DataFrame(tuning_curve_list)

        return tuning_curve_df

    @staticmethod
    def _get_aligned_data(param_data):
        cols_to_keep = ['session_id', 'animal', 'region', 'bins', 'encoder_bins', 'decoder_bins', 'virtual_track', 'aligned_data']
        temp_data = param_data[cols_to_keep].explode('aligned_data').reset_index(drop=True)
        aligned_data = pd.json_normalize(temp_data['aligned_data'], max_level=0).drop(['stats'], axis='columns')
        aligned_data_df = pd.concat([temp_data, aligned_data], axis=1)
        aligned_data_df.drop(['aligned_data'], axis='columns', inplace=True)  # remove the extra aligned data column

        data_to_explode = ['feature', 'decoding', 'error', 'probability', 'turn_type', 'correct', 'theta_phase',
                           'theta_amplitude']
        exploded_df = aligned_data_df.explode(data_to_explode).reset_index(drop=True)
        exploded_df.dropna(axis='rows', inplace=True)

        return exploded_df

    @staticmethod
    def _flip_y_position(data, bounds, bins=None):
        flipped_data = []
        for ind, row in data.iteritems():
            if bins is not None:
                areas = dict()
                for bound_name, bound_values in bounds.items():  # loop through left/right bounds
                    prob_map_bins = (bins[1:] + bins[:-1]) / 2
                    areas[f'{bound_name}_start'] = np.searchsorted(prob_map_bins, bound_values[0])
                    areas[f'{bound_name}_stop'] = np.searchsorted(prob_map_bins, bound_values[1])

                left_data = row[areas['left_start']:areas['left_stop'], :].copy()
                right_data = row[areas['right_start']:areas['right_stop'], :].copy()

                row[areas['left_start']:areas['left_stop'], :] = right_data
                row[areas['right_start']:areas['right_stop'], :] = left_data
            else:
                offset = bounds['right'][0] - bounds['left'][0]
                left_data = np.logical_and(row > bounds['left'][0], row < bounds['left'][1])
                right_data = np.logical_and(row > bounds['right'][0], row < bounds['right'][1])
                row[left_data] = row[left_data] + offset
                row[right_data] = row[right_data] - offset
            flipped_data.append(row)

        return flipped_data

    @staticmethod
    def _get_confusion_matrix_sum(confusion_matrix):
        num_bins_to_sum = int(
            (len(confusion_matrix) / 10 - 1) / 2)  # one tenth of the track minus the identity line bin
        values_to_sum = []
        for i in range(len(confusion_matrix)):
            values_to_sum.append(confusion_matrix[i, i])  # identity line values
            if i < len(confusion_matrix) - num_bins_to_sum:
                values_to_sum.append(confusion_matrix[i + num_bins_to_sum, i])  # bins above
                values_to_sum.append(confusion_matrix[i, i + num_bins_to_sum])  # bins to right
        confusion_matrix_sum = np.nansum(values_to_sum)

        return confusion_matrix_sum

    @staticmethod
    def _get_mean_prob_dist(probabilities, bins):
        if probabilities.empty:
            mean_prob_dist = np.empty(np.shape(bins[1:]))  # remove one bin to match other probabilities shapes
            mean_prob_dist[:] = np.nan
        else:
            mean_prob_dist = np.nanmean(np.vstack(probabilities.values), axis=0)

        return mean_prob_dist

    @staticmethod
    def _get_group_combined_df(data):
        # get giant dataframe of all decoding data
        summary_df_list = []
        for _, sess_data in data.iterrows():
            summary_df_list.append(sess_data['summary_df'])
        group_summary_df = pd.concat(summary_df_list, axis=0, ignore_index=True)

        return group_summary_df

    @staticmethod
    def _summarize(decoder):
        # get decoding error
        time_index = []
        feature_mean = []
        for index, trial in decoder.decoder_times.iterrows():
            trial_bins = np.arange(trial['start'], trial['end'] + decoder.decoder_bin_size, decoder.decoder_bin_size)
            bins = pd.cut(decoder.features_test.index, trial_bins)
            feature_mean.append(decoder.features_test.iloc[:, 0].groupby(bins).mean())
            time_index.append(trial_bins[0:-1] + np.diff(trial_bins) / 2)
        time_index = np.hstack(time_index)
        feature_means = np.hstack(feature_mean)

        actual_series = pd.Series(feature_means, index=np.round(time_index, 4), name='actual_feature')
        if decoder.decoded_values.any().any():
            decoded_series = decoder.decoded_values.as_series()
        else:
            decoded_series = pd.Series()
        df_decode_results = pd.merge(decoded_series.rename('decoded_feature'), actual_series, how='left',
                                     left_index=True, right_index=True)
        df_decode_results['decoding_error'] = abs(
            df_decode_results['decoded_feature'] - df_decode_results['actual_feature'])
        df_decode_results['decoding_error_rolling'] = df_decode_results['decoding_error'].rolling(20,
                                                                                                  min_periods=20).mean()
        if decoder.decoded_probs.any().any():
            df_decode_results['prob_dist'] = [x for x in decoder.decoded_probs.as_dataframe().to_numpy()]
            df_positions = df_decode_results[['actual_feature', 'decoded_feature']].dropna(how='any')
            rmse = sqrt(mean_squared_error(df_positions['actual_feature'], df_positions['decoded_feature']))
        else:
            df_decode_results['prob_dist'] = [x for x in decoder.decoded_probs.to_numpy()]
            rmse = np.nan

        # add summary data
        df_decode_results['session_rmse'] = rmse
        df_decode_results['animal'] = decoder.results_io.animal
        df_decode_results['session'] = decoder.results_io.session_id

        return df_decode_results

    def _meets_exclusion_criteria(self, data):
        exclude_session = False  # default to include session

        # apply exclusion criteria
        units_threshold = self.exclusion_criteria.get('units', 0)
        if len(data.spikes) < units_threshold:
            exclude_session = True

        trials_threshold = self.exclusion_criteria.get('trials', 0)
        if len(data.train_df) < trials_threshold:
            exclude_session = True

        return exclude_session

    def _get_confusion_matrix(self, data, bins):
        if len(bins):
            df_bins = pd.cut(data['actual_feature'], bins, include_lowest=True)
            decoding_matrix = data['prob_dist'].groupby(df_bins).apply(
                lambda x: self._get_mean_prob_dist(x, bins)).values

            confusion_matrix = np.vstack(decoding_matrix).T  # transpose so that true position is on the x-axis
        else:
            confusion_matrix = []

        return confusion_matrix

    def _align_by_times(self, decoder, window):
        nbins = int(window*2/decoder.decoder_bin_size)

        trial_type_dict = dict(non_update=1, switch=2, stay=3)
        output = []
        for trial_name in ['non_update', 'switch', 'stay']:
            trials_to_agg = decoder.test_df[decoder.test_df['update_type'] == trial_type_dict[trial_name]]

            for time_label in self.align_times[:-1]:  # skip last align times so only until stop of trial
                window_start, window_stop = window, window
                if time_label == 't_choice_made':
                    window_stop = 0  # if choice made, don't grab data past bc could be end of trial
                elif time_label == 'start_time':
                    window_start = 0
                new_times = np.linspace(-window_start, window_stop, num=nbins)

                mid_times = trials_to_agg[time_label]
                turns = trials_to_agg['turn_type'][~mid_times.isna()].values
                outcomes = trials_to_agg['correct'][~mid_times.isna()].values
                mid_times.dropna(inplace=True)

                phase_col = np.argwhere(decoder.theta.columns == 'phase')[0][0]
                amp_col = np.argwhere(decoder.theta.columns == 'amplitude')[0][0]
                vars = dict(feature=decoder.features_test.iloc[:, 0], decoded=decoder.decoded_values,
                            probability=decoder.decoded_probs, theta_phase=decoder.theta.iloc[:, phase_col],
                            theta_amplitude=decoder.theta.iloc[:, amp_col])
                locs = self._get_start_stop_locs(vars, mid_times, window_start, window_stop)

                interp_dict = dict()
                for name in decoder.feature_names:
                    interp_dict[name] = dict()
                    if decoder.decoded_values.any().any() and mid_times.any():
                        for key, val in vars.items():
                            if decoder.dim_num == 1 and key == 'probability':
                                interp_dict[name][key] = griddata_time_intervals(val, locs[key]['start'],
                                                                                 locs[key]['stop'], nbins, mid_times)
                            elif decoder.dim_num == 2 and key == 'probability':
                                interp_dict[name][key] = griddata_2d_time_intervals(val, decoder.bins,
                                                                               decoder.decoding_values.index.values,
                                                                               locs[key]['start'], locs[key]['stop'],
                                                                               mid_times, nbins)
                            else:
                                interp_dict[name][key] = interp1d_time_intervals(val, locs[key]['start'],
                                                                                 locs[key]['stop'],
                                                                                 new_times, mid_times)
                        interp_dict[name]['error'] = [abs(dec_feat - true_feat) for true_feat, dec_feat in
                                                zip(interp_dict[name]['feature'], interp_dict[name]['decoded'])]

                    else:
                        interp_dict[name] = {k: [] for k in [*list(vars.keys()), 'error']}
                        turns = []
                        outcomes = []

                # get means and sem
                data = dict()
                for name in decoder.feature_names:
                    assert np.shape(np.array(interp_dict[name]['feature']))[0] == np.shape(turns)[0]
                    data = dict(feature_name=name,
                                update_type=trial_name,
                                time_label=time_label,
                                feature=np.array(interp_dict[name]['feature']),
                                decoding=np.array(interp_dict[name]['decoded']),
                                error=np.array(interp_dict[name]['error']),
                                probability=interp_dict[name]['probability'],
                                theta_phase=interp_dict[name]['theta_phase'],
                                theta_amplitude=interp_dict[name]['theta_amplitude'],
                                turn_type=turns,
                                correct=outcomes,
                                window_start=-window_start,
                                window_stop=window_stop,
                                window=window,
                                nbins=nbins,
                                times=new_times)
                    data.update(stats={k: get_fig_stats(v, axis=1) for k, v in data.items()
                                       if k in ['feature', 'decoding', 'error']})
                output.append(data)

        return output

    @staticmethod
    def _get_start_stop_locs(vars, mid_times, window_start, window_stop):
        locs = dict()
        for k, v in vars.items():
            # add extra index step to stop locs for interpolation and go one index earlier for start locs
            start_locs = v.index.searchsorted(mid_times - window_start) - 1
            start_locs[start_locs < 0] = 0  # catch for cases too close to start of trial
            stop_locs = v.index.searchsorted(mid_times + window_stop) + 1
            locs[k] = dict(start=start_locs, stop=stop_locs)

        return locs

    def _get_session_error(self, data, summary_df):
        # get error from heatmap
        if data.convert_to_binary:
            bins = [-1, 0, 1]
        else:
            bins = data.bins
        confusion_matrix = self._get_confusion_matrix(summary_df, bins)
        confusion_matrix_sum = self._get_confusion_matrix_sum(confusion_matrix)

        # get error from
        rmse = summary_df['session_rmse'].mean()
        raw_error_median = summary_df['decoding_error'].median()
        session_error = dict(confusion_matrix_sum=confusion_matrix_sum,
                             rmse=rmse,
                             raw_error_median=raw_error_median)

        return session_error

    def get_aligned_stats(self, comp, data_dict, quant, tags):
        data_for_stats = []
        for ind, v in enumerate(data_dict['comparison']):
            data = data_dict['data'][data_dict['data'][comp] == v]['data'].values[0]
            initial_data = dict(prob_sum=quant['left']['prob_sum'], bound='initial', comparison=comp, group=v)
            new_data = dict(prob_sum=quant['right']['prob_sum'], bound='new', comparison=comp, group=v)
            data_for_stats.extend([initial_data, new_data])

        # add significance stars to sections of plot
        prob_sum_df = pd.DataFrame(data_for_stats)
        prob_sum_df = prob_sum_df.explode('prob_sum').reset_index(drop=True)
        bins_to_grab = [0, len(data['times'])]
        times = data['times'][bins_to_grab[0]:bins_to_grab[1]]
        stars_to_plot = dict(initial=[], new=[])
        blanks_to_plot = dict(initial=[], new=[])
        for b_ind, b in enumerate(range(bins_to_grab[0], bins_to_grab[1])):
            prob_sum_df['data'] = prob_sum_df['prob_sum'].apply(lambda x: np.nansum(x[b]))
            temp_df = prob_sum_df[['bound', 'comparison', 'group', 'data']].explode('data').reset_index(
                drop=True)
            df = pd.DataFrame(temp_df.to_dict())
            for n, group in df.groupby(['comparison', 'bound']):
                data_to_compare = {'_'.join((*n, str(v))): group[group['group'] == v]['data'].values for v in
                                   list(group['group'].unique())}
                comp_stats = get_comparative_stats(*data_to_compare.values())
                self.results_io.export_statistics(data_to_compare,
                                                  f'aligned_data_{"_".join(n)}_stats_{tags}_bin{b_ind}')
                if comp_stats['ranksum']['p_value'] < 0.05:
                    stars_to_plot[n[1]].append(times[b_ind])
                else:
                    blanks_to_plot[n[1]].append(times[b_ind])

        return data_for_stats, dict(sig=stars_to_plot, ns=blanks_to_plot)

    def _load_data(self):
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, format=file_info['format'])

            import_data = self.results_io.load_pickled_data(fname)
            for v, data in zip(file_info['vars'], import_data):
                setattr(self, v, data)

        return self

    def _export_data(self):
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, format=file_info['format'])

            with open(fname, 'wb') as f:
                [pickle.dump(getattr(self, v), f) for v in file_info['vars']]