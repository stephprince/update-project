import itertools
import numpy as np
import pickle
import pandas as pd
import warnings
import statsmodels.api as sm

from math import sqrt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn import svm
from sklearn.utils import resample
from scipy.stats import sem, pearsonr, uniform, loguniform
from scipy import signal
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from update_project.general.interpolate import interp1d_time_intervals, griddata_2d_time_intervals, \
    griddata_time_intervals
from update_project.general.results_io import ResultsIO
from update_project.statistics.statistics import Stats, get_fig_stats, get_comparative_stats


class BayesianDecoderAggregator:

    def __init__(self, exclusion_criteria=None, align_times=None, analyzer_name=None, turn_to_flip=2):
        self.exclusion_criteria = exclusion_criteria
        self.align_times = align_times or ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made']
        self.flip_trials_by_turn = True  #
        self.turn_to_flip = turn_to_flip#
        self.analyzer_name = analyzer_name or 'analyzer'
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)
        self.data_files = dict(bayesian_aggregator_output=dict(vars=['group_df', 'group_aligned_df'], format='pkl'),
                               params=dict(vars=['exclusion_criteria', 'align_times', 'flip_trials_by_turn'],
                                           format='npz'))

    def run_df_aggregation(self, data, overwrite=False, window=5):
        # aggregate session data
        self.window = window
        for sess_dict in data:
            # get aggregate data and add to session dictionary
            bins = \
                [[-1, 0, 1] if sess_dict[self.analyzer_name].convert_to_binary else sess_dict[self.analyzer_name].bins][
                    0]
            summary_df = self._summarize(sess_dict[self.analyzer_name])
            session_error = self._get_session_error(sess_dict[self.analyzer_name], summary_df)
            if hasattr(sess_dict[self.analyzer_name].units_types, 'any'):
                region = tuple(sess_dict[self.analyzer_name].units_types.any()['region'])
            else:
                region = tuple(sess_dict[self.analyzer_name].units_types['region'])
            session_aggregate_dict = dict(
                aligned_data=self._align_by_times(sess_dict[self.analyzer_name], window=window),
                summary_df=summary_df,
                confusion_matrix=self._get_confusion_matrix(summary_df, bins),
                confusion_matrix_sum=session_error['confusion_matrix_sum'],
                rmse=session_error['rmse'],
                raw_error=session_error['raw_error_median'],
                region=region,
                feature=sess_dict[self.analyzer_name].feature_names[0],
                num_units=len(sess_dict[self.analyzer_name].spikes),
                num_trials=len(sess_dict[self.analyzer_name].train_df),
                excluded_session=self._meets_exclusion_criteria(sess_dict, sess_dict[self.analyzer_name]), )
            metadata_keys = ['bins', 'virtual_track', 'model', 'model_test', 'model_delay_only', 'model_update_only',
                             'results_io', 'results_tags', 'convert_to_binary', 'encoder_bin_num', 'decoder_bin_size',
                             'decoder_test_size']#model needs to be singular once done with cv. change back
            metadata_dict = {k: getattr(sess_dict[self.analyzer_name], k) for k in metadata_keys}
            if hasattr(sess_dict[self.analyzer_name].encoder_bin_num, 'item'):
                metadata_dict['encoder_bin_num'] = sess_dict[self.analyzer_name].encoder_bin_num.item()
                metadata_dict['decoder_bin_size'] = sess_dict[self.analyzer_name].decoder_bin_size.item()
                metadata_dict['decoder_test_size'] = sess_dict[self.analyzer_name].decoder_test_size.item()
            sess_dict.update({**session_aggregate_dict, **metadata_dict})

        # get group dataframe
        group_df_raw = pd.DataFrame(data)
        self.group_df = group_df_raw[~group_df_raw['excluded_session']]  # only keep non-excluded sessions
        self.group_df.drop(self.analyzer_name, axis='columns',
                           inplace=True)  # remove decoding section bc can't pickle h5py

        # get aligned dataframe:
        self.group_aligned_df = self._get_aligned_data(self.group_df)
        if self.flip_trials_by_turn:
            self._flip_aligned_trials()

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
        flipped_data = []
        for feat, feat_df in self.group_aligned_df.groupby('feature_name'):
            trials_to_flip = feat_df[feat_df['turn_type'] == self.turn_to_flip]
            if np.size(trials_to_flip):
                if feat == 'y_position':
                    feat_bins = feat_df['bins'].values[0]
                    virtual_track = feat_df['virtual_track'].values[0]
                    bounds = virtual_track.choice_boundaries.get(feat, dict())

                    trials_to_flip = trials_to_flip.apply(
                        lambda x: self._flip_y_position(x, bounds) if x.name in ['feature', 'decoding'] else x)
                    trials_to_flip = trials_to_flip.apply(
                        lambda x: self._flip_y_position(x, bounds, feat_bins) if x.name in ['probability'] else x)
                    trials_to_flip = trials_to_flip.apply(
                        lambda x: x * -1 if x.name in ['rotational_velocity', 'view_angle', 'choice_commitment'] else x)
                    feat_df.loc[feat_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
                else:
                    cols_to_flip = ['feature', 'decoding', 'rotational_velocity', 'view_angle', 'choice_commitment']
                    feat_before_flip = trials_to_flip['feature'].values[0][0]
                    prob_before_flip = trials_to_flip['probability'].values[0][0][0]
                    trials_to_flip = trials_to_flip.apply(lambda x: x * -1 if x.name in cols_to_flip else x)
                    trials_to_flip['probability'] = trials_to_flip['probability'].apply(lambda x: np.flipud(x))
                    feat_df.loc[feat_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
                    if ~np.isnan(feat_before_flip):
                        assert feat_before_flip == -trials_to_flip['feature'].values[0][0], 'Data not correctly flipped'
                    if ~np.isnan(prob_before_flip):
                        assert prob_before_flip == trials_to_flip['probability'].values[0][-1][
                            0], 'Data not correctly flipped'

            flipped_data.append(feat_df)

        self.group_aligned_df = pd.concat(flipped_data, axis=0)
        self.group_aligned_df.sort_index(inplace=True)

    def select_group_aligned_data(self, param_data, filter_dict, ret_df=False, summary=False):
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
        elif summary:
            return group_data, summary_df
        else:
            return group_data

    def calc_choice_commitment_data(self, param_data, plot_groups, prob_value='prob_sum', time_window=(-2.5, 2.5),
                                    n_time_bins=11, quantiles=4):
        # combine choice commitment with aligned df
        group_aligned_data = self.select_group_aligned_data(param_data, plot_groups, ret_df=True)
        quant_df = self.quantify_aligned_data(param_data, group_aligned_data, ret_df=True)
        combined_df = pd.merge(quant_df, group_aligned_data[['choice_commitment', 'view_angle']], left_index=True,
                               right_index=True, validate='many_to_one')

        if np.size(combined_df) and combined_df is not None:
            # get diff from baseline (t=0) and choice at (t=0)
            align_time = np.argwhere(combined_df['times'].values[0] >= 0)[0][0]
            if align_time != 0:
                align_time = align_time - 1  # get index immediately preceding 0 if not the first one

            prob_sum_mat = np.vstack(combined_df[prob_value])
            prob_sum_diff = prob_sum_mat.T - prob_sum_mat[:, align_time]
            combined_df['diff_baseline'] = list(prob_sum_diff.T)

            choice_commitment_mat = np.vstack(combined_df['choice_commitment'])
            view_angle_mat = np.vstack(combined_df['view_angle'])
            combined_df['choice_commitment_at_0'] = choice_commitment_mat[:, align_time]
            combined_df['view_angle_at_0'] = view_angle_mat[:, align_time]

            # get diff from left vs. right bounds
            combined_df['trial_index'] = combined_df.index
            combined_df = (combined_df
                           .explode(['times', 'prob_sum', 'prob_over_chance', 'diff_baseline', 'choice_commitment',
                                     'view_angle'])
                           .reset_index(drop=True))
            combined_df['times_binned'] = pd.cut(combined_df['times'], np.linspace(*time_window, n_time_bins)).apply(
                lambda x: x.mid)
            combined_df = pd.DataFrame(combined_df.to_dict())  # fix to avoid object dtype errors in seaborn

            combined_df['choice_commitment_quantile'] = pd.qcut(combined_df['choice_commitment_at_0'], q=quantiles,
                                                                labels=[f'q{i}' for i in range(quantiles)])
            combined_df['choice_commitment_pos_or_neg'] = pd.cut(combined_df['choice_commitment_at_0'], (-1, 0, 1),
                                                                labels=['positive', 'negative'])
            combined_df['view_angle_quantile'] = pd.qcut(combined_df['view_angle_at_0'], q=quantiles,
                                                         labels=[f'q{i}' for i in range(quantiles)][::-1])  # reversed
            combined_df['view_angle_pos_or_neg'] = pd.cut(combined_df['view_angle_at_0'], (-1, 0, 1),
                                                         labels=['negative', 'positive'])  # reversed

            return combined_df
        else:
            return None

    def calc_theta_phase_data(self, param_data, filter_dict, n_time_bins=3, ret_by_trial=False,
                              time_window=(-1.5, 1.5)):
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
        bounds = dict(**choice_bounds)
        bounds['home'] = virtual_track.home_boundaries.get(param_data['feature_name'].values[0], dict())
        bound_mapping = dict(left='initial', right='new', home='central')
        for b_name, b_value in bounds.items():
            theta_phase_df[bound_mapping[b_name]] = theta_phase_df['probability'].apply(
                lambda x: self._integrate_prob_density(x, prob_map_bins, b_value))

        # get histogram, ratio, and mean probability values for different theta phases
        theta_bins = dict(full=np.linspace(-np.pi, np.pi, 12), quarter=np.linspace(-np.pi, np.pi, 5),
                          half=np.linspace(-np.pi, np.pi, 3))
        time_bins = np.linspace(time_window[0], time_window[-1], n_time_bins)
        data_to_average = [*list(bound_mapping.values()), 'theta_amplitude']
        theta_phase_list = []
        theta_phase_by_trial_list = []
        for t_name, t_bins in theta_bins.items():
            theta_df_bins = pd.cut(theta_phase_df['theta_phase'], t_bins)
            time_df_bins = pd.cut(theta_phase_df['times'], time_bins, right=False)
            if not theta_phase_df.empty:
                mean = theta_phase_df[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(
                    lambda x: np.mean(x))
                err_upper = theta_phase_df[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(
                    lambda x: np.mean(x) + sem(x.astype(float)))
                err_lower = theta_phase_df[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(
                    lambda x: np.mean(x) - sem(x.astype(float)))
                prob_df = (theta_phase_df
                           .groupby([time_df_bins, theta_df_bins])['probability']
                           .apply(lambda x: np.nanmean(np.vstack(x), axis=0)))
                t_df = err_lower.join(err_upper, lsuffix='_err_lower', rsuffix='_err_upper')
                t_df = mean.join(t_df)
                t_df = t_df.join(prob_df).reset_index()
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

                theta_phase_df['time_bins'] = time_df_bins
                theta_phase_df['phase_bins'] = theta_df_bins
                groupby_cols = ['animal', 'session_id', 'region', 'feature_name', 'trial_id', 'update_type', 'correct',
                                'time_bins', 'phase_bins']
                t_by_trial_df = (theta_phase_df
                                 .groupby(groupby_cols)[data_to_average]
                                 .agg(np.nanmean)
                                 .dropna()
                                 .reset_index())
                t_by_trial_df['probability'] = (theta_phase_df
                                 .groupby(groupby_cols)['probability']
                                 .apply(lambda x: np.nanmean(np.vstack(x), axis=0))
                                 .dropna()
                                 .reset_index()['probability'])
                t_by_trial_df['theta_amplitude'] = (theta_phase_df
                                                    .groupby(groupby_cols)['theta_amplitude']
                                                    .apply(np.nanmean).dropna().reset_index())['theta_amplitude']
                t_by_trial_df['phase_mid'] = t_by_trial_df['phase_bins'].astype('interval').apply(lambda x: x.mid)
                t_by_trial_df['time_mid'] = t_by_trial_df['time_bins'].astype('interval').apply(lambda x: x.mid)

                for g_name, g_value in t_by_trial_df.groupby('time_mid'):
                    if g_value['time_mid'].values[0] < 0:
                        times_label = 'pre'
                    elif g_value['time_mid'].values[0] > 0:
                        times_label = 'post'
                    g_value['bin_name'] = t_name
                    g_value['times'] = times_label
                    g_value['time_bins'] = [time_bins] * len(g_value)
                    theta_phase_by_trial_list.append(g_value)

        if len(theta_phase_list):
            theta_phase_output = pd.concat(theta_phase_list, axis=0, ignore_index=True)
        else:
            theta_phase_output = None

        if ret_by_trial:
            return pd.concat(theta_phase_by_trial_list, axis=0, ignore_index=True)
        else:
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

    def _integrate_prob_density(self, prob_density, prob_density_bins, bounds, axis=0):
        start_bin, stop_bin = self.get_bound_bins(prob_density_bins, bounds)
        integrated_prob = np.nansum(prob_density[start_bin:stop_bin], axis=axis)  # (trials, feature bins, window)
        # num_bins = len(prob_density[start_bin:stop_bin])
        # prob_over_chance = integrated_prob * len(prob_density_bins) / num_bins#uncomment these lines to normalize maybe. try

        return integrated_prob

    @staticmethod
    def get_bound_bins(bins, bound_values):
        start_bin, stop_bin = np.searchsorted(bins, bound_values)
        if bins[0] == bound_values[0]:
            stop_bin = stop_bin + 1
        if bins[-1] == bound_values[-1]:
            stop_bin = stop_bin + 1
        stop_bin = stop_bin - 1

        return start_bin, stop_bin

    def adjust_bound_bins(self, bins, bounds):
        bin_inds = dict()
        for bound_name, bound_values in bounds.items():  # loop through left/right bounds
            start_bin, stop_bin = self.get_bound_bins(bins, bound_values)
            bin_inds[f'{bound_name}_start'] = start_bin
            bin_inds[f'{bound_name}_stop'] = stop_bin

        virtual_track = self.group_df['virtual_track'].values[0]
        while len(np.arange(bin_inds['left_start'], bin_inds['left_stop'])) != \
                len(np.arange(bin_inds['right_start'], bin_inds['right_stop'])):
            left_bins = bins[bin_inds['left_start']:bin_inds['left_stop'] + 1]
            right_bins = bins[bin_inds['right_start']:bin_inds['right_stop'] + 1]

            if len(left_bins) < len(right_bins) and np.max(left_bins) < np.max(virtual_track.edge_spacing[1]):
                bin_inds['left_stop'] = bin_inds['left_stop'] + 1

            if len(left_bins) > len(right_bins) and np.max(left_bins) >= bounds['left'][-1]:
                bin_inds['left_stop'] = bin_inds['left_stop'] - 1

        return bin_inds

    def quantify_aligned_data(self, param_data, aligned_data, ret_df=False, other_zones=None, half=None, prospective_reps=False, hm=False):
        if np.size(aligned_data) and aligned_data is not None:
            # get bounds to use to quantify choices
            bins = param_data['bins'].values[0]
            if len(bins) == 5:
                bins = bins[0]
            virtual_track = param_data['virtual_track'].values[0]
            bounds = virtual_track.choice_boundaries.get(param_data['feature_name'].values[0], dict())
            other_zones_dict = dict()
            if other_zones:
                if 'central' in other_zones:
                    bounds['home'] = virtual_track.home_boundaries.get(param_data['feature_name'].values[0], dict())
                    if prospective_reps:
                        bounds['home']=(5, 50)
                    other_zones_dict['home'] = 'central'
                if 'original' in other_zones:
                    bounds['original'] = (virtual_track.cue_start_locations['y_position']['initial cue'],
                                         virtual_track.cue_end_locations['y_position']['initial cue'])
                    other_zones_dict['original'] = 'original'
                if 'local' in other_zones:
                    bounds['local'] = virtual_track.home_boundaries.get(param_data['feature_name'].values[0], dict())
                    other_zones_dict['local'] = 'local'

            if self.turn_to_flip == 2:
                choice_mapping = dict(left='initial_stay', right='switch', **other_zones_dict)
            elif self.turn_to_flip == 1:
                choice_mapping = dict(left='switch', right='initial_stay', **other_zones_dict)

            if isinstance(aligned_data, dict):
                prob_map = aligned_data['probability']
            else:
                prob_map = np.stack(aligned_data['probability'])
                local_position_map = np.stack(aligned_data['feature'])
                bin_map = np.searchsorted(bins, local_position_map)
            prob_choice = dict()
            num_bins = dict()
            bin_inds = self.adjust_bound_bins(bins, bounds)
            if bin_inds['left_stop'] == 43:
                bin_inds['home_stop'] = 7
            if half=='backbin':
                bin_inds['right_start'] = bin_inds['right_stop'] - 1
                bin_inds['left_start'] = bin_inds['left_stop']-1
            elif half == 'front':
                bin_inds['right_stop'] = bin_inds['right_stop'] - round((bin_inds['right_stop'] - bin_inds['right_start'])/2)
                bin_inds['left_stop'] = bin_inds['left_stop']-round((bin_inds['left_stop'] - bin_inds['left_start'])/2)
            elif half == 'rearhalf':
                bin_inds['right_start'] = bin_inds['right_start'] + ((bin_inds['right_stop'] - bin_inds['right_start'])//2)
                bin_inds['left_start'] = bin_inds['left_start'] + ((bin_inds['left_stop'] - bin_inds['left_start'])//2)

            output_list = []
            for bound_name, bound_values in bounds.items():  # loop through left/right bounds
                threshold = 0.1  # total probability density to call a left/right choice
                if bound_name == 'local':
                    integrated_prob = []
                    for prob, bi in zip(prob_map, bin_map):
                        local_probs = np.array([np.sum(prob[b - 3: b + 3, i]) for i, b in enumerate(bi)])
                        integrated_prob.append(local_probs)
                    integrated_prob = np.array(integrated_prob)
                else:
                    integrated_prob = np.sum(prob_map[:, bin_inds[f'{bound_name}_start']:bin_inds[f'{bound_name}_stop'], :],
                                                axis=1)  # (trials, feature bins, window)
                num_bins[bound_name] = len(prob_map[0, bin_inds[f'{bound_name}_start']:bin_inds[f'{bound_name}_stop'], 0])
                prob_over_chance = integrated_prob * len(bins) / num_bins[
                    bound_name]  # integrated prob * bins / goal bins
                assert len(bins) - 1 == np.shape(prob_map)[1], 'Bound bins not being sorted on right dimension'

                bound_quantification = dict(prob_sum=integrated_prob,  # (trials x window_bins)
                                            prob_over_chance=prob_over_chance,
                                            thresh_crossing=integrated_prob > threshold)  # (trials x window_bins)
                bound_quantification.update(
                    stats={k: get_fig_stats(v, axis=0) for k, v in bound_quantification.items()})
                bound_quantification.update(bound_values=bound_values,
                                            threshold=threshold)
                prob_choice[bound_name] = bound_quantification  # choice calculating probabilities for#prob_sum has values here half

                if ret_df:  # if returning dataframe compile data accordingly
                    bound_quantification.update(dict(choice=bound_name, trial_index=aligned_data.index.to_numpy()))
                    output_list.append(bound_quantification)

            if 'original' not in num_bins:
                assert len(np.unique(list(num_bins.values()))) == 1, 'Number of bins for different bounds are not equal'

            if ret_df:
                list_df = pd.DataFrame(output_list)[['trial_index', 'prob_sum', 'prob_over_chance', 'bound_values',
                                                     'choice']]
                list_df = (list_df
                           .explode(['prob_sum', 'prob_over_chance', 'trial_index'])
                           .reset_index(drop=True)
                           .set_index('trial_index'))
                output_df = pd.merge(aligned_data[['session_id', 'animal', 'region', 'trial_id', 'update_type',
                                                   'correct', 'turn_type', 'feature_name', 'time_label', 'times',
                                                   'rotational_velocity', 'translational_velocity', 'error','location']],
                                     list_df, left_index=True, right_index=True)
                output_df['choice'] = output_df['choice'].map(choice_mapping)
                return output_df
            else:
                return prob_choice
        else:
            warnings.warn('No aligned data found')
            return None

    def calc_region_interactions(self, param_data, plot_groups):
        # get quantification data
        df_list = []
        group_aligned_data = self.select_group_aligned_data(param_data, plot_groups, ret_df=True)
        quant_df = self.quantify_aligned_data(param_data, group_aligned_data, ret_df=True)
        if quant_df is not None:
            quant_df['times'] = quant_df['times'].apply(tuple)
            quant_df['time_label'] = pd.Categorical(quant_df['time_label'],
                                                    ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made'])

            # pivot data to compare between regions
            index_cols = ['session_id', 'animal', 'update_type', 'trial_id', 'correct', 'time_label', 'choice', 'times']
            region_df = quant_df.pivot(index=index_cols,
                                       columns=['region', 'feature_name'],
                                       values=['prob_sum']).reset_index()
            region_df.columns = ['_'.join(['_'.join(s) if isinstance(s, tuple) else s for s in c])
                                 if c[1] != '' else c[0] for c in region_df.columns.to_flat_index()]
            region_df.sort_values('time_label', inplace=True)

            # loop through brain region/feature to calculate correlations
            name_mapping = dict(y_position='pos', x_position='pos', dynamic_choice='choice')
            for feat1, feat2 in itertools.product(quant_df['feature_name'].unique(), repeat=2):
                if set([f'prob_sum_CA1_{feat1}', f'prob_sum_PFC_{feat2}']).issubset(region_df.columns):
                    corr_df = region_df.apply(lambda x: self.calc_correlation_metrics(x[f'prob_sum_CA1_{feat1}'],
                                                                                      x[f'prob_sum_PFC_{feat2}'],
                                                                                      x['times']), axis=1)
                    corr_df['signal_a'] = f'CA1_{name_mapping[feat1]}'
                    corr_df['signal_b'] = f'PFC_{name_mapping[feat2]}'
                    corr_df['a_vs_b'] = corr_df[['signal_a', 'signal_b']].agg('_'.join, axis=1)
                    df_list.append(pd.concat([region_df[index_cols], corr_df], axis='columns'))  # readd metadata

        if np.size(df_list):
            return pd.concat(df_list, axis='rows')
        else:
            return []

    @staticmethod
    def calc_correlation_metrics(a, b, times, mode='full'):
        # prep a and b vectors (fill in nans, mean subtract, normalize)
        if np.isnan(a).all():
            a = np.empty(np.shape(times))
            a[:] = np.nan

        if np.isnan(b).all():
            b = np.empty(np.shape(times))
            b[:] = np.nan

        a = a - np.nanmean(a)  # mean subtract so average is around 0
        b = b - np.nanmean(b)  # mean subtract so average is around 0

        a = a / np.linalg.norm(a)  # normalize so output values interpretable
        b = b / np.linalg.norm(b)  # normalize so output values interpretable

        corr = signal.correlate(a, b, mode=mode)
        corr_lags = signal.correlation_lags(len(a), len(b), mode=mode) * np.diff(times)[0]

        window_size = int(np.round(1 / np.diff(times)[0]))  # approximately 1s window
        a_windowed = np.lib.stride_tricks.sliding_window_view(a, window_size)
        b_windowed = np.lib.stride_tricks.sliding_window_view(b, window_size)
        times_sliding = np.lib.stride_tricks.sliding_window_view(times, window_size).mean(axis=1)
        corr_sliding = np.stack([signal.correlate(aw, bw, mode=mode) for aw, bw in zip(a_windowed, b_windowed)])
        lags_sliding = signal.correlation_lags(len(a_windowed[0]), len(b_windowed[0]), mode=mode) * np.diff(times)[0]

        # using pearson r will mean subtract the windowed data (better for getting at signal shapes vs. values?)
        corr_coeff = np.nan  # default value
        corr_coeff_sliding = np.empty(np.shape(a_windowed)[0])
        corr_coeff_sliding[:] = np.nan
        if (~np.isnan(a)).all() and (~np.isnan(b)).all():
            corr_coeff_sliding = np.stack([pearsonr(aw, bw)[0] for aw, bw in zip(a_windowed, b_windowed)])
            post_start = np.argwhere(np.array(times) > 0)
            if np.size(post_start):
                corr_coeff = pearsonr(a[post_start[0][0]:], b[post_start[0][0]:])[0]

        return pd.Series([corr, corr_coeff, corr_lags, corr_sliding, corr_coeff_sliding, lags_sliding, times_sliding],
                         index=['corr', 'corr_coeff', 'corr_lags', 'corr_sliding',
                                'corr_coeff_sliding', 'lags_sliding', 'times_sliding'])
        
    def calc_trial_by_trial_quant_data(self, param_data, plot_groups, prob_value='prob_over_chance', n_time_bins=3,
                                       time_window=(0, 2.5), other_zones=None, half=None, prospective_reps=False, hm=False):#put in heatmap visualizer
        # get data for z-scoring (all trial types but only times around update)
        group_aligned_data = self.select_group_aligned_data(param_data, dict(time_label=['t_update']), ret_df=True)
        quant_df = self.quantify_aligned_data(param_data, group_aligned_data, ret_df=True, other_zones=other_zones, prospective_reps=prospective_reps)#, half=half)#half=xx here
        zscore_mean = np.nanmean(np.stack(quant_df[prob_value]))
        zscore_std = np.nanstd(np.stack(quant_df[prob_value]))

        # get data for quantification
        group_aligned_data = self.select_group_aligned_data(param_data, plot_groups, ret_df=True)
        quant_df = self.quantify_aligned_data(param_data, group_aligned_data, ret_df=True, other_zones=other_zones, half=half, prospective_reps=prospective_reps, hm=hm)

        if np.size(quant_df) and quant_df is not None:
            # get diff from baseline
            prob_sum_mat = np.vstack(quant_df[prob_value])
            align_end = np.argwhere(quant_df['times'].values[0] >= 0)[0][0]
            align_start = np.argwhere(quant_df['times'].values[0] >= -1.5)[0][0]
            if align_end != 0:
                # align_end = align_end - 1  # get index immediately preceding 0 if not the first one
                prob_sum_diff = prob_sum_mat.T - np.nanmean(prob_sum_mat[:, align_start:align_end], axis=1)
            elif align_end == 0:
                prob_sum_diff = prob_sum_mat.T - prob_sum_mat[:, align_end]  # only use first bin if that's all there is
            quant_df['diff_baseline'] = list(prob_sum_diff.T)
            quant_df['zscore_prob'] = list((prob_sum_mat - zscore_mean) / zscore_std)

            # get diff from left vs. right bounds
            quant_df['trial_index'] = quant_df.index
            quant_df = (quant_df
                        .explode(['times', 'prob_sum', 'prob_over_chance', 'diff_baseline', 'zscore_prob',
                                  'rotational_velocity', 'translational_velocity', 'error'])
                        .reset_index())
            quant_df['times_binned'] = pd.cut(quant_df['times'], np.linspace(*time_window, n_time_bins)).apply(
                lambda x: x.mid)
            quant_df = pd.DataFrame(quant_df.to_dict())  # fix to avoid object dtype errors in seaborn

            choice_df = quant_df.pivot_table(
                index=['session_id', 'animal', 'time_label', 'times', 'times_binned', 'trial_index'],
                columns=['choice'],
                values=[prob_value, 'diff_baseline', 'zscore_prob']).reset_index()  # had times here before
            choice_df['diff_switch_stay'] = choice_df[(prob_value, 'switch')] - choice_df[(prob_value, 'initial_stay')]
            choice_df['zscore_diff_switch_stay'] = choice_df[('zscore_prob', 'switch')] - \
                                                   choice_df[('zscore_prob', 'initial_stay')]
            choice_df.columns = ['_'.join(c) if c[1] != '' else c[0] for c in choice_df.columns.to_flat_index()]

            return quant_df, choice_df
        else:
            return None, None

    @staticmethod
    def get_tuning_data(param_data, model_name='model'):
        tuning_curve_list = []
        for _, sess_data in param_data.iterrows():
            for key, unit_tuning in sess_data[model_name].items():
                tuning_curve_list.append(dict(animal=sess_data['results_io'].animal,
                                              session=sess_data['results_io'].session_id,
                                              unit=key,
                                              tuning_curve=unit_tuning,
                                              bins=sess_data['bins']))

        tuning_curve_df = pd.DataFrame(tuning_curve_list)

        return tuning_curve_df

    @staticmethod
    def _get_aligned_data(param_data):
        cols_to_keep = ['session_id', 'animal', 'region', 'bins', 'encoder_bin_num', 'decoder_bin_size',
                        'decoder_test_size', 'virtual_track', 'aligned_data']
        temp_data = param_data[cols_to_keep].explode('aligned_data').reset_index(drop=True)
        aligned_data = pd.json_normalize(temp_data['aligned_data'], max_level=0)
        if 'stats' in aligned_data.columns:
            aligned_data = aligned_data.drop(['stats'], axis='columns')
        aligned_data_df = pd.concat([temp_data, aligned_data], axis=1)
        aligned_data_df.drop(['aligned_data'], axis='columns', inplace=True)  # remove the extra aligned data column

        data_to_explode = ['trial_id', 'feature', 'decoding', 'error', 'probability', 'turn_type', 'correct',
                           'theta_phase', 'theta_amplitude', 'rotational_velocity', 'translational_velocity',
                           'choice_commitment', 'view_angle','location']
        exploded_df = aligned_data_df.explode(data_to_explode).reset_index(drop=True)
        exploded_df.dropna(axis='rows', inplace=True)

        return exploded_df

    def _flip_y_position(self, data, bounds, bins=None):
        if bins is not None:
            areas = self.adjust_bound_bins(bins, bounds)

        flipped_data = []
        for ind, row in data.iteritems():
            if bins is not None:
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
        # get decoded values
        if decoder.decoded_values.any().any():
            if isinstance(decoder.decoded_values, (pd.DataFrame, pd.Series)):
                decoded_series = decoder.decoded_values
                decoded_series = pd.DataFrame(decoded_series)
                decoded_series.index = np.round(decoded_series.index.to_numpy(), 4)
                if isinstance(decoded_series, pd.DataFrame):
                    if decoded_series.shape[1] == 1:  # Check if there is only one column
                        decoded_series = decoded_series.iloc[:, 0]  # Convert to Series by selecting the only column
                    else:
                        raise ValueError("decoded_series is a DataFrame with more than one column. Specify which column to use.")
                decoded_series = pd.Series(decoded_series)
                
            else:
                decoded_series = decoder.decoded_values.as_series()
                decoded_series.index = np.round(decoded_series.index.to_numpy(), 4)
        else:
            decoded_series = pd.Series()

        # get actual feature values for each bin
        half_bin = np.round(decoder.decoder_bin_size / 2, 4)
        output_bins = [(np.round(t - half_bin, 4), np.round(t + half_bin, 4)) for t in decoded_series.index]
        time_bins = pd.IntervalIndex.from_tuples(output_bins)
        feat_bins = pd.cut(decoder.features_test.index, time_bins)
        actual_feature = decoder.features_test.iloc[:, 0].groupby(feat_bins).mean()
        actual_series = pd.Series(actual_feature.values, index=np.round(decoded_series.index, 4), name='actual_feature')

        # combine actual and decoded features and calculate error
        df_decode_results = pd.merge(decoded_series.rename('decoded_feature'), actual_series, how='left',
                                     left_index=True, right_index=True)
        df_decode_results['decoding_error'] = abs(
            df_decode_results['decoded_feature'] - df_decode_results['actual_feature'])
        df_decode_results['decoding_error_rolling'] = df_decode_results['decoding_error'].rolling(20,
                                                                                                  min_periods=20).mean()
        if decoder.decoded_probs.any().any():
            if isinstance(decoder.decoded_probs, pd.DataFrame):
                df_decode_results['prob_dist'] = [x for x in decoder.decoded_probs.to_numpy()]
            else:
                df_decode_results['prob_dist'] = [x for x in decoder.decoded_probs.as_dataframe().to_numpy()]
            df_positions = df_decode_results[['actual_feature', 'decoded_feature']].dropna(how='any')
            rmse = sqrt(mean_squared_error(df_positions['actual_feature'], df_positions['decoded_feature']))
        else:
            df_decode_results['prob_dist'] = [x for x in decoder.decoded_probs.to_numpy()]
            rmse = np.nan

        # add trial type info and labels
        time_bins = pd.IntervalIndex.from_tuples(list(zip(decoder.test_df['start_time'], decoder.test_df['stop_time'])))
        trial_labels = {interval: ind for interval, ind in zip(time_bins, decoder.test_df.index)}
        trial_bins = pd.cut(df_decode_results.index, time_bins)
        df_decode_results['trial_id'] = trial_bins.map(trial_labels)
        df_decode_results = df_decode_results.merge(decoder.test_df[['turn_type', 'update_type', 'correct']],
                                                    how='left',
                                                    left_on='trial_id', right_index=True, validate='many_to_one')
        df_decode_results['update_type'] = df_decode_results['update_type'].map({1: 'delay only',
                                                                                 2: 'switch',
                                                                                 3: 'stay'})
        # add summary data
        df_decode_results['session_rmse'] = rmse
        df_decode_results['animal'] = decoder.results_io.animal
        df_decode_results['session'] = decoder.results_io.session_id

        return df_decode_results

    def _meets_exclusion_criteria(self, sess_dict, data):
        exclude_session = False  # default to include session

        # apply exclusion criteria
        units_threshold = self.exclusion_criteria.get('units', 0)
        trials_threshold = self.exclusion_criteria.get('trials', 0)

        if data.subset_reg == True:
            path = self.results_io.get_results_path(results_type='response')
            file_path = path / "unit_counts_per_region.xlsx"
            df = pd.read_excel(file_path)#loading in the excel spreadsheet with all of the unit numbers for each region from each nwb session
            df['file_without_ext'] = df['File'].str.replace('.nwb', '', regex=False)#removing .nwb from each session name
            # Find the row where the 'file' column matches the session_id
            matching_row = df[df['file_without_ext'] == sess_dict['session_id']]
            # Extract the number of units for CA1 and PFC (assuming the columns are named 'CA1_units' and 'PFC_units')
            if not matching_row.empty:
                ca1_units = matching_row['CA1'].values[0]
                pfc_units = matching_row['PFC'].values[0]
                if ca1_units < units_threshold or pfc_units < units_threshold:
                    exclude_session = True
            else:
                exclude_session = True
            if len(data.test_df) < trials_threshold:
                exclude_session = True
        else:
            if len(data.spikes) < units_threshold:
                exclude_session = True
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
        nbins = int(window * 2 / decoder.decoder_bin_size)

        trial_type_dict = dict(non_update=1, switch=2, stay=3)
        output = []
        for trial_name in ['non_update', 'switch', 'stay']:
            trials_to_agg = decoder.test_df[decoder.test_df['update_type'] == trial_type_dict[trial_name]]

            for time_label in self.align_times:  # skip last align times so only until stop of trial
                window_start, window_stop = window, window
                if time_label == 't_choice_made':
                    window_stop = 0  # if choice made, don't grab data past bc could be end of trial
                elif time_label == 'start_time':
                    window_start = 0
                new_times = np.linspace(-window_start, window_stop, num=nbins + 1)

                mid_times = trials_to_agg[time_label]
                turns = trials_to_agg['turn_type'][~mid_times.isna()].values
                outcomes = trials_to_agg['correct'][~mid_times.isna()].values
                trials = trials_to_agg.index[~mid_times.isna()].values
                mid_times = mid_times.dropna()

                phase_col = np.argwhere(decoder.theta.columns == 'phase')[0][0]
                amp_col = np.argwhere(decoder.theta.columns == 'amplitude')[0][0]
                rot_col = np.argwhere(decoder.velocity.columns == 'rotational')[0][0]
                trans_col = np.argwhere(decoder.velocity.columns == 'translational')[0][0]
                view_col = np.argwhere(decoder.commitment.columns == 'view_angle')[0][0]
                choice_col = np.argwhere(decoder.commitment.columns == 'choice_commitment')[0][0]
                location_col = np.argwhere(decoder.location.columns == 'location')[0][0]
                vars = dict(feature=decoder.features_test.iloc[:, 0], decoded=decoder.decoded_values,
                            probability=decoder.decoded_probs, theta_phase=decoder.theta.iloc[:, phase_col],
                            theta_amplitude=decoder.theta.iloc[:, amp_col],
                            rotational_velocity=decoder.velocity.iloc[:, rot_col],
                            translational_velocity=decoder.velocity.iloc[:, trans_col],
                            choice_commitment=decoder.commitment.iloc[:, choice_col],
                            view_angle=decoder.commitment.iloc[:, view_col],
                            location=decoder.location.iloc[:, location_col])
                locs = self._get_start_stop_locs(vars, mid_times, window_start, window_stop)

                interp_dict = dict()
                for name in decoder.feature_names:
                    interp_dict[name] = dict()
                    if decoder.decoded_values.any().any() and mid_times.any():
                        for key, val in vars.items():
                            if decoder.dim_num == 1 and key == 'probability':
                                interp_dict[name][key] = griddata_time_intervals(val, locs[key]['start'],
                                                                                 locs[key]['stop'], nbins, mid_times,
                                                                                 time_bins=new_times)
                            elif decoder.dim_num == 2 and key == 'probability':
                                interp_dict[name][key] = griddata_2d_time_intervals(val, decoder.bins,
                                                                                    decoder.decoding_values.index.values,
                                                                                    locs[key]['start'],
                                                                                    locs[key]['stop'],
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
                vars_to_save = ['feature', 'error', 'probability', 'theta_phase', 'theta_amplitude',
                                'rotational_velocity', 'translational_velocity', 'choice_commitment', 'view_angle', 'location']
                for name in decoder.feature_names:
                    assert np.shape(np.array(interp_dict[name]['feature']))[0] == np.shape(turns)[0]
                    data = dict(feature_name=name,
                                update_type=trial_name,
                                time_label=time_label,
                                trial_id=trials,
                                decoding=np.array(interp_dict[name]['decoded']),
                                **{k: v for k, v in interp_dict[name].items() if k in vars_to_save},
                                turn_type=turns,
                                correct=outcomes,
                                window_start=-window_start,
                                window_stop=window_stop,
                                window=window,
                                nbins=nbins,
                                times=new_times, )  # times are the middle of the bin
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

    def calc_prediction_data(self, group_aligned_df, plot_groups, prob_value, comparison):
        trial_data, _ = self.calc_trial_by_trial_quant_data(group_aligned_df, plot_groups)
        trial_data['pre_or_post'] = pd.cut(trial_data['times'], [-1.5, 0, 1.5], labels=['pre', 'post'])
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'turn_type',
                        'feature_name',
                        'pre_or_post', 'choice', ]
        data_for_predict = (trial_data
                            .groupby(groupby_cols)[[prob_value, 'diff_baseline']]  # group by trial type
                            .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                            .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1))  # fix columns so flattened
                            .dropna()
                            .reset_index()
                            .assign(choice_code=lambda x: x.choice.map({'initial_stay': 'initial',
                                                                        'switch': 'new'}))
                            .pivot(index=groupby_cols[:-1], columns=['choice_code'], values=[f'{prob_value}_mean'])
                            .reset_index()
                            )

        data_for_predict['initial'] = data_for_predict[(f'{prob_value}_mean', 'initial')]
        data_for_predict['new'] = data_for_predict[(f'{prob_value}_mean', 'new')]
        data_for_predict[('initial_vs_new', '')] = data_for_predict['initial'] - data_for_predict['new']
        data_for_predict = (data_for_predict
                            .droplevel(1, axis=1)
                            .drop(f'{prob_value}_mean', axis=1))

        # convert data to original left/right terms and choices made for prediction purposes
        # drew this out for confirmation that it's correct. Initial/new sides have no dependence on the animals
        # decision or update type. Chosen sides depend on whether choice was correct and what update type
        left, right, left_vs_right, choice_made = [], [], [], []
        for i, row in data_for_predict.iterrows():
            if row['turn_type'] == 1:
                right.append(row['new'])
                left.append(row['initial'])
                left_vs_right.append(row['initial_vs_new'])
                if (row['update_type'] == 'switch' and row['correct'] == 1) or (
                        row['update_type'] != 'switch' and row['correct'] == 0):
                    choice_made.append('right')
                else:
                    choice_made.append('left')
            elif row['turn_type'] == 2:
                right.append(row['initial'])
                left.append(row['new'])
                left_vs_right.append(-1 * row['initial_vs_new'])
                if (row['update_type'] == 'switch' and row['correct'] == 1) or (
                        row['update_type'] != 'switch' and row['correct'] == 0):
                    choice_made.append('left')
                else:
                    choice_made.append('right')

        data_for_predict['left'] = left
        data_for_predict['right'] = right
        data_for_predict['left_vs_right'] = left_vs_right
        data_for_predict['choice_made'] = choice_made

        return data_for_predict

    def predict_trial_outcomes(self, group_aligned_df, plot_groups, prob_value, comparison='correct', iterations=100,
                               switch_only=True, results_io=None, tags=''):
        results_io = results_io if results_io else self.results_io  # default if none provided

        # get input and target variable
        input_target_groups = dict(
            choice_made=dict(prediction_groups=['region', 'update_type', 'pre_or_post', 'correct'],
                             input_variables=['left', 'right', 'left_vs_right'],
                             target_variable='choice_made'),
            correct=dict(prediction_groups=['region', 'update_type', 'pre_or_post'],
                         input_variables=['initial', 'new', 'initial_vs_new'],
                         target_variable='correct'), )
        prediction_groups = input_target_groups[comparison]['prediction_groups']
        input_variables = input_target_groups[comparison]['input_variables']
        target_variable = input_target_groups[comparison]['target_variable']
        enc = OrdinalEncoder()

        data_for_prediction = self.calc_prediction_data(group_aligned_df, plot_groups, prob_value, comparison)
        if switch_only:
            data_for_prediction = data_for_prediction.query('update_type == "switch"')

        # preprocess the data - balance classes and scale
        prediction_outputs = []
        # data_for_prediction = data_for_prediction.query('pre_or_post == "post"')
        for name, data in data_for_prediction.groupby(prediction_groups): # print (name), filter data_for prediction, then rerun here
            unbalanced_data = dict()
            for var in input_variables:
                unbalanced_data[var] = data[var].to_numpy()
            unbalanced_data[target_variable] = (
                enc.fit_transform(data[target_variable].to_numpy().reshape(-1, 1))).squeeze()

            # resample incorrect trials to balance classes
            unique_counts, counts = np.unique(unbalanced_data[target_variable], return_counts=True)
            unbalanced_inds = unbalanced_data[target_variable] == unique_counts[np.argmin(counts)]
            balanced_inds = unbalanced_data[target_variable] == unique_counts[np.argmax(counts)]

            target_undersampled = unbalanced_data[target_variable][unbalanced_inds]
            target_oversampled = unbalanced_data[target_variable][balanced_inds]
            inputs_undersampled = np.vstack([unbalanced_data[v][unbalanced_inds] for v in input_variables]).T
            inputs_oversampled = np.vstack([unbalanced_data[v][balanced_inds] for v in input_variables]).T

            inputs_resampled, target_resampled = resample(inputs_undersampled, target_undersampled, replace=True,
                                                          random_state=123, n_samples=np.shape(target_oversampled)[0])
            inputs_balanced = np.vstack((inputs_oversampled, inputs_resampled))
            target_balanced = np.hstack((target_oversampled, target_resampled))

            # fit svm for each input
            inputs = dict(probability_difference=inputs_balanced[:, 2])  # probability=inputs_balanced[:, :1],
            distributions = dict(C=loguniform(1e0, 1e3), gamma=loguniform(1e-4, 1e0))
            for i, t in itertools.product(list(inputs.keys()), ['actual', 'shuffled']):
                n_iter = iterations if t == 'shuffled' else 1  # for shuffled data get distribution of values
                for iter in tqdm(range(n_iter), desc='randomized hyperparameter search'):
                    filename = f'prediction_{"_".join(name[1:])}_{prob_value}_{comparison}_{tags}_{t}_iter{iter}'
                    path = results_io.get_data_filename(filename=filename, format='csv')
                    if path.is_file():
                        print(f'Loading {t} iteration: {iter}')
                        predict_df = pd.read_csv(path)
                    else:
                        print(f'Running {t} iteration: {iter}')
                        # resample shuffled data each iteration if needed, otherwise use real data
                        target = resample(target_balanced) if t == 'shuffled' else target_balanced

                        # use randomized search to get best hyperparameters
                        model = svm.SVC(kernel='rbf')  # will have to tune the hyperparameters
                        clf = RandomizedSearchCV(model, distributions, n_iter=10, cv=LeaveOneOut(), n_jobs=8)
                        input_data = inputs[i] if np.ndim(inputs[i]) > 1 else inputs[i].reshape(-1, 1)
                        search = clf.fit(input_data, target)
                        model_params = search.best_params_

                        # assess model performance with leave-one-out cross validation
                        model = svm.SVC(kernel='rbf', **model_params)
                        model_score = np.nanmean(cross_val_score(model, input_data, target, cv=LeaveOneOut(),
                                                                 scoring='accuracy', n_jobs=12))

                        predict_dict = {k: v for k, v in zip(prediction_groups, name)}
                        predict_dict.update(dict(input=i, target=t, iter=iter, score=model_score, **model_params))

                        # save data if needed and compile for appending
                        predict_df = pd.DataFrame.from_dict(predict_dict)
                        predict_df.to_csv(path, index=False)

                    prediction_outputs.append(predict_df)

        prediction_data = pd.concat(prediction_outputs)

        return prediction_data

    def calc_significant_bins(self, group_aligned_df, plot_groups, prob_value='prob_sum', tags=None):
        trial_data, _ = self.calc_trial_by_trial_quant_data(group_aligned_df, plot_groups,
                                                                    prob_value=prob_value)
        data_for_stats = (trial_data
                          .query(f'times > 0')
                          .groupby(['session_id', 'update_type', 'correct', 'times', 'choice'])
                          .mean()
                          .reset_index()
                          .assign(choice=lambda x: x.choice.map({'initial_stay': 'initial', 'switch': 'new'})))
                          # average by sessions as a control
        df_list = []
        for name, data in data_for_stats.groupby(['update_type', 'correct', 'times']):
            update_type, correct, time_bin = name

            # setup stats - group variables, pairs to compare, and levels of hierarchical data
            pairs = [[['initial']], [['new']]]
            stats = Stats(levels=['animal', 'session_id'], results_io=self.results_io,
                          approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='manuscript')
            stats.run(data, dependent_vars=['diff_baseline'], group_vars=['choice'],
                      pairs=pairs, filename=f'goal_coding_sig_bins_{tags}_{update_type}_{correct}_{time_bin}')
            stats.stats_df['pair'] = stats.stats_df['pair'].apply(lambda x: x[0][0])  # TODO - add to stats function
            stats.stats_df.drop_duplicates(inplace=True)
            stats.stats_df[['update_type', 'correct', 'times']] = name

            df_list.append(stats.stats_df)

        sig_bins_df = pd.concat(df_list)
        sig_bins_df['fdr'], sig_bins_df['pval_corrected'] = fdrcorrection(sig_bins_df['p_val'].to_numpy())

        return sig_bins_df

    @staticmethod
    def get_residuals(trial_data, prob_value='prob_sum', by_session=False):
        residual_data = (trial_data
                         .dropna(subset=[prob_value, 'choice', 'rotational_velocity', 'translational_velocity'], axis=0)
                         .sort_values(by=['choice', 'update_type', 'session_id', 'trial_id']))
        switch_data = residual_data.query('choice == "switch"')
        input_data = pd.DataFrame().assign(switch=switch_data[prob_value].to_numpy(),
                                           initial_stay=residual_data.query('choice == "initial_stay"')[
                                               prob_value].to_numpy(),
                                           rotation=switch_data['rotational_velocity'].to_numpy(),
                                           translation=switch_data['translational_velocity'].to_numpy(),
                                           session_id=switch_data['session_id'].to_numpy())

        # fit GLM and get residuals
        residuals, r_squared, sessions, choices = [], [], [], []
        for choice in residual_data['choice'].unique():
            if by_session:
                for sess, i_data in input_data.groupby('session_id'):
                    y = i_data[choice]  # changed from choice
                    x = i_data[['rotation', 'translation']]
                    x = sm.add_constant(x)
                    poisson_model = sm.GLM(y, x, family=sm.families.Poisson())
                    glm_results = poisson_model.fit()
                    # print(glm_results.summary())
                    r_squared.append(glm_results.pseudo_rsquared())
                    sessions.append(sess)
                    choices.append(choice)
                    residuals.append(glm_results.resid_pearson)
            else:
                y = input_data[choice]  # changed from choice
                x = input_data[['rotation', 'translation']]
                x = sm.add_constant(x)
                poisson_model = sm.GLM(y, x, family=sm.families.Poisson())
                glm_results = poisson_model.fit()
                # print(glm_results.summary())
                residuals.append(glm_results.resid_pearson)

        # add to output data structure
        residual_data['resid'] = pd.concat(residuals).to_numpy()
        if by_session:
            r_squared_vals = pd.DataFrame(dict(session_id=sessions, choice=choices, r_squared=r_squared))
            residual_data = residual_data.merge(r_squared_vals, on=['session_id', 'choice'], validate='many_to_one')

        return residual_data

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
