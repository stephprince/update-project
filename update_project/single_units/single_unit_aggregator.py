import numpy as np
import pandas as pd

from scipy.stats import sem, zscore
from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate


class SingleUnitAggregator:

    def __init__(self, exclusion_criteria=dict()):
        self.flip_trials_by_turn = True  # default false
        self.turn_to_flip = 2
        self.exclusion_criteria = exclusion_criteria

    def run_aggregation(self, data):
        # aggregate session data
        for sess_dict in data:
            sess_dict.update(dict(tuning_curves=sess_dict['analyzer'].tuning_curves,
                                  unit_selectivity=sess_dict['analyzer'].unit_selectivity,
                                  aligned_data=sess_dict['analyzer'].aligned_data,
                                  tuning_bins=sess_dict['analyzer'].bins,
                                  trial_info=sess_dict['analyzer'].trials,
                                  excluded_session=self._meets_exclusion_criteria(sess_dict['analyzer'])),)
        self.group_df = pd.DataFrame(data)
        self.group_df = self.group_df[~self.group_df['excluded_session']]  # only keep non-excluded sessions
        self.group_df.drop('analyzer', axis='columns', inplace=True)

        # get aligned dataframe:
        self.group_aligned_data = self._get_aligned_data()
        if self.flip_trials_by_turn:
            self._flip_aligned_trials()
        self.group_tuning_curves = self._get_group_tuning_curves()

        # save memory once information is extracted
        self.group_df.drop(['tuning_curves', 'unit_selectivity', 'aligned_data'], axis=1, inplace=True)

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

    def select_group_aligned_data(self, param_data, filter_dict):
        # filter for specific features
        mask = pd.concat([param_data[k].isin(v) for k, v in filter_dict.items()], axis=1).all(axis=1)
        data_subset = param_data[mask]

        return data_subset

    def _get_group_tuning_curves(self):
        tuning_curve_list = []
        for _, sess_data in self.group_df.iterrows():
            tuning_curves = sess_data['tuning_curves'].transpose()
            tuning_curves.columns = np.round((sess_data['tuning_bins'][1:] + sess_data['tuning_bins'][:-1]) / 2, 2)
            tuning_curves.insert(loc=0, column='unit_id', value=tuning_curves.index)
            tuning_curves.insert(loc=0, column='feature_name', value=sess_data['feature_name'])
            tuning_curves.insert(loc=0, column='session_id', value=sess_data['session_id'])
            tuning_curves.insert(loc=0, column='animal', value=sess_data['animal'])
            tuning_curves = tuning_curves.merge(sess_data['unit_selectivity'], how='left', on='unit_id')

            tuning_curve_list.append(tuning_curves)

        tuning_curve_df = pd.concat(tuning_curve_list, axis=0)

        return tuning_curve_df

    def _get_aligned_data(self):
        aligned_data_list = []
        for _, sess_data in self.group_df.iterrows():
            aligned_data = sess_data['aligned_data']
            event_times = ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made', 'stop_time']
            trial_data = sess_data['trial_info'][event_times]
            aligned_data = pd.merge(aligned_data, trial_data, left_on='trial_ids', right_index=True)
            aligned_data.insert(loc=0, column='feature_name', value=sess_data['feature_name'])
            aligned_data.insert(loc=0, column='session_id', value=sess_data['session_id'])
            aligned_data.insert(loc=0, column='animal', value=sess_data['animal'])
            aligned_data_list.append(aligned_data)
        aligned_data_df = pd.concat(aligned_data_list, axis=0)

        return aligned_data_df

    def get_peak_sorting_index(self):

        cols_to_skip = ['session_id', 'animal', 'feature_name', 'unit_id', 'region', 'cell_type',
                        'mean_selectivity_index', 'max_selectivity_index', 'place_field_threshold']
        tuning_curve_mat = np.stack(self.group_tuning_curves[self.group_tuning_curves.columns.difference(cols_to_skip)]
                                    .to_numpy())
        tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
        sort_index = np.argsort(np.argmax(tuning_curve_scaled, axis=1))

        metadata = (self.group_tuning_curves[['session_id', 'animal', 'feature_name', 'unit_id', 'region']]
                    .reset_index(drop=True))
        sorting_df = pd.concat([metadata, pd.Series(sort_index, name='peak_sort_index')], axis=1)

        return sorting_df

    def get_aligned_psth(self, data):
        # really just grouping by session and unit id but keep rest of data bc same
        cols_to_group = ['animal', 'session_id', 'feature_name', 'region', 'update_type', 'time_label', 'cell_type',
                         'unit_id', 'max_selectivity_type', 'mean_selectivity_type']

        if np.size(data):
            # get mean and std firing rates for all units
            start_time = np.nanmin(np.hstack(data['new_times']))
            stop_time = np.nanmax(np.hstack(data['new_times']))
            zscore_info = (data
                           .groupby(['session_id', 'unit_id'])
                           .apply(lambda x: self._get_mean_std_firing_rate(x['spikes'].to_list(), start_time, stop_time))
                           .reset_index())
            data = data.merge(zscore_info, on=['session_id', 'unit_id'], how='left')

            # calc psth for each aligned time
            psth_data_list = []
            for g_name, g_data in data.groupby('time_label'):
                start, stop = g_data['new_times'].to_numpy()[0][0], g_data['new_times'].to_numpy()[0][-1]
                psth_data = (g_data
                             .groupby(cols_to_group, sort=False, dropna=False)
                             .apply(lambda x: pd.Series({'psth': self._calc_psth(x['spikes'].to_list(), start, stop,
                                                                                 zscore_mean=np.mean(x['mean_fr']),
                                                                                 zscore_std=np.mean(x['std_fr']))}))
                             .reset_index())

                # calc psth for each unit
                grouped_data = (g_data
                                 .groupby(cols_to_group, sort=False, dropna=False)
                                 .agg(spikes=('spikes', lambda x: x.to_list()),
                                      times=('new_times', np.mean),
                                      turn_type=('turn_type', 'mean'),
                                      correct=('correct', 'mean'),
                                      mean_selectivity=('mean_selectivity_index', 'mean'),
                                      max_selectivity=('max_selectivity_index', 'mean'),)
                                 .reset_index())
                psth_data = pd.concat([grouped_data, pd.json_normalize(psth_data['psth'])], axis='columns')
                psth_data_list.append(psth_data)

            return pd.concat(psth_data_list, axis=0)
        else:
            return []

    @staticmethod
    def _calc_firing_rate(data, start, stop, ret_timestamps=False, ntt=None):
        sigma_in_secs = 0.01
        ntt = ntt or 100  # default value
        tt = np.linspace(start, stop, ntt)

        all_data = np.hstack(data)
        if len(all_data):  # if any spikes
            firing_rate = np.array([compute_smoothed_firing_rate(x, tt, sigma_in_secs) for x in data])
        else:
            firing_rate = np.empty((np.shape(data)[0], ntt))
            firing_rate[:] = np.nan

        if ret_timestamps:
            return firing_rate, tt
        else:
            return firing_rate

    def _get_mean_std_firing_rate(self, data, start, stop):
        firing_rate = self._calc_firing_rate(data, start, stop)
        mean = np.mean(firing_rate)
        std = np.std(firing_rate)

        return pd.Series([mean, std], index=['mean_fr', 'std_fr'])

    def _calc_psth(self, data, start, stop, apply_zscore=True, zscore_mean=None, zscore_std=None, ntt=None):
        firing_rate, tt = self._calc_firing_rate(data, start, stop, ret_timestamps=True, ntt=ntt)

        if apply_zscore:
            if zscore_mean and zscore_std:
                out = (firing_rate - zscore_mean) / zscore_std
            else:
                out = zscore(firing_rate, axis=None, nan_policy='omit')
            mean = np.nanmean(out, axis=0)
            err = sem(out, axis=0)
        else:
            mean = np.nanmean(firing_rate, axis=0)
            err = sem(firing_rate, axis=0)  # just filled with nan values

        return dict(psth_mean=mean, psth_err=err, psth_times=tt)

    def _flip_aligned_trials(self):
        # flip the data based on turn type
        flipped_data = []
        for feat, feat_df in self.group_aligned_data.groupby('feature_name', sort=False):
            trials_to_flip = feat_df[feat_df['turn_type'] == self.turn_to_flip]
            if np.size(trials_to_flip):
                cols_to_flip = ['rotational_velocity', 'mean_selectivity_index', 'max_selectivity_index']
                trials_to_flip = trials_to_flip.apply(lambda x: x * -1 if x.name in cols_to_flip else x)
                feat_df.loc[feat_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
            flipped_data.append(feat_df)

        self.group_aligned_data = pd.concat(flipped_data, axis=0)

        # add initial/new goal information
        self.group_aligned_data['max_selectivity_type'] = ['switch' if x < 0 else 'stay' if x > 0 else np.nan
                                                           for x in self.group_aligned_data['max_selectivity_index']]
        self.group_aligned_data['mean_selectivity_type'] = ['switch' if x < 0 else 'stay' if x > 0 else np.nan
                                                            for x in self.group_aligned_data['mean_selectivity_index']]

    def calc_movement_reaction_times(self, param_data, plot_groups):
        group_aligned_data = self.select_group_aligned_data(param_data, plot_groups)
        reaction_df = (group_aligned_data
                       .apply(lambda x: self.get_reaction_time(x['rotational_velocity'],  x['new_times']), axis=1))

        cols = ['session_id', 'animal', 'region', 'trial_ids', 'time_label', 'feature_name', 'new_times',
                'rotational_velocity']
        reaction_df = pd.concat([group_aligned_data[cols], reaction_df], axis='columns')

        output = (reaction_df
                  .groupby(['session_id', 'animal', 'feature_name', 'region', 'time_label', 'trial_ids'])
                  .apply(lambda x: x.iloc[0, :])
                  .reset_index(drop=True))  # should be same for all, so just condensing
        output.dropna(subset='reaction_time', axis=0, inplace=True)

        return output

    def get_reaction_time(self, velocity, times):
        if np.size(velocity):
            # get slope of velocity over time
            veloc_diff = np.diff(velocity, prepend=velocity[0])

            # only calc reaction times post-event for aligned times
            reaction_time = np.max(times)  # default value is max time
            if any(times > 0):
                event_time = np.argwhere(times >= 0)[0][0]
                post_event = veloc_diff[event_time:]

                # calc reaction time
                slope_sign = np.sign(post_event)
                reaction_time_post = np.where(np.diff(slope_sign, prepend=slope_sign[0]))[0]  # get elements immediately after sign switch
                if np.size(reaction_time_post):  # if any slope changes, otherwise keep default
                    reaction_time = times[event_time:][reaction_time_post[0]]

        else:
            veloc_diff = np.empty(np.shape(times))
            veloc_diff[:] = np.nan
            reaction_time = np.nan

        return pd.Series([veloc_diff, reaction_time], index=['veloc_diff', 'reaction_time'])

    def calc_theta_phase_data(self, data, by_trial=False):
        # setup bins
        data.rename(columns={'trial_ids': 'trial_id'}, inplace=True)
        data = data[data['theta_phase'].map(lambda x: len(x) > 0)]

        # break down so each row has a single theta phase/amplitude value
        data['spike_counts'] = data.apply(lambda x: np.histogram(x['spikes'], range=(x['new_times'][0], x['new_times'][-1]),
                                                    bins=len(x['new_times']))[0], axis=1)
        data_to_explode = ['theta_phase', 'theta_amplitude', 'spike_counts', 'new_times', 'timestamps']
        data_to_keep = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'feature_name', 'time_label',
                        'unit_id', 'max_selectivity_type', 'start_time', 't_delay', 't_update', 't_delay2',
                        't_choice_made', *data_to_explode]
        theta_phase_df = data[data_to_keep].explode(data_to_explode).reset_index(drop=True)

        # get histogram, ratio, and mean probability values for different theta phases
        theta_bins = np.linspace(-np.pi, np.pi, 12)
        time_bins =[theta_phase_df['new_times'].min(), 0, theta_phase_df['new_times'].max()]
        data_to_average = ['spike_counts', 'theta_amplitude']
        groups = ['feature_name', 'region', 'max_selectivity_type', 'time_label']
        if by_trial:
            groups.extend(['session_id', 'trial_id'])
        theta_phase_list = []
        for t_name, t_data in theta_phase_df.groupby(groups, dropna=False):
            # time_bins = t_data.loc[:, ['start_time', 't_update', 't_choice_made']].values[0]

            theta_df_bins = pd.cut(t_data['theta_phase'], theta_bins)
            time_df_bins = pd.cut(t_data['new_times'], time_bins,
                                  labels=['pre-update', 'post-update'], right=False)

            mean = t_data[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(lambda x: np.mean(x))
            err = t_data[data_to_average].groupby([time_df_bins, theta_df_bins]).apply(
                lambda x: pd.Series(sem(x.astype(float), nan_policy='omit')))
            err.columns = [f'{d}_err' for d in data_to_average]
            t_df = mean.join(err).reset_index()
            t_df['phase_mid'] = t_df['theta_phase'].astype('interval').apply(lambda x: x.mid)
            for c in ['session_id', 'animal', 'region', 'trial_id', 'time_label', 'update_type', 'feature_name',
                      'max_selectivity_type']:
                t_df[c] = t_data[c].values[0]

            theta_phase_list.append(t_df)
        theta_phase_output = pd.concat(theta_phase_list, axis=0, ignore_index=True)

        return theta_phase_output