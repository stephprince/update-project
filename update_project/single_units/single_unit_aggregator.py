import numpy as np
import pandas as pd

from scipy.stats import sem, zscore
from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate


class SingleUnitAggregator:

    def __init__(self, exclusion_criteria=None):
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
    def _calc_firing_rate(data, start, stop, ret_timestamps=False):
        sigma_in_secs = 0.05
        ntt = 100
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

    def _calc_psth(self, data, start, stop, apply_zscore=True, zscore_mean=None, zscore_std=None):
        firing_rate, tt = self._calc_firing_rate(data, start, stop, ret_timestamps=True)

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