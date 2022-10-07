import numpy as np
import pandas as pd

from scipy.stats import sem
from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate


class SingleUnitAggregator:

    def __init__(self, exclusion_criteria=None):
        self.flip_trials_by_turn = True  # default false
        self.turn_to_flip = 2

    def run_aggregation(self, data, overwrite=False, window=5):
        # aggregate session data
        for sess_dict in data:
            sess_dict.update(dict(tuning_curves=sess_dict['analyzer'].tuning_curves,
                                  unit_selectivity=sess_dict['analyzer'].unit_selectivity,
                                  aligned_data=sess_dict['analyzer'].aligned_data,
                                  tuning_bins=sess_dict['analyzer'].bins))

        self.group_df = pd.DataFrame(data)
        self.group_df.drop('analyzer', axis='columns', inplace=True)

        # get aligned dataframe:
        self.group_aligned_data = self._get_aligned_data()
        if self.flip_trials_by_turn:
            self._flip_aligned_trials()
        self.group_tuning_curves = self._get_group_tuning_curves()

        # save memory once information is extracted
        self.group_df.drop(['tuning_curves', 'unit_selectivity', 'aligned_data'], axis=1, inplace=True)

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
            aligned_data.insert(loc=0, column='feature_name', value=sess_data['feature_name'])
            aligned_data.insert(loc=0, column='session_id', value=sess_data['session_id'])
            aligned_data.insert(loc=0, column='animal', value=sess_data['animal'])
            aligned_data_list.append(aligned_data)
        aligned_data_df = pd.concat(aligned_data_list, axis=0)

        return aligned_data_df

    def get_aligned_psth(self, data):
        # really just grouping by session and unit id but keep rest of data bc same
        cols_to_group = ['animal', 'session_id', 'feature_name', 'region', 'update_type', 'time_label', 'cell_type',
                         'unit_id', 'max_selectivity_type', 'mean_selectivity_type']

        psth_data_list = []
        for g_name, g_data in data.groupby('time_label'):
            start, stop = g_data['new_times'].to_numpy()[0][0], g_data['new_times'].to_numpy()[0][-1]

            # calc psth for each unit
            psth_data = (g_data
                         .groupby(cols_to_group, sort=False, dropna=False)
                         .agg(psth=('spikes', lambda x: self._calc_psth(x.to_list(), start, stop)),
                              times=('new_times', np.mean),
                              turn_type=('turn_type', 'mean'),
                              outcomes=('outcomes', 'mean'),
                              mean_selectivity=('mean_selectivity_index', 'mean'),
                              max_selectivity=('max_selectivity_index', 'mean'),)
                         .reset_index())
            psth_data = pd.concat([psth_data, pd.json_normalize(psth_data['psth'])], axis='columns')
            psth_data.drop('psth', axis='columns', inplace=True)
            psth_data_list.append(psth_data)

        return pd.concat(psth_data_list, axis=0)

    @staticmethod
    def _calc_psth(data, start, stop):
        sigma_in_secs = 0.05
        ntt = 500
        tt = np.linspace(start, stop, ntt)

        all_data = np.hstack(data)
        if len(all_data):  # if any spikes
            smoothed = np.array([compute_smoothed_firing_rate(x, tt, sigma_in_secs) for x in data])

            mean = np.mean(smoothed, axis=0)
            err = sem(smoothed, axis=0)
        else:
            mean, err = np.empty(np.shape(tt)),  np.empty(np.shape(tt))
            mean[:] = np.nan
            err[:] = np.nan

        return dict(psth_mean=mean, psth_err=err, psth_times=tt)

    def _flip_aligned_trials(self):
        # flip the data based on turn type
        flipped_data = []
        for feat, feat_df in self.group_aligned_data.groupby('feature_name'):
            trials_to_flip = feat_df[feat_df['turn_type'] == self.turn_to_flip]
            if np.size(trials_to_flip):
                cols_to_flip = ['rotational_velocity', 'mean_selectivity_index', 'max_selectivity_index']
                trials_to_flip = trials_to_flip.apply(lambda x: x * -1 if x.name in cols_to_flip else x)
                feat_df.loc[feat_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
            flipped_data.append(feat_df)

        self.group_aligned_data = pd.concat(flipped_data, axis=0)
        self.group_aligned_data.sort_index(inplace=True)

        # add initial/new goal information  TODO - check this labelling is correct
        self.group_aligned_data['max_selectivity_type'] = ['stay' if x < 0 else 'switch' if x > 0 else np.nan
                                                           for x in self.group_aligned_data['max_selectivity_index']]
        self.group_aligned_data['mean_selectivity_type'] = ['stay' if x < 0 else 'switch' if x > 0 else np.nan
                                                            for x in self.group_aligned_data['mean_selectivity_index']]