import itertools
import numpy as np
import pickle
import pandas as pd
import warnings

from math import sqrt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from scipy.stats import sem, pearsonr
from scipy import signal

from update_project.decoding.interpolate import interp1d_time_intervals, griddata_2d_time_intervals, \
    griddata_time_intervals
from update_project.results_io import ResultsIO
from update_project.statistics import get_fig_stats, get_comparative_stats


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
        self.group_tuning_curves = self._get_group_tuning_curves()
        self.group_aligned_data = self._get_aligned_data()
        if self.flip_trials_by_turn:
            self._flip_aligned_trials()

        # save memory once information is extracted
        self.group_df.drop(['tuning_curves', 'unit_selectivity', 'aligned_data'], axis=1, inplace=True)

    def select_group_aligned_data(self, param_data, filter_dict):
        # filter for specific features
        mask = pd.concat([param_data[k].isin(v) for k, v in filter_dict.items()], axis=1).all(axis=1)
        data_subset = param_data[mask]  #  TODO do I need this function?

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

    def _flip_aligned_trials(self):
        flipped_data = []
        for feat, feat_df in self.group_aligned_data.groupby('feature_name'):
            trials_to_flip = feat_df[feat_df['turn_type'] == self.turn_to_flip]
            if np.size(trials_to_flip):
                cols_to_flip = ['rotational_velocity', 'selectivity_index']
                trials_to_flip = trials_to_flip.apply(lambda x: x * -1 if x.name in cols_to_flip else x)
                feat_df.loc[feat_df['turn_type'] == self.turn_to_flip, :] = trials_to_flip
            flipped_data.append(feat_df)

        self.group_aligned_df = pd.concat(flipped_data, axis=0)
        self.group_aligned_df.sort_index(inplace=True)
