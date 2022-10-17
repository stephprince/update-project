import numpy as np
import pandas as pd

from update_project.decoding.bayesian_decoder_aggregator import BayesianDecoderAggregator
from update_project.single_units.single_unit_aggregator import SingleUnitAggregator


class ExampleTrialAggregator:

    def __init__(self):
        self.turn_to_flip = 2
        self.plot_groups = dict(update_type=['non_update', 'switch', 'stay'],
                                turn_type=[1, 2],
                                correct=[0, 1])

    def run_aggregation(self, data, exclusion_criteria, align_window=5, align_times=['t_update']):
        # get single unit aggregated data
        self.single_unit_agg = SingleUnitAggregator(exclusion_criteria=exclusion_criteria)
        self.single_unit_agg.run_aggregation(data)
        spikes = self.single_unit_agg.select_group_aligned_data(self.single_unit_agg.group_aligned_data,
                                                                self.plot_groups)
        spikes.rename(columns={'trial_ids': 'trial_id'}, inplace=True)

        # get bayesian decoder aggregated data
        # note less trials in the decoder output bc some used for training the model and not included as output
        self.decoder_agg = BayesianDecoderAggregator(exclusion_criteria=exclusion_criteria, align_times=align_times)
        self.decoder_agg.run_df_aggregation(data, window=align_window)
        decoding = self.decoder_agg.select_group_aligned_data(self.decoder_agg.group_aligned_df, self.plot_groups,
                                                                ret_df=True)
        decoding['region'] = decoding['region'].apply(lambda x: x[0])  # pull region from list
        decoding_goal = self.decoder_agg.quantify_aligned_data(self.decoder_agg.group_aligned_df, decoding, ret_df=True)

        cols_merge_on = ['session_id', 'animal', 'feature_name', 'region', 'trial_id']
        decoding.drop(labels=decoding.columns.difference(['feature', 'decoding', 'probability', 'error', 'bins',
                                                          'turn_type', *cols_merge_on]).to_list(),
                      axis='columns',
                      inplace=True)
        decoding = decoding_goal.merge(decoding, on=cols_merge_on, how='left', validate='many_to_one')
        decoding_goal = []  # clear memory

        # combine data streams
        agg_data = decoding.merge(spikes, suffixes=('', '_r'),
                                  on=['animal', 'session_id', 'feature_name', 'region', 'trial_id', 'time_label'])
        assert not (agg_data['turn_type'] - agg_data['turn_type_r']).to_numpy().any()  # check trials aligned
        agg_data.drop(labels=list(agg_data.filter(regex='_r')), axis='columns', inplace=True)
        agg_data['trial_id'] = agg_data['trial_id'].astype(str)

        self.agg_data = agg_data

    def select_group_aligned_data(self, data=None, filter_dict=None):
        data = data or self.agg_data  # default to all data if none indicated

        # filter for specific features
        mask = pd.concat([data[k].isin(v) for k, v in filter_dict.items()], axis=1).all(axis=1)
        data_subset = data[mask]

        return data_subset
