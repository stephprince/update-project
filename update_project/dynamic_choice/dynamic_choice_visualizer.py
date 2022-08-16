import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from update_project.results_io import ResultsIO


class DynamicChoiceVisualizer:

    def __init__(self, data, session_id=None):
        self.data = data

        if session_id:
            self.results_type = 'session'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem, session_id=session_id)
        else:
            self.results_type = 'group'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

        # get session visualization info
        for sess_dict in self.data:
            sess_dict.update(proportion_correct=sess_dict['behavior'].proportion_correct)
            sess_dict.update(trajectories=sess_dict['behavior'].trajectories)
            sess_dict.update(aligned_data=sess_dict['behavior'].aligned_data)

        self.group_df = pd.DataFrame(data)

    def plot(self):
        self.plot_dynamic_choice_by_position()
        self.plot_grid_search_results()

    def plot_dynamic_choice_by_position(self):
        # plot the results for the session
        fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False)
        y_pos_left = self.agg_data['y_position'].T[:, self.agg_data['target_data'] == 1]
        y_pos_right = self.agg_data['y_position'].T[:, self.agg_data['target_data'] == 0]
        predict_left = self.agg_data['predict_data'].T[:, self.agg_data['target_data'] == 1]
        predict_right = self.agg_data['predict_data'].T[:, self.agg_data['target_data'] == 0]
        axes[0][0].plot(np.nanmean(y_pos_left, axis=1), np.nanmean(predict_left, axis=1), color='b',
                        label='left choice')  # TODO - should bin by position instead of doing weird averaging
        axes[0][0].plot(np.nanmean(y_pos_right, axis=1), np.nanmean(predict_right, axis=1), color='r',
                        label='right choice')
        axes[0][0].set(xlabel='position in track', ylabel='p(left)', ylim=[0, 1], title='LSTM prediction - test trials')
        axes[1][0].plot(y_pos_left, predict_left, color='b')
        axes[1][0].plot(y_pos_right, predict_right, color='r')
        axes[1][0].set(xlabel='position in track', ylabel='p(left)', ylim=[0, 1], title='LSTM prediction - test trials')

        axes[0][1].plot(np.nanmean(y_pos_left, axis=1), np.nanmean(log_likelihood.T[:, target_data == 1], axis=1),
                        color='b')
        axes[0][1].plot(np.nanmean(y_pos_right, axis=1), np.nanmean(log_likelihood.T[:, target_data == 0], axis=1),
                        color='r')
        axes[0][1].set(xlabel='position in track', ylabel='log_likelihood', ylim=[-3, 0],
                       title='Log likelihood (0 = perfect)')

        axes[1][1].plot(y_pos_left, self.agg_data['log_likelihood'].T[:, self.agg_data['target_data'] == 1], color='b')
        axes[1][1].plot(y_pos_right, self.agg_data['log_likelihood'].T[:, self.agg_data['target_data'] == 0], color='r')
        axes[1][1].set(xlabel='position in track', ylabel='log_likelihood', ylim=[-3, 0],
                       title='Log likelihood (0 = perfect)')
        axes[1][1].axhline(-1, linestyle='dashed', color='k')
        self.results_io.save_fig(fig=fig, axes=axes, filename='decoding performance', results_type='session')