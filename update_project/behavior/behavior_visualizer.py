import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from pathlib import Path

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions
from update_project.statistics import get_fig_stats


class BehaviorVisualizer:
    def __init__(self, data):
        self.data = data
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

        # get session visualization info
        for sess_dict in self.data:
            sess_dict.update(example_period=self._get_example_period(sess_dict['behavior']))

        self.group_df = pd.DataFrame(data)

    def plot(self):
        self.plot_proportion_correct()
        self.plot_trajectories()
        self.plot_data_around_update()

    def _get_example_period(self):
        test = 1

    def plot_proportion_correct(self):
        nrows = 3  # 1 row for each plot type (cum fract, hist, violin)
        ncols = 2  # 1 column for binned, 1 for rolling
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

        # rolling proportion correct
        title = 'Performance - rolling'
        xlabel = 'proportion correct'
        plot_distributions(data, axes=axes, column_name='prop_correct_rolling', group='animal', row_ids=[0, 1, 2],
                           col_ids=[0, 0], xlabel=xlabel, title=title)

        # binned proportion correct
        title = 'Performance - binned'
        xlabel = 'proportion correct'
        plot_distributions(data, axes=axes, column_name='prop_correct_rolling', group='animal', row_ids=[0, 1, 2],
                           col_ids=[1, 1], xlabel=xlabel, title=title)

        update_types = []
        trials_by_update_type = []
        for key, group in df_trials.groupby('update_type'):
            update_types.append(key)
            trials_by_update_type.append(group['prop_correct'])

        fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
        violin_parts = ax_0.violinplot(trials_by_update_type, update_types, showmeans=True)

        # plot horizontal dashed line across the middle
        horizontal_dashed_line_x = np.linspace(start=0.5, stop=3.5, num=100)
        horizontal_dashed_line_y = np.full(shape=(100,), fill_value=0.5)
        ax_0.plot(horizontal_dashed_line_x, horizontal_dashed_line_y, color='black', linestyle='dashed', alpha=0.5)

        # double check if the numbers are mapped to the right labels
        update_types_dict = {1.0: "Update",
                             2.0: "Non-update",
                             3.0: "Visual-guided"}
        ax_0.set_title("Behavioral Performance Violin Plot")
        ax_0.set_xlabel("Trial Type")
        ax_0.set_ylabel("Proportion Correct")
        ax_0.set_xticks([1.0, 2.0, 3.0])
        ax_0.set_xticklabels([update_types_dict[1.0], update_types_dict[2.0], update_types_dict[3.0]])
        ax_0.set_xlim([0.5, 3.5])
        ax_0.set_ylim([0.0, 1.0])

        # customize color of violins
        colors = ['red', 'green', 'blue']
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_color(colors[i])
        violin_parts['cmeans'].set_color(colors)
        violin_parts['cbars'].set_color(colors)
        violin_parts['cmaxes'].set_color(colors)
        violin_parts['cmins'].set_color(colors)

    def plot_trajectories(self):
        test = 1

    def plot_data_around_update(self):
        # TODO - this code is from Grace's project, need to adapt
        # fig = plt.figure(constrained_layout=True, figsize=(20, 10))
        # ax = fig.add_subplot(111)
        fig_1, ax_1 = plt.subplots(nrows=1, ncols=1)
        for idx, trace in enumerate(view_angle_around_update):
            print("idx = {}, len = {}".format(idx, len(trace)))  # the length of the traces are not uniform - ERROR
            time_from_aligned = time_around_update[idx] - time_around_update[idx][0] - 5
            if trace[-1] > 0.0:
                ax_1.plot(time_from_aligned, trace, color='red', alpha=0.4)  # alpha controls transparency, 1.0 = opaque
            else:
                ax_1.plot(time_from_aligned, trace, color='blue', alpha=0.4)

        # plot vertical dashed line down the middle
        vertical_dashed_line_x = np.zeros(100)
        vertical_dashed_line_y = np.linspace(start=-0.8, stop=0.8, num=100)
        ax_1.plot(vertical_dashed_line_x, vertical_dashed_line_y, color='black', linestyle='dashed', alpha=0.5)

        ax_1.set_title("View Angle Line Plot")
        ax_1.set_xlabel("Position Around Update")  # units? is this time? where does update happen?
        ax_1.set_ylabel("View Angle (rad)")
        ax_1.set_xlim([-5.0, 5.0])
        ax_1.set_ylim([-0.8, 0.8])

        plt.show()

    def plot_example_period(self):
        test = 1