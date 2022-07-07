import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from pathlib import Path

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions
from update_project.statistics import get_fig_stats
from update_project.virtual_track import UpdateTrack


class BehaviorVisualizer:
    def __init__(self, data):
        self.data = data
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)
        self.virtual_track = UpdateTrack()

        # get session visualization info
        for sess_dict in self.data:
            sess_dict.update(proportion_correct=sess_dict['behavior'].proportion_correct)
            sess_dict.update(trajectories=sess_dict['behavior'].trajectories)

            #sess_dict.update(example_period=self._get_example_period(sess_dict['behavior']))

        self.group_df = pd.DataFrame(data)

    def plot(self):
        self.plot_proportion_correct()
        self.plot_trajectories()
        self.plot_data_around_update()

    def _get_example_period(self, behavioral_data):
        test = 1

    def plot_proportion_correct(self):
        # explode df so each prop correct value has one row (duplicates for each session/animal
        temp_df = self.group_df[['animal', 'proportion_correct']].explode('proportion_correct').reset_index(drop=True)
        df_by_update_type = pd.concat([temp_df['animal'], pd.DataFrame(list(temp_df['proportion_correct']))], axis=1)
        df_by_bin = df_by_update_type.explode('prop_correct').reset_index(drop=True)
        df_by_bin = pd.DataFrame(df_by_bin.to_dict())  # fix bc data gets saved as object for some weird reason

        nrows = 3  # 1 row for each plot type (cum fract, hist, violin)
        ncols = 4  # 1 column for binned, 1 for rolling
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

        # rolling proportion correct
        title = 'Performance - rolling'
        xlabel = 'proportion correct'
        rolling_df = df_by_bin[df_by_bin["type"] == 'rolling']
        plot_distributions(rolling_df, axes=axes, column_name='prop_correct', group='update_type', row_ids=[0, 1, 2],
                           col_ids=[0, 0, 0, 0], xlabel=xlabel, title=title)
        plot_distributions(rolling_df, axes=axes, column_name='prop_correct', group='animal', row_ids=[0, 1, 2],
                           col_ids=[1, 1, 1, 1], xlabel=xlabel, title=title)

        # binned proportion correct
        title = 'Performance - binned'
        xlabel = 'proportion correct'
        binned_df = df_by_bin[df_by_bin["type"] == 'binned']
        plot_distributions(binned_df, axes=axes, column_name='prop_correct', group='update_type', row_ids=[0, 1, 2],
                           col_ids=[2, 2, 2, 2], xlabel=xlabel, title=title)
        plot_distributions(binned_df, axes=axes, column_name='prop_correct', group='animal', row_ids=[0, 1, 2],
                           col_ids=[3, 3, 3, 3], xlabel=xlabel, title=title)

        # wrap up and save plot
        for col in range(ncols):
            axes[0][col].set_xlim((0,1))
            axes[1][col].set_xlim((0,1))
            axes[2][col].set_ylim((0,1))
        fig.suptitle(f'Behavioral performance - all animals', fontsize=14)
        plt.tight_layout()
        kwargs = self.results_io.get_figure_args(filename=f'performance', format='pdf')
        plt.savefig(**kwargs)
        plt.close()

    def plot_trajectories(self):
        temp_df = self.group_df[['animal', 'trajectories']].explode('trajectories').reset_index(drop=True)
        trajectory_df = pd.concat([temp_df['animal'], pd.DataFrame(list(temp_df['trajectories']))], axis=1)

        vars_to_plot = ['x_position', 'view_angle', *self.data[0]['behavior'].analog_vars]
        turn_colors = dict(left='b', right='r')
        ncols = len(vars_to_plot)  # 1 col for each var
        nrows = 3*2  # 1 column for each update type (non, switch, stay) * indiv + averages
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

        group_data = trajectory_df.groupby(['update_type', 'turn_type'])
        for name, group in group_data:
            row_id = [ind*2 for ind, k in enumerate(self.virtual_track.mappings['update_type'].values()) if k == name[0]][0]

            for col_id, var in enumerate(vars_to_plot):
                y_position = pd.DataFrame(list(group[var]))  # TODO - check bins match for all animals
                y_labels = y_position.columns.mid.values
                stats = get_fig_stats(np.array(y_position), axis=0)

                # plot averages
                axes[row_id][col_id].plot(y_labels, stats['mean'], color=turn_colors[name[1]], label=f'{name[1]} mean')
                axes[row_id][col_id].fill_between(y_labels, stats['lower'], stats['upper'], alpha=0.2,
                                                color=turn_colors[name[1]], label=f'{name[1]} 95% CI')
                axes[row_id][col_id].set_ylabel(name[0])
                axes[row_id][col_id].set_title(var, fontsize=14)

                # plot individual traces
                axes[row_id + 1][col_id].plot(y_labels, np.array(y_position).T, color=turn_colors[name[1]], alpha=0.1)

                if row_id + 1 == nrows - 1:
                    axes[row_id + 1][col_id].set_xlabel('y_position')

        axes[0][col_id].legend(loc='upper right')

        # save figure
        fig.suptitle(f'Trajectories', fontsize=14)
        plt.tight_layout()
        kwargs = self.results_io.get_figure_args(filename=f'trajectories', format='pdf')
        plt.savefig(**kwargs)
        plt.close()

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