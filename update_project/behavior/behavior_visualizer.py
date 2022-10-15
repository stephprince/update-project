import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from pathlib import Path

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_color_theme
from update_project.statistics import get_fig_stats
from update_project.virtual_track import UpdateTrack

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')
rcparams = mpl.rcParams


class BehaviorVisualizer:
    def __init__(self, data, session_id=None):
        self.data = data
        self.virtual_track = UpdateTrack()
        self.colors = get_color_theme()

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
            sess_dict.update(event_durations=sess_dict['behavior'].event_durations)

        self.group_df = pd.DataFrame(data)

    def plot(self):
        self.plot_event_durations()
        self.plot_proportion_correct()
        self.plot_trajectories()
        self.plot_aligned_data()

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
        if np.size(rolling_df['prop_correct'].dropna()):
            plot_distributions(rolling_df, axes=axes, column_name='prop_correct', group='update_type', row_ids=[0, 1, 2],
                               col_ids=[0, 0, 0, 0], xlabel=xlabel, title=title, palette=self.colors['trials'])
            plot_distributions(rolling_df, axes=axes, column_name='prop_correct', group='animal', row_ids=[0, 1, 2],
                               col_ids=[1, 1, 1, 1], xlabel=xlabel, title=title)

        # binned proportion correct
        title = 'Performance - binned'
        xlabel = 'proportion correct'
        binned_df = df_by_bin[df_by_bin["type"] == 'binned']
        if np.size(binned_df['prop_correct'].dropna()):
            plot_distributions(binned_df, axes=axes, column_name='prop_correct', group='update_type', row_ids=[0, 1, 2],
                           col_ids=[2, 2, 2, 2], xlabel=xlabel, title=title, palette=self.colors['trials'])
            plot_distributions(binned_df, axes=axes, column_name='prop_correct', group='animal', row_ids=[0, 1, 2],
                           col_ids=[3, 3, 3, 3], xlabel=xlabel, title=title)

        # wrap up and save plot
        for col in range(ncols):
            axes[1][col].set_xlim((0, 1))
            axes[2][col].set_ylim((0, 1))
            axes[1][col].plot([0.5, 0.5], axes[1][col].get_ylim(), linestyle='dashed', color=[0, 0, 0, 0.5])
            axes[2][col].plot(axes[2][col].get_xlim(), [0.5, 0.5], linestyle='dashed', color=[0, 0, 0, 0.5])

        # save figure
        fig.suptitle(f'Behavioral performance - all animals')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'performance', results_type=self.results_type)

    def plot_trajectories(self):
        temp_df = self.group_df[['animal', 'trajectories']].explode('trajectories').reset_index(drop=True)
        trajectory_df = pd.concat([temp_df['animal'], pd.DataFrame(list(temp_df['trajectories']))], axis=1)

        vars_to_plot = ['x_position', 'view_angle', *self.data[0]['behavior'].analog_vars]
        turn_colors = dict(left=self.colors['left'], right=self.colors['right'])
        ncols = 3*2  # 1 column for each update type (non, switch, stay) * indiv + averages
        nrows = len(vars_to_plot)  # 1 col for each var
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 17), sharey='row')

        group_data = trajectory_df.groupby(['update_type', 'turn_type'])
        for name, group in group_data:
            col_id = [ind*2 for ind, k in enumerate(self.virtual_track.mappings['update_type'].values()) if k == name[0]][0]

            for row_id, var in enumerate(vars_to_plot):
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
                axes[row_id][col_id + 1].plot(y_labels, np.array(y_position).T, color=turn_colors[name[1]], alpha=0.1)

                if row_id + 1 == nrows - 1:
                    axes[row_id][col_id].set_xlabel('y_position')
                    axes[row_id][col_id + 1].set_xlabel('y_position')

        axes[0][col_id].legend(loc='upper right')

        # save figure
        fig.suptitle(f'Trajectories')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'trajectories', results_type=self.results_type)

    def plot_aligned_data(self):
        temp_df = self.group_df[['animal', 'aligned_data']].explode('aligned_data').reset_index(drop=True)
        aligned_df = pd.concat([temp_df['animal'], pd.DataFrame(list(temp_df['aligned_data']))], axis=1)
        vars_to_plot = ['x_position', 'view_angle', *self.data[0]['behavior'].analog_vars]

        # plot of all aligned timepoints in the tasks (1 plot for each variable)
        group_data = aligned_df.groupby(['update_type', 'start_label', 'turn_type'])
        for var in vars_to_plot:
            ncols = len(aligned_df['start_label'].unique())  # 1 col for each aligned time
            nrows = 2  # 1 row for each update type (non, switch, stay) indiv + 1 row for averages
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey='row')

            for name, group in group_data:
                var_data = group[group['var'] == var]
                row_id = 0
                col_id = [ind for ind, k in enumerate(aligned_df['start_label'].unique()) if k == name[1]][0]
                cleaned_data = [v for v in var_data['aligned_data'].values if len(np.shape(v)) == 2]

                if np.size(cleaned_data):
                    aligned_data = np.vstack(cleaned_data)
                    times = var_data['aligned_times'].values[0]
                    stats = get_fig_stats(aligned_data, axis=0)

                    # plot averages
                    axes[row_id][col_id].plot(times, stats['mean'], color=self.colors[name[0]], label=f'{name[0]} mean')
                    axes[row_id][col_id].fill_between(times, stats['lower'], stats['upper'], alpha=0.2,
                                                      color=self.colors[name[0]])

                    # plot individual traces
                    axes[row_id + 1][col_id].plot(times, aligned_data.T, color=self.colors[name[0]], alpha=0.1,
                                                  linewidth=0.5)

                    # add dashed lines for time alignment
                    if name[2] == 'right':
                        axes[row_id][col_id].plot([0, 0], [np.min(stats['lower']), np.max(stats['upper'])], color='k',
                                                  linestyle='dashed')
                        axes[row_id + 1][col_id].plot([0, 0], [np.min(aligned_data), np.max(aligned_data)], color='k',
                                                       linestyle='dashed')

                # add plot labels and row/col specific info
                axes[0][col_id].set_title(name[1])
                if row_id + 1 == nrows - 1:
                    axes[row_id + 1][col_id].set_xlabel('Time  (s)')
                if col_id == 0:
                    axes[row_id][col_id].set_ylabel(name[0])
                else:
                    sns.despine(ax=axes[row_id][col_id], left=True)
                    sns.despine(ax=axes[row_id + 1][col_id], left=True)  # need to set time to be the same then

            # save figure
            fig.suptitle(f'Aligned data')
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'aligned_data_{var}', results_type=self.results_type)

    def plot_event_durations(self):
        durations = pd.melt(self.event_durations, var_name='event', value_name='duration', id_vars='update_type',
                            value_vars=['initial_cue', 'delay1', 'update', 'delay2', 'total_trial'])
        durations.dropna(subset='duration', axis=0, inplace=True)
        durations['update_type'] = durations['update_type'].map({1: 'non_update', 2: 'switch', 3: 'stay'})
        summary = durations.groupby(['update_type', 'event'], sort=False).agg(['median', 'std'])
        summary.columns = summary.columns.droplevel()
        summary.reset_index(inplace=True)

        fig = plt.figure(figsize=(10, 5))
        (
            so.Plot(durations, x='event', y='duration')
            .facet(col='update_type')
            .add(so.Dots(marker="o", pointsize=10, fillalpha=1), so.Agg())
            .add(so.Range(), so.Est(errorbar="sd"))
            .share(y=True)
            .theme(rcparams)
            .on(fig)
            .plot()
        )

        # add text of medians + sd of
        new_line = '\n'
        plus_minus = '\u00B1'
        for ax in fig.axes:
            data = summary.query(f'update_type == "{ax.get_title()}"')
            text = [f'{r["event"]}: {r["median"]:.2f} {plus_minus} {r["std"]:.2f} {new_line}' for _, r in data.iterrows()]
            ax.text(0.05, 0.75, ''.join(text), transform=ax.transAxes)

        fig.suptitle('Task event durations')
        self.results_io.save_fig(fig=fig, filename=f'event_durations', results_type=self.results_type)
