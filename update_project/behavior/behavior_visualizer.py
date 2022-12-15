import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from pathlib import Path
from scipy.stats import sem

from update_project.general.results_io import ResultsIO
from update_project.general.plots import plot_distributions, rainbow_text, clean_box_plot, add_task_phase_lines
from update_project.statistics.statistics import get_fig_stats
from update_project.base_visualization_class import BaseVisualizationClass

rng = np.random.default_rng(12345)


class BehaviorVisualizer(BaseVisualizationClass):
    def __init__(self, data, session_id=None):
        super().__init__(data)

        if session_id:
            self.results_type = 'session'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path(__file__).parent.stem,
                                        session_id=session_id)
        else:
            self.results_type = 'group'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path(__file__).parent.stem)

        # get session visualization info
        for sess_dict in self.data:
            sess_dict.update(proportion_correct=sess_dict['analyzer'].proportion_correct)
            sess_dict.update(trajectories=sess_dict['analyzer'].trajectories)
            sess_dict.update(aligned_data=sess_dict['analyzer'].aligned_data)
            sess_dict.update(event_durations=sess_dict['analyzer'].event_durations)

        self.group_df = pd.DataFrame(data)

    def plot(self):
        self.plot_event_durations()
        self.plot_proportion_correct()
        self.plot_trajectories()
        self.plot_aligned_data()

    def plot_performance(self, sfig):
        """plot performance for manuscript figure"""
        temp_df = (self.group_df[['animal', 'session_id', 'proportion_correct']]
                   .explode('proportion_correct')
                   .reset_index(drop=True))

        df_by_update_type = pd.concat([temp_df[['animal', 'session_id']],
                                       pd.DataFrame(list(temp_df['proportion_correct']))], axis=1)
        df_by_bin = pd.DataFrame(df_by_update_type
                                 .explode('prop_correct')
                                 .query('type == "binned"')
                                 .assign(update_type=lambda x: x.update_type.map({'non_update': 'delay only',
                                                                                  'switch_update': 'switch',
                                                                                  'stay_update': 'stay'}))
                                 .rename(columns={'prop_correct': 'proportion correct', 'update_type': 'trial type'})
                                 .reset_index(drop=True)
                                 .to_dict())  # convert to dict and back bc violin plot has object format err otherwise

        ax = sfig.subplots(nrows=1, ncols=1)
        sns.boxplot(data=df_by_bin, x='trial type', y='proportion correct', ax=ax, width=0.5,
                    palette=self.colors['trials'], medianprops={'color': 'white'}, showfliers=False)
        clean_box_plot(ax)

        ax.axhline(0.5, linestyle='dashed', color=self.colors['nan'])
        ax.set(title='Task performance', xlabel=None, ylim=(0, 1))

        return sfig

    def plot_performance_by_animals(self, sfig):
        temp_df = (self.group_df[['animal', 'session_id', 'proportion_correct']]
                   .explode('proportion_correct')
                   .reset_index(drop=True))

        df_by_update_type = pd.concat([temp_df[['animal', 'session_id']],
                                       pd.DataFrame(list(temp_df['proportion_correct']))], axis=1)
        df_by_bin = pd.DataFrame(df_by_update_type
                                 .explode('prop_correct')
                                 .query('type == "binned"')
                                 .assign(update_type=lambda x: x.update_type.map({'non_update': 'delay only',
                                                                                  'switch_update': 'switch',
                                                                                  'stay_update': 'stay'}))
                                 .rename(columns={'prop_correct': 'proportion correct', 'update_type': 'trial type'})
                                 .reset_index(drop=True)
                                 .to_dict())  # convert to dict and back bc violin plot has object format err otherwise

        ax = sfig.subplots(nrows=1, ncols=1)
        sns.boxplot(data=df_by_bin, x='trial type', y='proportion correct', hue='animal', ax=ax, width=0.5,
                    palette=self.colors['animals'], medianprops={'color': 'white'}, showfliers=False)
        clean_box_plot(ax, labelcolors=[self.colors[c] for c in df_by_bin['trial type'].unique()])
        ax.get_legend().remove()
        rainbow_text(0.85, 0.85, df_by_bin['animal'].unique(), self.colors['animals'], ax=ax, size=10)

        ax.axhline(0.5, linestyle='dashed', color=self.colors['nan'])
        ax.set(title='Task performance per animal', xlabel=None, ylim=(0, 1))

        return sfig

    def plot_trajectories_by_position(self, sfig, var='view_angle', num_trials=20):
        """plot trajectories for manuscript figure"""

        # get data for example trial
        example_animal, example_session = (25, 'S25_210913')  # TODO - find good representative session
        temp_df = (self.group_df[['animal', 'session_id', 'trajectories']]
                   .query(f'animal == {example_animal} & session_id == "{example_session}"')
                   .explode('trajectories')
                   .reset_index(drop=True))
        trajectory_df = pd.concat([temp_df[['animal', 'session_id']],
                                   pd.DataFrame(list(temp_df['trajectories']))], axis=1)
        trajectory_df['update_type'] = trajectory_df['update_type'].map({'non_update': 'delay only',
                                                                         'switch_update': 'switch',
                                                                         'stay_update': 'stay'})
        trajectory_df = trajectory_df.query('correct == 1')

        ax = sfig.subplots(nrows=1, ncols=3, sharey=True)
        for (update, turn), group in trajectory_df.groupby(['update_type', 'turn_type']):
            ax_id = np.argwhere(trajectory_df['update_type'].unique() == update)[0][0]
            linestyle = 'dotted' if turn == 'right' else 'solid'

            # transform y data to fraction of track
            y_position = pd.DataFrame(list(group[var]))
            y_labels = y_position.columns.mid.values
            track_fraction = (y_labels - np.min(y_labels)) / np.max(y_labels - np.min(y_labels))
            cue_locations = {k: np.round((v - np.min(y_labels)) / np.max(y_labels - np.min(y_labels)), 4)
                             for k, v in self.virtual_track.cue_start_locations['y_position'].items()}
            if turn == 'right':
                ax[ax_id] = add_task_phase_lines(ax[ax_id], cue_locations=cue_locations, text_brackets=True)

            # plot data
            random_samples = rng.choice(range(np.shape(y_position)[0]), num_trials)  # TODO - only full delay non-update
            trajectory = np.rad2deg(np.array(y_position).T[:, random_samples])
            clipping_mask = mpl.patches.Rectangle(xy=(cue_locations['delay cue'], 0),
                                                  width=cue_locations['delay2 cue'] - cue_locations['delay cue'],
                                                  height=1, facecolor='white', alpha=0,
                                                  transform=ax[ax_id].transAxes)
            ax[ax_id].add_patch(clipping_mask)
            ax[ax_id].plot(track_fraction, trajectory, color=self.colors['nan'], alpha=0.2, linestyle=linestyle, label=turn)
            lines = ax[ax_id].plot(track_fraction, trajectory, color=self.colors['trials'][ax_id], alpha=0.25,
                                        linestyle=linestyle, clip_path=clipping_mask)
            for line in lines:
                line.set_clip_path(clipping_mask)
            ax[ax_id].set_title(update, color=self.colors['trials'][ax_id])
            ax[ax_id].set(ylim=(-61, 61))
        handles, labels = ax[0].get_legend_handles_labels()
        which_handles = [np.where(np.array(labels) == t)[0][0] for t in trajectory_df['turn_type'].unique()]
        handles, labels = np.array([[handles[i], labels[i]] for i in which_handles]).T
        [h.set_alpha(1) for h in handles]
        ax[0].legend(list(handles), list(labels), loc='lower left')

        sfig.suptitle('Example trial trajectories', fontsize=12)
        sfig.supxlabel('fraction of track', fontsize=10)
        sfig.axes[0].set(ylabel='view angle (degrees)')

        return sfig

    def plot_trajectories_all_metrics(self, sfig, vars=['x_position', 'rotational_velocity', 'translational_velocity']):
        # get data for example trial
        temp_df = (self.group_df[['animal', 'session_id', 'trajectories']]
                   .explode('trajectories')
                   .reset_index(drop=True))
        trajectory_df = pd.concat([temp_df[['animal', 'session_id']],
                                   pd.DataFrame(list(temp_df['trajectories']))], axis=1)
        trajectory_df['update_type'] = trajectory_df['update_type'].map({'non_update': 'delay only',
                                                                         'switch_update': 'switch',
                                                                         'stay_update': 'stay'})
        var_mapping = dict(x_position=dict(title='lateral position', ylabel='fraction of lateral distance', norm=60),
                           rotational_velocity=dict(title='rotational velocity', ylabel='roll (a.u.)', norm=False),
                           translational_velocity=dict(title='forward velocity', ylabel='pitch (a.u.)', norm=False),)

        # plot data
        ax = sfig.subplots(nrows=3, ncols=1, sharex=True)
        for ax_id, var in enumerate(vars):
            for (update, turn), group in trajectory_df.groupby(['update_type', 'turn_type']):
                type_id = np.argwhere(trajectory_df['update_type'].unique() == update)[0][0]
                linestyle = 'dotted' if turn == 'right' else 'solid'

                # transform y data to fraction of track
                var_df = pd.DataFrame(list(group[var]))
                y_labels = var_df.columns.mid.values  # TODO - check same across animals
                cue_locations = {k: np.round((v - np.min(y_labels)) / np.max(y_labels - np.min(y_labels)), 4)
                                 for k, v in self.virtual_track.cue_start_locations['y_position'].items()}
                for i, (cue_name, cue_loc) in enumerate(cue_locations.items()):
                    if turn == 'right':
                        ax[ax_id].axvline(cue_loc, linestyle='solid', color=self.colors['phase_dividers'], zorder=0,
                                   linewidth=0.75)
                track_fraction = (y_labels - np.min(y_labels)) / (np.max(y_labels) - np.min(y_labels))

                if var_mapping[var]['norm']:
                    mat = np.array(var_df) / var_mapping[var]['norm']
                else:
                    mat = np.array(var_df)

                # plot data
                trajectory_mean = np.nanmean(mat, 0)
                trajectory_err = sem(mat, 0)
                ax[ax_id].plot(track_fraction, trajectory_mean, color=self.colors['trials'][type_id],
                               linestyle=linestyle, label=update)
                ax[ax_id].fill_between(track_fraction, trajectory_mean - trajectory_err,
                                       trajectory_mean + trajectory_err, alpha=0.2,
                                       color=self.colors['trials'][type_id])
            ax[ax_id].set(title=var_mapping[var]['title'], ylabel=var_mapping[var]['ylabel'],
                   xlabel='fraction of track', xlim=(0, 1))

        handles, labels = ax[0].get_legend_handles_labels()
        which_handles = np.where(np.array(labels) == 'delay only')[0]
        handles, labels = np.array([[handles[i], turn] for i, turn in zip(which_handles,
                                                                          trajectory_df['turn_type'].unique())]).T
        ax[0].legend(list(handles), list(labels), loc='lower left')

        colors = [self.colors[t] for t in trajectory_df['update_type'].unique()]
        rainbow_text(0.05, 0.85, trajectory_df['update_type'].unique(), colors, ax=ax[0], size=10)

        return sfig

    def plot_trajectories_by_event(self, sfig, var='view_angle'):
        """plot trajectories aligned to update time for manuscript figure"""
        # get data for example trial
        temp_df = (self.group_df[['animal', 'session_id', 'trajectories']]
                   .explode('trajectories')
                   .reset_index(drop=True))
        trajectory_df = pd.concat([temp_df[['animal', 'session_id']],
                                   pd.DataFrame(list(temp_df['trajectories']))], axis=1)
        trajectory_df['update_type'] = trajectory_df['update_type'].map({'non_update': 'delay only',
                                                                         'switch_update': 'switch',
                                                                         'stay_update': 'stay'})

        ax = sfig.subplots(nrows=1, ncols=1)
        for (update, turn), group in trajectory_df.groupby(['update_type', 'turn_type']):
            ax_id = np.argwhere(trajectory_df['update_type'].unique() == update)[0][0]
            linestyle = 'dotted' if turn == 'right' else 'solid'

            # transform y data to fraction of track
            y_position = pd.DataFrame(list(group[var]))
            y_labels = y_position.columns.mid.values  # TODO - check same across animals
            cue_locations = {k: np.round((v - np.min(y_labels)) / np.max(y_labels - np.min(y_labels)), 4)
                             for k, v in self.virtual_track.cue_start_locations['y_position'].items()}
            for i, (cue_name, cue_loc) in enumerate(cue_locations.items()):
                if cue_name in ['delay cue', 'update cue', 'delay2 cue'] and turn == 'right':
                    ax.axvline(cue_loc - cue_locations['update cue'], linestyle='solid',
                               color=self.colors['phase_dividers'], zorder=0, linewidth=0.75)
            track_fraction = (y_labels - np.min(y_labels)) / np.max(y_labels - np.min(y_labels))
            track_fraction = track_fraction - cue_locations['update cue']

            # plot data
            trajectory_mean = np.rad2deg(np.nanmean(np.array(y_position), 0))
            trajectory_err = np.rad2deg(sem(np.array(y_position), 0))
            ax.plot(track_fraction, trajectory_mean, color=self.colors['trials'][ax_id], linestyle=linestyle, label=update)
            ax.fill_between(track_fraction, trajectory_mean - trajectory_err, trajectory_mean + trajectory_err,
                            alpha=0.2, color=self.colors['trials'][ax_id])

        ax.set(title='Trajectories', ylabel='view angle (degrees)', xlabel='position relative to update',
               ylim=(-35, 35), xlim=(cue_locations['delay cue'] - cue_locations['update cue'],
                                     cue_locations['delay2 cue'] - cue_locations['update cue']))
        colors = [self.colors[t] for t in trajectory_df['update_type'].unique()]
        rainbow_text(0.05, 0.85, trajectory_df['update_type'].unique(), colors, ax=ax, size=10)

        handles, labels = ax.get_legend_handles_labels()
        which_handles = np.where(np.array(labels) == 'delay only')[0]
        handles, labels = np.array([[handles[i], turn] for i, turn in zip(which_handles,
                                                                          trajectory_df['turn_type'].unique())]).T
        ax.legend(list(handles), list(labels), loc='lower left')

        return sfig

    def plot_event_durations(self, sfig):
        # get data
        group_durations = pd.concat(self.group_df['event_durations'].to_list(), axis=0)
        durations = (group_durations
                     .assign(sample_to_choice=lambda x: np.nansum([x['delay1'], x['delay2'], x['update']], axis=0))
                     .assign(delay1=lambda x: [np.nan if u == 1 else d for d, u in zip(x['delay1'], x['update_type'])])
                     .melt(var_name='event', value_name='duration', id_vars='update_type',
                           value_vars=['initial_cue', 'delay1', 'update', 'delay2', 'sample_to_choice', 'total_trial'])
                     .dropna(subset='duration', axis=0)
                     .assign(update_type=lambda x: x['update_type'].map({1: 'delay only', 2: 'switch', 3: 'stay'}))
                     .assign(event=lambda x: x['event'].map({'initial_cue': 'sample',
                                                             'delay1': '1st delay',
                                                             'update': 'update',
                                                             'delay2': '2nd delay',
                                                             'sample_to_choice': 'sample to choice',
                                                             'total_trial': 'total trial'})))

        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[1, 4]))

        # plot total trial duration
        sns.boxplot(data=durations.query('event == "total trial"'), x='update_type', y='duration', ax=ax[0], width=0.5,
                    palette=self.colors['trials'], medianprops={'color': 'white'}, showfliers=False)
        ax[0].set(title='Total trial length', ylabel='duration (s)', xlabel='')
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # breakdown by task phase
        sns.boxplot(data=durations.query('event not in ["total trial"]'), x='event', y='duration', hue='update_type',
                    ax=ax[1], width=0.5, palette=self.colors['trials'], medianprops={'color': 'white'},
                    showfliers=False, order=['sample', 'sample to choice', '1st delay', 'update', '2nd delay'])
        ax[1].set(title='Event lengths', ylabel='duration (s)')
        ax[1].get_legend().remove()
        colors = [self.colors[t] for t in durations['update_type'].unique()]
        rainbow_text(0.85, 0.85, durations['update_type'].unique(), colors, ax=ax[1], size=10)

        for a in ax:
            clean_box_plot(a, labelcolors=[self.colors['nan']] * len(durations['event'].unique()))

        return sfig

    def plot_trajectories_by_times(self, ax, var='view_angle', event='t_update'):
        temp_df = self.group_df[['animal', 'session_id', 'aligned_data']].explode('aligned_data').reset_index(drop=True)
        aligned_df = pd.concat([temp_df[['animal', 'session_id']], pd.DataFrame(list(temp_df['aligned_data']))], axis=1)
        aligned_times = aligned_df['aligned_times'].to_numpy()[0]
        aligned_df = (aligned_df
                      .query(f'start_label == "{event}" & var in (["y_position", "{var}"])')
                      .pivot(index=['animal', 'session_id', 'start_label', 'update_type', 'turn_type'],
                             columns='var',
                             values=['aligned_data'])
                      .droplevel(0, axis=1)
                      .reset_index()
                      .explode(['view_angle', 'y_position'])
                      .assign(update_type=lambda x: x.update_type.map({'non_update': 'non update',
                                                                       'switch_update': 'switch',
                                                                       'stay_update': 'stay'}))
                      .assign(aligned_times=[aligned_times] * 280))  #TODO - move this to the analysis side of things

        for (update, _, turn), group in aligned_df.groupby(['update_type', 'start_label', 'turn_type']):
            cleaned_data = [v for v in group['aligned_data'].values if len(np.shape(v)) == 2]
            aligned_data = np.vstack(cleaned_data)
            times = group['aligned_times'].values[0]

            # plot averages
            ax.plot(times, mean, color=self.colors[name[0]], label=f'{update}')
            ax.fill_between(times, mean - err, mean + err, alpha=0.2, color=self.colors[update])

        ax.axvline(0, linestyle='dashed', color=self.colors['nan'])
        colors = [self.colors[t] for t in aligned_df['update_type'].unique()]
        rainbow_text(0.05, 0.85, aligned_df['update_type'].unique(), colors, ax=ax[-1], size=10)

        return ax

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
            plot_distributions(rolling_df, axes=axes, column_name='prop_correct', group='update_type',
                               row_ids=[0, 1, 2],
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

        vars_to_plot = ['x_position', 'view_angle', *self.data[0]['analyzer'].analog_vars]
        turn_colors = dict(left=self.colors['left'], right=self.colors['right'])
        ncols = 3 * 2  # 1 column for each update type (non, switch, stay) * indiv + averages
        nrows = len(vars_to_plot)  # 1 col for each var
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 17), sharey='row')

        group_data = trajectory_df.groupby(['update_type', 'turn_type'])
        for name, group in group_data:
            col_id = \
            [ind * 2 for ind, k in enumerate(self.virtual_track.mappings['update_type'].values()) if k == name[0]][0]

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
        vars_to_plot = ['x_position', 'view_angle', *self.data[0]['analyzer'].analog_vars]

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

    def plot_all_event_durations(self, ):
        group_durations = pd.concat(self.group_df['event_durations'].to_list(), axis=0)
        durations = pd.melt(group_durations, var_name='event', value_name='duration', id_vars='update_type',
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
                .theme(self.rcparams)
                .on(fig)
                .plot()
        )

        # add text of medians + sd of
        new_line = '\n'
        plus_minus = '\u00B1'
        for ax in fig.axes:
            data = summary.query(f'update_type == "{ax.get_title()}"')
            text = [f'{r["event"]}: {r["median"]:.2f} {plus_minus} {r["std"]:.2f} {new_line}' for _, r in
                    data.iterrows()]
            ax.text(0.05, 0.75, ''.join(text), transform=ax.transAxes)

        fig.suptitle('Task event durations')
        self.results_io.save_fig(fig=fig, filename=f'event_durations', results_type=self.results_type)
