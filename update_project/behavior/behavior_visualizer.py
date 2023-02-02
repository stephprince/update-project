import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from pathlib import Path

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_color_theme
from update_project.statistics import get_fig_stats
from update_project.virtual_track import UpdateTrack


class BehaviorVisualizer:
    def __init__(self, data, session_id=None):
        self.virtual_track = UpdateTrack()
        self.colors = get_color_theme()

        if session_id:
            self.results_type = 'session'
            self.session_id = session_id
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem, session_id=session_id)
        else:
            self.results_type = 'group'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

        # get session visualization info
        for sess_dict in data:
            sess_dict.update(proportion_correct_by_update=sess_dict['behavior'].proportion_correct_by_update)
            sess_dict.update(proportion_correct_by_phase=sess_dict['behavior'].proportion_correct_by_phase)
            sess_dict.update(proportion_correct_by_trial=sess_dict['behavior'].proportion_correct_by_trial)
            sess_dict.update(session_type=sess_dict['behavior'].session_type)
            sess_dict.update(trajectories=sess_dict['behavior'].trajectories)
            sess_dict.update(aligned_data=sess_dict['behavior'].aligned_data)

        self.group_df = pd.DataFrame(data)
        self.data = data

    def plot(self):
        if self.results_type=='session':
            self.plot_session_performance()
        else:
            self.plot_total_phase_count()
        self.plot_proportion_correct_by_phase()
        # self.plot_trajectories()
        self.plot_aligned_data()
        self.plot_proportion_correct_by_update()
    def plot_proportion_correct_by_phase(self):
        temp_df = self.group_df[['animal', 'proportion_correct_by_phase']].explode('proportion_correct_by_phase').reset_index(drop=True)
        df_by_phase = pd.concat([temp_df['animal'], pd.DataFrame(list(temp_df['proportion_correct_by_phase']))], axis=1)
        df_by_bin = df_by_phase.explode('prop_correct').reset_index(drop=True)
        df_by_bin = pd.DataFrame(df_by_bin.to_dict())

        nrows = 1  # 1 row for each plot type (cum fract, hist, violin)
        ncols = 1  # 1 column for binned, 1 for rolling
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,8))

        # rolling proportion correct
        title = 'Performance - rolling'
        xlabel = 'proportion correct by phase'
        rolling_df = df_by_bin[df_by_bin["type"] == 'rolling']
        if np.size(rolling_df['prop_correct'].dropna()):
            data_to_plot = rolling_df.query('phase != "linear"')
            # plot_distributions(data_to_plot, axes=axes, column_name='prop_correct', group='phase',
            #                    row_ids=[0, 1, 2],
            #                    col_ids=[0, 0, 0, 0], xlabel=xlabel, title=title, stripplot=False)
            # plot_distributions(data_to_plot, axes=axes, column_name='prop_correct', group='animal', row_ids=[0, 1, 2],
            #                    col_ids=[1, 1, 1, 1], xlabel=xlabel, title=title, stripplot=False)
            sns.boxplot(data=data_to_plot, x='phase', y='prop_correct', showfliers=False)
            plt.axhline(y=0.5, alpha=0.3, linestyle='dashed', color='k')
            fig.suptitle('Rolling Behavioral Performance by Phase for All Animals')
            self.results_io.save_fig(fig=fig, axes=axes, filename='performance_by_phase_boxplot',
                                     results_type=self.results_type)

        # # binned proportion correct
        # title = 'Performance - binned'
        # xlabel = 'proportion correct by phase'
        # binned_df = df_by_bin[df_by_bin["type"] == 'binned']
        # if np.size(binned_df['prop_correct'].dropna()):
        #     plot_distributions(binned_df, axes=axes, column_name='prop_correct', group='phase', row_ids=[0, 1, 2],
        #                        col_ids=[2, 2, 2, 2], xlabel=xlabel, title=title)
        #     plot_distributions(binned_df, axes=axes, column_name='prop_correct', group='animal', row_ids=[0, 1, 2],
        #                        col_ids=[3, 3, 3, 3], xlabel=xlabel, title=title)

        # wrap up and save plot
        # for col in range(ncols):
        #     axes[1][col].set_xlim((0, 1))
        #     axes[2][col].set_ylim((0, 1))
        #     axes[1][col].plot([0.5, 0.5], axes[1][col].get_ylim(), linestyle='dashed', color=[0, 0, 0, 0.5])
        #     axes[2][col].plot(axes[2][col].get_xlim(), [0.5, 0.5], linestyle='dashed', color=[0, 0, 0, 0.5])
        #
        # # save figure
        # fig.suptitle(f'Behavioral performance by phase - all animals')
        # self.results_io.save_fig(fig=fig, axes=axes, filename=f'performance by phase', results_type=self.results_type)

    def plot_proportion_correct_by_update(self):
        # explode df so each prop correct value has one row (duplicates for each session/animal
        temp_df = self.group_df[['animal', 'proportion_correct_by_update']].explode('proportion_correct_by_update').reset_index(drop=True)
        df_by_update_type = pd.concat([temp_df['animal'], pd.DataFrame(list(temp_df['proportion_correct_by_update']))], axis=1)
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

    def plot_total_phase_count(self):
        # sort, group, and clean up data
        temp_df = self.group_df[['animal','session_id','session_type', 'is_ephys_session']]
        temp_df = temp_df[temp_df.is_ephys_session != 1]
        temp_df=temp_df.drop(columns=['is_ephys_session'])
        grouped_df = temp_df.groupby(['session_type', 'animal']).describe().reset_index()
        plot_data = pd.DataFrame(data={'animal': grouped_df['animal'], 'session_type': grouped_df['session_type'], 'count': grouped_df['session_id']['count']})
        plot_data=pd.DataFrame(plot_data.to_dict())
        plot_data['session_type'] = pd.Categorical(plot_data['session_type'],
                                                   categories=['linear', 'ymaze_short', 'ymaze_long', 'delay1',
                                                               'delay2',
                                                               'delay3', 'update'], ordered=True)
        nrows = 1
        ncols = 1

        # bar graph
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
        sns.barplot(x=plot_data['session_type'], y=plot_data['count'], alpha=0.5)
        sns.swarmplot(x=plot_data['session_type'], y=plot_data['count'])
        fig.suptitle('Average Time Spent on Each Trial Type')
        self.results_io.save_fig(fig=fig, axes=axes, filename='total_phase_count_barplot', results_type=self.results_type)

        # violin plot
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
        sns.violinplot(data=plot_data, x='session_type', y='count')
        fig.suptitle('Average Time Spent on Each Trial Type')
        self.results_io.save_fig(fig=fig, axes=axes, filename='total_phase_count_violinplot',
                                 results_type=self.results_type)

        # box plot
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 8))
        sns.boxplot(data=plot_data, x='session_type', y='count', showfliers=False)
        fig.suptitle('Average Time Spent on Each Trial Type')
        self.results_io.save_fig(fig=fig, axes=axes, filename='total_phase_count_boxplot',
                                 results_type=self.results_type)
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
    def plot_session_performance(self):
        temp_df=self.group_df['proportion_correct_by_trial'].values[0]
        nrows = 2
        ncols = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), sharex=True)

        #plot proportion correct line
        axes[0].plot(temp_df['prop_correct'])

        # plot the switch and stay trials
        switch_df = temp_df[temp_df['phase'] == 'switch_update']['prop_correct']
        stay_df = temp_df[temp_df['phase'] == 'stay_update']['prop_correct']
        axes[0].plot(switch_df,marker='*',linestyle='None', markersize='5', label='switch update', color=self.colors['switch'])
        axes[0].plot(stay_df, marker='D', linestyle='None', markersize='3', label='stay update', color=self.colors['stay'])

        #plot the left and right proportion correct lines
        #TODO move this to analyzer somehow and remove hardcoded trial window
        left_df = (temp_df[temp_df['turn_type'] == 2]['correct']).rolling(30, min_periods=0).mean()
        right_df = (temp_df[temp_df['turn_type'] == 1]['correct']).rolling(30, min_periods=0).mean()
        axes[1].plot(left_df, label='left trials', marker='*', linestyle='None', markersize='5', color=self.colors['left'])
        axes[1].plot(right_df, label='right trials', marker='*', linestyle='None', markersize='5', color=self.colors['right'])

        # add shading to indicate trial type to both
        current_phase=temp_df['phase'][0]
        ind=0
        update=False
        all_phases=[]
        for trial in range(len(temp_df)-1):
            if temp_df['phase'][trial+1] != current_phase:
                previous_phase=current_phase
                current_phase=temp_df['phase'][trial+1]
                if current_phase in all_phases:
                    label='_'
                else:
                    label=''
                    all_phases.append(current_phase)
                if update and current_phase not in ['stay_update','switch_update','delay4']:
                    update=False
                    axes[0].axvspan(ind, trial+1,facecolor='c', alpha=0.1, label=f'{label}update')
                    axes[1].axvspan(ind, trial + 1, facecolor='c', alpha=0.1, label=f'{label}update')
                    ind = trial + 1
                elif not update and current_phase in ['stay_update', 'switch_update']:
                    update = True
                    axes[0].axvspan(ind, trial + 1, color=self.colors[previous_phase], label=f'{label}{previous_phase}')
                    axes[1].axvspan(ind, trial + 1, color=self.colors[previous_phase], label=f'{label}{previous_phase}')
                    ind = trial + 1
                elif not update:
                    update=False
                    axes[0].axvspan(ind, trial+1, color=self.colors[previous_phase], label=f'{label}{previous_phase}')
                    axes[1].axvspan(ind, trial + 1, color=self.colors[previous_phase], label=f'{label}{previous_phase}')
                    ind=trial+1
                elif trial==len(temp_df)-2 and previous_phase in ['stay_update','switch_update','delay4']:
                    axes[0].axvspan(ind, len(temp_df), facecolor='c', alpha=0.1, label=f'{label}update')
                    axes[1].axvspan(ind, len(temp_df), facecolor='c', alpha=0.1, label=f'{label}update')

        # make switch/stay visually appealing
        axes[0].legend()
        axes[0].set(ylabel='Proportion Correct')
        axes[0].set(ylim=[0,1], xlim=[0,len(temp_df)])
        axes[0].axhline(y=0.5, alpha=0.1, linestyle='dashed', color='k')

        # make left/right visually appealing
        axes[1].legend()
        axes[1].set(xlabel='Number of Trials', ylabel='Proportion Correct')
        axes[1].set(ylim=[0, 1], xlim=[0, len(temp_df)])
        axes[1].axhline(y=0.5, alpha=0.1, linestyle='dashed', color='k')

        # add overall title and save
        axes[0].set_title(f'Session Performance - %s' % self.session_id)
        self.results_io.save_fig(fig=fig,axes=axes,filename=f'session_performance', results_type=self.results_type)