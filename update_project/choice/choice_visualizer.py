import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import more_itertools as mit

from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import sem

from update_project.general.results_io import ResultsIO
from update_project.general.plots import rainbow_text, clean_box_plot, add_task_phase_lines, colorline
from update_project.base_visualization_class import BaseVisualizationClass

rng = np.random.default_rng(12345)


class ChoiceVisualizer(BaseVisualizationClass):

    def __init__(self, data, session_id=None, grid_search=False, target_var='choice'):
        super().__init__(data)
        self.grid_search = grid_search

        if session_id:
            self.results_type = 'session'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path(__file__).parent.stem,
                                        session_id=session_id, tags=target_var)
        else:
            self.results_type = 'group'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path(__file__).parent.stem,
                                        tags=target_var)

        # get session visualization info
        for sess_dict in self.data:
            if grid_search:
                sess_dict.update(dict(grid_search_data=sess_dict['analyzer'].grid_search_data,
                                      grid_search_params=sess_dict['analyzer'].grid_search_params))
            sess_dict.update(dict(agg_data=sess_dict['analyzer'].agg_data,
                                  # vis_data=sess_dict['analyzer'].vis_data,
                                  output_data=sess_dict['analyzer'].output_data))

        self.group_df = pd.DataFrame(self.data)
        self.group_df.drop('analyzer', axis='columns', inplace=True)

    def plot(self):
        self.plot_choice_switches()
        self.plot_dynamic_choice_by_position()

        if self.grid_search:
            self.plot_grid_search_results()

    def plot_choice_commitment(self, sfig, num_trials=20):
        # get data
        predict_df = self._get_group_prediction().reset_index(drop=True)
        predict_df = predict_df.rename_axis('trial').reset_index()
        predict_df_by_time = (predict_df
                              .explode(['predict', 'y_position', 'timestamps', 'log_likelihood']))

        # split up into bins by cues and by y_position
        y_lims = (self.virtual_track.cue_start_locations['y_position']['initial cue'],
                  np.max(self.virtual_track.coords))
        cue_fractions = dict()
        for cue_name, loc in self.virtual_track.cue_start_locations['y_position'].items():
            cue_fractions[cue_name] = (loc - y_lims[0]) / (y_lims[-1] - y_lims[0])
        cue_bins = pd.cut(predict_df_by_time['y_position'], [*list(cue_fractions.values()), 1],
                                 labels=list(cue_fractions.keys()))
        predict_df_by_time['cue'] = cue_bins
        predict_df_by_time['target'] = predict_df_by_time['target'].map({0: 'left', 1: 'right'})
        cue_performance = predict_df_by_time.groupby(['cue', 'session_id'])['log_likelihood'].mean().reset_index()

        example_animal, example_session = (25, 'S25_210913')  # TODO - find good representative session
        y_position_bins = pd.cut(predict_df_by_time['y_position'], np.linspace(0, 1.01, 30))
        predict_df_by_time['y_position_binned'] = y_position_bins.apply(lambda x: x.mid)
        trial_performance = (predict_df_by_time
                      .query(f'animal == {example_animal} & session_id == "{example_session}"')
                      .groupby(['session_id', 'trial', 'target', 'y_position_binned'])[['predict']]
                      .mean().reset_index())

        # plot example trials
        ax = sfig.subplots(nrows=1, ncols=2)
        for g_name, g_data in trial_performance.groupby('target'):
            indiv_mat = g_data.pivot(index=['trial'], columns=['y_position_binned'], values='predict').dropna(axis=0)  # drop extra trial/target grouping artifacts
            pos_bins = indiv_mat.columns.to_numpy()
            random_samples = rng.choice(range(np.shape(indiv_mat)[0]), num_trials)  # TODO - only full delay non-update
            trials = np.stack(indiv_mat.to_numpy()).T[:, random_samples]

            linestyle = 'solid' if g_name == 'right' else 'dashed'
            for t in trials.T:
                ax[0] = colorline(pos_bins, t, cmap=self.colors['choice_cmap'], linestyle=linestyle, alpha=0.5, ax=ax[0])
        ax[0] = add_task_phase_lines(ax[0], cue_locations=cue_fractions, text_brackets=True)
        ax[0].set(title=f'Example trials', xlabel='fraction of track', ylabel='p(right choice)', ylim=(0, 1.1))

        custom_lines = [Line2D([0], [0], color=self.colors['choice'][-1], linestyle='solid', linewidth=1),
                        Line2D([0], [0], color=self.colors['choice'][-1], linestyle='dashed', linewidth=1),]
        ax[0].legend(custom_lines, trial_performance['target'].unique(), loc='lower left')

        # plot session averages
        sns.boxplot(cue_performance, x='cue', y='log_likelihood', ax=ax[1], width=0.5, medianprops={'color': 'white'},
                    showfliers=False, palette=self.colors['choice_commitment'][1:])
        ax[1].set(title='Accuracy', xlabel='task phase', ylabel='log likelihood')
        ax[1].axhline(-1, linestyle='dashed', color='k', label='chance', alpha=0.5, zorder=0)
        ax[1].axhline(0, linestyle='dashed', color='k', label='perfect', alpha=0.5, zorder=0)
        clean_box_plot(ax[1])

        sfig.suptitle('Choice commitment estimation', fontsize=12)

        return sfig

    def _get_group_prediction(self):
        # df_list = []
        # for ind, sess in self.group_df.iterrows():
        #     sess['vis_data']['session_id'] = self.group_df['session_id']
        #     sess['vis_data']['animal'] = self.group_df['animal']
        #     df_list.append(sess['vis_data'])
        # predict_df = pd.concat(df_list, axis=1)
        #
        # data_to_explode = [col for col in predict_df.columns.values if col not in ['session_id', 'animal', 'target']]
        # predict_df = predict_df.explode(data_to_explode)
        # self.group_df.drop(['vis_data'], axis=1)

        predict_df = pd.concat([self.group_df[['session_id', 'animal']], pd.json_normalize(self.group_df['agg_data'])],
                               axis=1)
        predict_df.drop(['update_timestamps', 'update_predict_data'], axis=1, inplace=True)
        try:  # TODO - fix later that target data is being stored differently sometimes
            data_to_explode = [col for col in predict_df.columns.values if col not in ['session_id', 'animal']]
            predict_df = predict_df.explode(data_to_explode)
        except ValueError:
            data_to_explode = [col for col in predict_df.columns.values if col not in ['session_id', 'animal', 'target']]
            predict_df = predict_df.explode(data_to_explode)

        # TODO - update explosion once I have the update data input correctly
        # predict_df = pd.concat([predict_df.explode(d).reset_index()[d] if d != 'predict'
        #                         else predict_df[['session_id', 'animal', 'predict']].explode(d).reset_index(drop=True)
        #                         for d in data_to_explode], axis=1)

        return predict_df

    def _get_group_grid_search(self):
        # get giant dataframe of all decoding data
        summary_df_list = []
        for _, sess_data in self.group_df.iterrows():
            grid_search_df = sess_data['grid_search_data'].copy(deep=True)
            grid_search_df['session_id'] = sess_data['session_id']
            grid_search_df['animal'] = sess_data['animal']

            # get l2 regularizer in number form
            grid_search_df['regularizer'] = grid_search_df['regularizer'].apply(lambda x: np.round(x.l2, 3) if x is not None else x)

            # explode history so has it's own columns
            new_df = pd.concat([grid_search_df, pd.json_normalize(grid_search_df['history'])], axis=1)
            new_df.drop(['history'], axis='columns', inplace=True)

            summary_df_list.append(new_df)
        group_grid_search_df = pd.concat(summary_df_list, axis=0, ignore_index=True)

        return group_grid_search_df

    def plot_dynamic_choice_by_position(self):
        predict_df = self._get_group_prediction()
        predict_df = predict_df.rename_axis('trial').reset_index()
        predict_df_by_time = (predict_df
                              .explode(['predict', 'y_position', 'timestamps', 'log_likelihood']))
        # TODO - readd update data when I have it saved correctly
        y_position_bins = pd.cut(predict_df_by_time['y_position'], np.linspace(0, 1.01, 30))
        predict_df_by_time['y_position_binned'] = y_position_bins.apply(lambda x: x.mid)
        quadrant_bins = pd.cut(predict_df_by_time['y_position'], np.linspace(0, 1.01, 5))
        predict_df_by_time['quadrant'] = quadrant_bins.apply(lambda x: x.left)
        predict_df_by_time['target'] = predict_df_by_time['target'].map({0: 'left', 1: 'right'})

        group_avg = (predict_df_by_time
                     .groupby(['y_position_binned', 'target'])[['predict', 'log_likelihood']]
                     .agg(['mean', 'sem'])).reset_index()
        group_avg.columns = ['_'.join(c) if c[1] != '' else c[0] for c in group_avg.columns.to_flat_index()]
        sess_avg = (predict_df_by_time
                    .groupby(['session_id', 'y_position_binned', 'target'])[['predict', 'log_likelihood']]
                    .agg(['mean', 'sem'])).reset_index()
        sess_avg.columns = ['_'.join(c) if c[1] != '' else c[0] for c in sess_avg.columns.to_flat_index()]
        sess_indiv = (predict_df_by_time
                      .groupby(['session_id', 'trial', 'target', 'y_position_binned'])[['predict', 'log_likelihood']]
                      .mean().reset_index())

        fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
        sfigs = fig.subfigures(1, 3)

        axes = sfigs[0].subplots(2, 1)
        axes_s = sfigs[1].subplots(2, 1)
        for mi, m in enumerate(['predict', 'log_likelihood']):
            for g_name, g_data in group_avg.groupby('target'):
                axes[mi].plot(g_data['y_position_binned'], g_data[f'{m}_mean'], color=self.colors[g_name])
                axes[mi].fill_between(g_data['y_position_binned'], g_data[f'{m}_mean'] + g_data[f'{m}_sem'],
                                 g_data[f'{m}_mean'] - g_data[f'{m}_sem'], color=self.colors[g_name], alpha=0.5)
                axes[mi].set(title=f'Group {m} average', xlabel='fracton of track', ylabel=m)

            for g_name, g_data in sess_avg.groupby(['target', 'session_id']):
                axes_s[mi].plot(g_data['y_position_binned'], g_data[f'{m}_mean'], color=self.colors[g_name[0]])
                axes_s[mi].set(title=f'Group {m} average', xlabel='fracton of track', ylabel=m)

        sess_performance = predict_df_by_time.groupby(['quadrant', 'session_id'])['log_likelihood'].mean().reset_index()
        (  # plot averages for log likelihood and overall position by trial
            so.Plot(sess_performance, x='quadrant', y='log_likelihood')
                .add(so.Line(marker="o"), so.Agg())
                .add(so.Range(), so.Est(errorbar="sd"))
                .theme(rcparams)
                .layout(engine='constrained')
                .label(title='Session averages of estimated choice')
                .on(sfigs[2])
                .plot()
        )
        sfigs[2].axes[0].axhline(-1, linestyle='dashed', color='k')
        sfigs[2].axes[0].axhline(0, linestyle='dashed', color='k')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'prediction_group', tight_layout=False)

        for s_name, s_data in sess_indiv.groupby(['session_id']):
            fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
            sfigs = fig.subfigures(1, 2)
            axes = sfigs[0].subplots(2, 1)
            axes_indiv = sfigs[1].subplots(2, 1)
            for mi, m in enumerate(['predict', 'log_likelihood']):
                for g_name, g_data in s_data.groupby(['target']):
                    indiv_mat = g_data.pivot(index=['session_id', 'trial'], columns=['y_position_binned'],
                                             values=m).dropna(axis=0) # drop extra trial/target grouping artifacts
                    if np.size(indiv_mat):
                        pos_bins = indiv_mat.columns.to_numpy()
                        sess_mean = np.nanmean(np.stack(indiv_mat.to_numpy()), axis=0)
                        sess_err = sem(np.stack(indiv_mat.to_numpy()), axis=0)
                        axes[mi].plot(pos_bins, sess_mean, color=self.colors[g_name])
                        axes[mi].fill_between(pos_bins, sess_mean + sess_err, sess_mean - sess_err, color=self.colors[g_name],
                                              alpha=0.5)
                        axes[mi].set(title=f'Group {m} average', xlabel='fracton of track', ylabel=m)

                        axes_indiv[mi].plot(pos_bins, np.stack(indiv_mat.to_numpy()).T,
                                            color=self.colors[g_name])
                        axes_indiv[mi].set(title=f'Individual {m} trials', xlabel='fracton of track', ylabel=m)

            self.results_io.save_fig(fig=fig, axes=axes, filename=f'prediction_{s_name}', tight_layout=False)

    def plot_grid_search_results(self):
        grid_search_df = self._get_group_grid_search()
        grid_search_params = self.group_df['grid_search_params'].to_numpy()[0]
        metrics = ['loss', 'val_loss', 'binary_accuracy', 'val_binary_accuracy']
        cmap_b = sns.color_palette(palette='Blues', n_colors=len(grid_search_params['regularizer']))
        cmap_p = sns.color_palette(palette='Purples', n_colors=len(grid_search_params['regularizer']))
        cmap = sns.color_palette([val for pair in zip(cmap_b, cmap_p) for val in pair])
        cmap_mapping = {str(round(k, 3)): v for k, v in zip(grid_search_df['regularizer'].unique(), range(len(cmap)))}

        # 2 rows for accuracy, loss history, 2 for final scores, 1 column for each param
        fig, axes = plt.subplots(nrows=4, ncols=len(grid_search_params['batch_size']), squeeze=False, sharey='row')
        for name, group in grid_search_df.groupby(list(grid_search_params.keys()), dropna=False):
            # get group specific data
            col_id = [ind for ind, val in enumerate(grid_search_params['batch_size']) if val == name[0]][0]
            if name[3] == grid_search_params['learning_rate'][0]:
                color = cmap_b[cmap_mapping[str(round(name[2], 3))]]
            elif name[3] == grid_search_params['learning_rate'][1]:
                color = cmap_p[cmap_mapping[str(round(name[2], 3))]]
            title = f'batch_size: {name[0]}'
            epoch = range(group['epochs'].values[0])
            results = dict()
            for m in metrics:
                results[m] = np.nanmean(np.array([l for l in group[m].to_numpy()]).T, axis=1)
            label = f'reg: {np.round(group["regularizer"].to_numpy()[0], 3)}, learn: {name[3]}'

            # plot history over training
            kwargs = dict(color=color, linestyle='dashed')
            if name[1] == 20:
                kwargs.update(label=label)
            axes[0][col_id].plot(epoch, results['loss'], **kwargs)
            axes[0][col_id].plot(epoch, results['val_loss'], color=color)
            axes[0][col_id].set(xlabel='epoch', ylabel='loss', ylim=(0, 1.2), title=f'Loss - {title}')
            axes[0][col_id].legend()

            axes[1][col_id].plot(epoch, results['binary_accuracy'], **kwargs)
            axes[1][col_id].plot(epoch, results['val_binary_accuracy'], color=color)
            axes[1][col_id].set(xlabel='epoch', ylabel='accuracy', ylim=(0, 1), title=f'Accuracy - {title}')
            axes[1][col_id].legend()

        # violin plots
        count = 0
        for name, group in grid_search_df.groupby(['batch_size']):
            group['regularizer'] = group['regularizer'].fillna(0)
            axes[2][count] = sns.violinplot(data=group, x='epochs', y='score', palette=cmap, ax=axes[2][count],
                                            hue=group[['learning_rate', 'regularizer']].apply(tuple, axis=1),)
            plt.setp(axes[2][count].collections, alpha=.25)
            sns.stripplot(data=group, y='score', x='epochs', palette=cmap, size=3, jitter=True, dodge=True,
                          ax=axes[2][count], hue=group[['learning_rate', 'regularizer']].apply(tuple, axis=1),)
            axes[2][count].set(title=f'Final loss - batch_size {name}')
            axes[2][count].legend([], [], frameon=False)

            axes[3][count] = sns.violinplot(data=group, x='epochs', y='accuracy', palette=cmap, ax=axes[3][count],
                                            hue=group[['learning_rate', 'regularizer']].apply(tuple, axis=1))
            plt.setp(axes[3][count].collections, alpha=.25)
            sns.stripplot(data=group, y='accuracy', x='epochs', size=3, jitter=True, palette=cmap, dodge=True,
                          hue=group[['learning_rate', 'regularizer']].apply(tuple, axis=1), ax=axes[3][count])
            axes[3][count].set(title=f'Final accuracy - batch_size {name}')
            axes[3][count].legend([], [], frameon=False)
            count += 1

        self.results_io.save_fig(fig=fig, axes=axes, filename='grid_search_results', results_type=self.results_type)

        # sort outputs by best accuracy
        grid_search_results = grid_search_df.groupby(list(grid_search_params.keys()), dropna=False).mean()
        grid_search_results.sort_values(['accuracy'], ascending=False, inplace=True)
        grid_search_results.reset_index()
        fname = self.results_io.get_data_filename(filename='grid_search_table', format='csv')
        grid_search_results.to_csv(fname, na_rep='nan')

    def get_choice_switch_times(self, choice_estimate, timestamps, decision_thresholds=(0.1, 0.9), min_duration=100):
        choice_switches = []
        if (choice_estimate < decision_thresholds[0]).any() and (choice_estimate > decision_thresholds[1]).any():
            choices = [list(mit.run_length.encode(choice_estimate > decision_thresholds[1])),
                       list(mit.run_length.encode(choice_estimate < decision_thresholds[0]))]

            for c_type, c in enumerate(choices):
                valid_choices_left = [count if f and count > min_duration else False for f, count in c]
                choice_ind = 0
                for ind, count in enumerate(c):
                    if (count[1] in valid_choices_left) and (ind in np.where(valid_choices_left)[0]):
                        if ~np.isnan(timestamps[choice_ind]):
                            choice_switches.append(dict(choice_type=c_type,
                                                        choice_start_ind=choice_ind,
                                                        choice_stop_ind=choice_ind + count[1] - 1,
                                                        t_start_choice=timestamps[choice_ind],
                                                        t_stop_choice=timestamps[choice_ind + count[1] - 1]))
                    choice_ind = choice_ind + count[1]

            if len(np.unique([c['choice_type'] for c in choice_switches])) > 1:  # at least one switch btwn choices
                choice_df = pd.DataFrame(choice_switches)
                choice_df.sort_values('t_start_choice', inplace=True)
                diff_from_next = choice_df['t_start_choice'].shift - choice_df['t_stop_choice'].to_numpy()[0]

        return choice_switches

    def plot_choice_switches(self):
        predict_df = self._get_group_prediction()
        choice_switch_df = predict_df.apply(lambda x: self.get_choice_switch_times(x['predict'], x['timestamps'],),
                                            axis=1)

        self.results_io.save_fig(fig=fig, axes=axes, filename='choice_switches', results_type=self.results_type)
