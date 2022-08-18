import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_color_theme


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
            sess_dict.update(dict(grid_search_data=sess_dict['rnn'].grid_search_data,
                                  grid_search_params=sess_dict['rnn'].grid_search_params))

        self.group_df = pd.DataFrame(data)

    def plot(self):
        self.plot_grid_search_results()
        self.plot_dynamic_choice_by_position()

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

    def plot_grid_search_results(self):
        grid_search_df = self._get_group_grid_search()
        grid_search_params = self.group_df['grid_search_params'].to_numpy()[0]
        metrics = ['loss', 'val_loss', 'binary_accuracy', 'val_binary_accuracy']
        cmap = sns.color_palette(palette='Blues', n_colors=len(grid_search_params['regularizer']))
        cmap_mapping = {str(round(k, 3)): v for k, v in zip(grid_search_df['regularizer'].unique(), range(len(cmap)))}

        # 2 rows for accuracy, loss history, 2 for final scores, 1 column for each param
        fig, axes = plt.subplots(nrows=4, ncols=len(grid_search_params['batch_size']), squeeze=False, sharey='row')
        for name, group in grid_search_df.groupby(list(grid_search_params.keys()), dropna=False):
            # get group specific data
            col_id = [ind for ind, val in enumerate(grid_search_params['batch_size']) if val == name[0]][0]
            color = cmap[cmap_mapping[str(round(name[2], 3))]]
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
                                            hue=group[['regularizer', 'learning_rate']].apply(tuple, axis=1),)
            plt.setp(axes[2][count].collections, alpha=.25)
            sns.stripplot(data=group, y='score', x='epochs', palette=cmap, size=3, jitter=True, dodge=True,
                          ax=axes[2][count], hue=group[['regularizer', 'learning_rate']].apply(tuple, axis=1),)
            axes[2][count].set(title=f'Final loss - batch_size {name}')
            handles, labels = axes[2][count].get_legend_handles_labels()
            new_labels = [str(np.round(float(l), 3)) for l in labels]
            axes[2][count].legend(handles[:int(len(new_labels)/2)], new_labels[:int(len(new_labels)/2)])

            axes[3][count] = sns.violinplot(data=group, x='epochs', y='accuracy', palette=cmap, ax=axes[3][count],
                                            hue=group[['regularizer', 'learning_rate']].apply(tuple, axis=1))
            plt.setp(axes[3][count].collections, alpha=.25)
            sns.stripplot(data=group, y='accuracy', x='epochs', size=3, jitter=True, palette=cmap, dodge=True,
                          hue=group[['regularizer', 'learning_rate']].apply(tuple, axis=1), ax=axes[3][count])
            axes[3][count].set(title=f'Final accuracy - batch_size {name}')
            handles, labels = axes[3][count].get_legend_handles_labels()
            new_labels = [str(np.round(float(l), 3)) for l in labels]
            axes[3][count].legend(handles[:int(len(new_labels)/2)], new_labels[:int(len(new_labels)/2)])
            count += 1

        self.results_io.save_fig(fig=fig, axes=axes, filename='grid_search_results', results_type=self.results_type)

        grid_search_results = grid_search_df.groupby(list(grid_search_params.keys())).mean()
        grid_search_results.sort_values(['accuracy'], ascending=False)

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