import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_color_theme


class DynamicChoiceVisualizer:

    def __init__(self, data, session_id=None, grid_search=False, target_var='choice'):
        self.data = data
        self.grid_search = grid_search

        if session_id:
            self.results_type = 'session'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem,
                                        session_id=session_id, tags=target_var)
        else:
            self.results_type = 'group'
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem,
                                        tags=target_var)

        # get session visualization info
        for sess_dict in self.data:
            if grid_search:
                sess_dict.update(dict(grid_search_data=sess_dict['rnn'].grid_search_data,
                                      grid_search_params=sess_dict['rnn'].grid_search_params))
            sess_dict.update(dict(agg_data=sess_dict['rnn'].agg_data,
                                  output_data=sess_dict['rnn'].output_data))

        self.group_df = pd.DataFrame(self.data)
        self.group_df.drop('rnn', axis='columns', inplace=True)

    def plot(self):
        self.plot_dynamic_choice_by_position()

        if self.grid_search:
            self.plot_grid_search_results()

    def _get_group_prediction(self):
        predict_df = pd.concat([self.group_df[['session_id', 'animal']], pd.json_normalize(self.group_df['agg_data'])],
                               axis=1)

        data_to_explode = [col for col in predict_df.columns.values if col not in ['session_id', 'animal']]
        predict_df = pd.concat([predict_df.explode(d).reset_index()[d] if d != 'predict'
                                else predict_df[['session_id', 'animal', 'predict']].explode(d).reset_index(drop=True)
                                for d in data_to_explode], axis=1)

        # think below only works with multiple animals/sessions so using above
        # predict_df = predict_df.set_index(['session_id', 'animal']).apply(pd.Series.explode).reset_index()

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
        data_to_not_explode = ['session_id', 'animal', 'trial', 'target']
        data_to_explode = [col for col in predict_df.columns.values if col not in data_to_not_explode]
        pos_summary_df = pd.concat([predict_df.explode(d).reset_index()[d] if d != 'predict'
                                else predict_df[[*data_to_not_explode, 'predict']].explode(d).reset_index(drop=True)
                                for d in data_to_explode], axis=1)

        y_position_bins = pd.cut(pos_summary_df['y_position'], np.linspace(0, 1.01, 30))
        pos_summary_df['y_position_binned'] = y_position_bins.apply(lambda x: x.mid)
        trial_cols = ['trial', 'y_position_binned', 'target', 'predict', 'log_likelihood', 'update_predict_data']
        palette = sns.color_palette(n_colors=len(pos_summary_df['target'].unique()))

        # plot averages for whole group
        fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
        axes[0][0] = sns.lineplot(data=pos_summary_df, x='y_position_binned', y='predict', hue='target',
                                  estimator='mean', ci=None, ax=axes[0][0], palette=palette)
        axes[0][0].set(title='LSTM prediction - non-update trials')
        axes[0][1] = sns.lineplot(data=pos_summary_df, x='y_position_binned', y='log_likelihood', hue='target',
                                  estimator='mean', ci=None, ax=axes[0][1], palette=palette)
        axes[0][1].set(title='Log likelihood - non-update trials')
        axes[0][2] = sns.lineplot(data=pos_summary_df, x='y_position_binned', y='update_predict_data', hue='target',
                                  estimator='mean', ci=None, ax=axes[0][2], palette=palette)
        axes[0][2].set(title='LSTM prediction - update trials')

        for a, l in {'0': (0.9, 0.1), '1': (0, -1), '2': (0.9, 0.1)}.items():  # threshold labels for each plot
            axes[0][int(a)].axhline(l[0], linestyle='dashed', color='k')
            axes[0][int(a)].axhline(l[1], linestyle='dashed', color='k')

        self.results_io.save_fig(fig=fig, axes=axes, filename=f'prediction_group')

        for name, session_df in pos_summary_df.groupby(['session_id']):
            spatial_map_trials = session_df[trial_cols].groupby(['trial', 'y_position_binned']).mean()
            palette = sns.color_palette(n_colors=len(session_df['target'].unique()))

            # plot overall predictions  # TODO - fix target plotting using new seaborn visualization objects
            fig, axes = plt.subplots(nrows=2, ncols=3, squeeze=False)
            axes[0][0] = sns.lineplot(data=session_df, x='y_position_binned', y='predict', hue='target',
                                      estimator='mean', ci=None, ax=axes[0][0], palette=palette)
            axes[0][0].set(title='LSTM prediction - non-update trials')
            axes[1][0] = sns.lineplot(data=spatial_map_trials, x='y_position_binned', y='predict', hue='target',
                                      style='trial', estimator=None, ci=None, ax=axes[1][0], alpha=0.2,
                                      palette=palette)
            axes[1][0].legend([], [], frameon=False)
            axes[1][0].set(xlabel='position in track', ylabel='p(left)')

            # plot log likelihood performance
            axes[0][1] = sns.lineplot(data=session_df, x='y_position_binned', y='log_likelihood', hue='target',
                                      estimator='mean', ci=None, ax=axes[0][1], palette=palette)
            axes[0][1].set(title='Log likelihood - non-update trials')
            axes[1][1] = sns.lineplot(data=spatial_map_trials, x='y_position_binned', y='log_likelihood', hue='target',
                                      style='trial', estimator=None, ci=None, ax=axes[1][1], alpha=0.2,
                                      palette=palette)
            axes[1][1].legend([], [], frameon=False)
            axes[1][1].set(xlabel='position in track', ylabel='log_likelihood', ylim=[-2, 0])

            # plot update trials prediction
            axes[0][2] = sns.lineplot(data=session_df, x='y_position_binned', y='update_predict_data', hue='target',
                                      estimator='mean', ci=None, ax=axes[0][2], palette=palette)
            axes[0][2].set(title='LSTM prediction - update trials')
            axes[1][2] = sns.lineplot(data=spatial_map_trials, x='y_position_binned', y='update_predict_data', hue='target',
                                      style='trial', estimator=None, ci=None, ax=axes[1][2], alpha=0.2,
                                      palette=palette)
            axes[1][2].legend([], [], frameon=False)
            axes[1][2].set(xlabel='position in track', ylabel='p(left)')
            for r in [0, 1]:  # for each row
                for a, l in {'0': (0.9, 0.1), '1': (0, -1), '2': (0.9, 0.1)}.items():  # threshold labels for each plot
                    axes[r][int(a)].axhline(l[0], linestyle='dashed', color='k')
                    axes[r][int(a)].axhline(l[1], linestyle='dashed', color='k')
            fig.suptitle(f'LSTM predictions and performance - {name}')

            self.results_io.save_fig(fig=fig, axes=axes, filename=f'prediction_{name}')

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
