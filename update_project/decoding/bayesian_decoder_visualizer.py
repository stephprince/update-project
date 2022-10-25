import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

from pathlib import Path
from matplotlib import ticker
from scipy.stats import sem, pearsonr

from update_project.decoding.bayesian_decoder_aggregator import BayesianDecoderAggregator
from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_limits_from_data, get_color_theme, \
    plot_scatter_with_distributions

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')
rcparams = mpl.rcParams
new_line = '\n'


class BayesianDecoderVisualizer:
    def __init__(self, data, exclusion_criteria=None, params=None, threshold_params=None, overwrite=False):
        self.data = data
        self.data_exists = True
        self.exclusion_criteria = exclusion_criteria
        self.params = params
        self.threshold_params = threshold_params or dict(num_units=[self.exclusion_criteria['units']],
                                                         num_trials=[self.exclusion_criteria['trials']])
        self.colors = get_color_theme()
        self.plot_groups = dict(update_type=[['non_update'], ['switch'], ['stay']],
                                turn_type=[[1], [2], [1, 2]],
                                correct=[[0], [1], [0, 1]])
        self.plot_group_comparisons = dict(update_type=[['non_update'], ['switch'], ['stay']],
                                           turn_type=[[1, 2]],
                                           correct=[[0], [1]])

        self.aggregator = BayesianDecoderAggregator(exclusion_criteria=exclusion_criteria)
        self.aggregator.run_df_aggregation(data, overwrite=True, window=2.5)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def plot(self, group_by=None):
        if self.data_exists:
            group_names = list(group_by.keys())

            # make plots for different parameters, groups, features
            groups = [g if g != 'feature' else 'feature_name' for g in [*group_names, *self.params]]
            for g_name, data in self.aggregator.group_aligned_df.groupby(groups):
                # plot comparisons between plot groups (e.g. correct/incorrect, left/right, update/non-update)
                tags = "_".join([str(n) for n in g_name])
                kwargs = dict(plot_groups=self.plot_group_comparisons, tags=tags)
                self.plot_performance_comparisons(data, tags=tags)
                # self.plot_group_aligned_comparisons(data, **kwargs)
                # self.plot_theta_phase_comparisons(data, **kwargs)

                # make plots for individual plot groups (e.g. correct/incorrect, left/right, update/non-update)
                for plot_types in list(itertools.product(*self.plot_groups.values())):
                    plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
                    title = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
                    kwargs = dict(title=title, plot_groups=plot_group_dict, tags=f'{tags}_{title}')

                    self.plot_group_aligned_data(data, **kwargs)
                    # self.plot_scatter_dists_around_update(data, **kwargs)
                    # self.plot_trial_by_trial_around_update(data, **kwargs)
                    self.plot_phase_modulation_around_update(data, **kwargs)
                    self.plot_theta_phase_histogram(data, **kwargs)

            # plot region interactions
            for name, data in self.aggregator.group_aligned_df.groupby(self.params):
                for plot_types in list(itertools.product(*self.plot_groups.values())):
                    plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
                    tags = "_".join([str(n) for n in name])
                    title = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
                    kwargs = dict(title=title, plot_groups=plot_group_dict, tags=f'{tags}_{title}')
                    self.plot_region_interaction_data(data, **kwargs)

            # plots decoding errors across all groups
            self.plot_all_groups_error(main_group=group_names[0], sub_group=group_names[1])
            self.plot_all_groups_error(main_group=group_names, sub_group='animal')
            self.plot_all_groups_error(main_group='feature', sub_group='num_units', thresh_params=True)
            self.plot_all_groups_error(main_group='feature', sub_group='num_trials', thresh_params=True)

            # plot model metrics (comparing parameters across brain regions and features)
            for name, data in self.aggregator.group_df.groupby(group_names):  # TODO - regions/features in same plot?
                self.plot_tuning_curves(data, name,)
                self.plot_group_confusion_matrices(data, name,)
                self.plot_all_confusion_matrices(data, name)
                self.plot_parameter_comparison(data, name, thresh_params=True)
        else:
            print(f'No data found to plot')

    def _sort_group_confusion_matrices(self, data):
        param_group_data_sorted = []
        for iter_list in itertools.product(*self.threshold_params.values()):
            thresh_mask = pd.concat([data[t] >= i for t, i in zip(self.threshold_params.keys(), iter_list)],
                                    axis=1).all(axis=1)
            subset_data = data[thresh_mask]

            param_group_data = subset_data.groupby(self.params)  # main group is what gets the different plots
            for param_name, param_data in param_group_data:
                sorted_dict = self.aggregator.get_group_confusion_matrices(param_name, param_data)
                param_group_data_sorted.append(dict(**sorted_dict, thresh_values=iter_list))

        # sort the list output
        sorted_data = pd.DataFrame(param_group_data_sorted)
        sorted_data.sort_values('confusion_matrix_sum', ascending=False, inplace=True)

        return sorted_data

    def plot_all_groups_error(self, main_group, sub_group, thresh_params=False):
        print('Plotting group decoding error distributions...')

        # select no threshold data to plot if thresholds indicated, otherwise combine
        if thresh_params:
            df_list = []
            for thresh_val in self.threshold_params[sub_group]:
                new_df = self.aggregator.group_df[self.aggregator.group_df[sub_group] >= thresh_val]
                new_df[f'{sub_group}_thresh'] = thresh_val
                df_list.append(new_df)  # duplicate dfs for each thresh
            df = pd.concat(df_list, axis=0, ignore_index=True)
            sub_group = f'{sub_group}_thresh'  # rename threshold
        else:
            df = self.aggregator.group_df

        group_data = df.groupby(main_group)  # main group is what gets the different plots
        for name, data in group_data:
            nrows = 3  # 1 row for each plot type (cum fract, hist, violin)
            ncols = 3  # 1 column for RMSE dist, 1 for error dist, 1 for confusion_matrix dist
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

            # raw decoding errors
            title = 'Median raw error - all sessions'
            xlabel = 'Decoding error (|true - decoded|)'
            plot_distributions(data, axes=axes, column_name='raw_error', group=sub_group, row_ids=[0, 1, 2],
                               col_ids=[0, 0, 0], xlabel=xlabel, title=title)

            # rmse
            title = 'Root mean square error - all sessions'
            xlabel = 'RMSE'
            plot_distributions(data, axes=axes, column_name='rmse', group=sub_group, row_ids=[0, 1, 2],
                               col_ids=[1, 1, 1], xlabel=xlabel, title=title)

            # confusion matrix sums
            title = 'Confusion matrix sum - all sessions'
            xlabel = 'Probability'
            plot_distributions(data, axes=axes, column_name='confusion_matrix_sum', group=sub_group, row_ids=[0, 1, 2],
                               col_ids=[2, 2, 2], xlabel=xlabel, title=title)

            # wrap up and save plot
            fig.suptitle(f'Decoding error - all sessions - {name}')
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_error', additional_tags=f'{name}_{sub_group}')

    def plot_parameter_comparison(self, data, name, thresh_params=False):
        print('Plotting parameter comparisons..')

        # select no threshold data to plot if thresholds indicated, otherwise combine
        params_to_compare = self.params.copy()
        if thresh_params:
            df_list = []
            for thresh_key, thresh_val in self.threshold_params.items():
                for val in thresh_val:
                    new_df = data[data[thresh_key] >= val]
                    new_df[f'{thresh_key}_thresh'] = val
                    other_key = [k for k in self.threshold_params.keys() if k is not thresh_key]
                    new_df[f'{other_key[0]}_thresh'] = self.threshold_params[other_key[0]][0]
                    df_list.append(new_df)  # duplicate dfs for each thresh
                params_to_compare.append(f'{thresh_key}_thresh')
            df = pd.concat(df_list, axis=0, ignore_index=True)
        else:
            df = data

        nrows = len(list(itertools.combinations(params_to_compare, r=2)))  # 1 row for each parameter combo
        ncols = 3  # 1 column for RMSE dist, 1 for error dist, 1 for confusion_matrix dist
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.5, 10), squeeze=False)
        error_metrics = dict(raw_error=0, rmse=1, confusion_matrix_sum=2)

        # plot heatmaps comparing parameters (1 heatmap/2 parameters)
        medians = df.groupby(params_to_compare).median().reset_index()
        row = 0
        for thresh1, thresh2 in itertools.combinations(params_to_compare, r=2):
            for err, col in error_metrics.items():
                other_keys = [p for p in params_to_compare if p not in [thresh1, thresh2]]
                medians_set_others_const = medians[(medians[other_keys] == medians[other_keys].min()).all(axis=1)]
                df = medians_set_others_const.pivot(thresh1, thresh2, err)
                axes[row][col] = sns.heatmap(df, annot=True, fmt='.2f', ax=axes[row][col], cmap='mako_r',
                                             cbar_kws={'pad': 0.01, 'label': err, 'fraction': 0.046},
                                             annot_kws={'size': 8}, )
                axes[row][col].invert_yaxis()
                if row == 0:
                    axes[row][col].set_title(f'Parameter median {err}')
            row += 1

        # wrap up and save plot
        fig.suptitle(f'Parameter comparison - median all sessions - {name}')
        tags = '_'.join([''.join(str(n)) for n in name])
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_param_comparison', additional_tags=tags)

        # plot heatmaps for all parameters
        nrows = len(params_to_compare)  # 1 row for each parameter combo
        ncols = 3  # 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for row, par in enumerate(params_to_compare):
            for err, col in error_metrics.items():
                other_keys = [p for p in params_to_compare if p not in [par]]
                df = medians.pivot(par, other_keys, err)
                axes[row][col] = sns.heatmap(df, fmt='.2f', ax=axes[row][col], cmap='mako_r',
                                             cbar_kws={'pad': 0.01, 'label': err, 'fraction': 0.046}, )
                axes[row][col].invert_yaxis()
                axes[row][col].set_xticklabels(axes[row][col].get_xmajorticklabels(), fontsize=8)
                if row == 0:
                    axes[row][col].set_title(f'Parameter median {err}')

        fig.suptitle(f'Parameter comparison - median all sessions - {name}')
        tags = '_'.join([''.join(str(n)) for n in name])
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_param_comparison_all', additional_tags=tags)

    def plot_all_confusion_matrices(self, data, name):
        print('Plotting all confusion matrices...')

        for iter_list in itertools.product(*self.threshold_params.values()):
            thresh_mask = pd.concat([data[k] >= v for k, v in zip(self.threshold_params.keys(), iter_list)],
                                    axis=1).all(axis=1)
            subset_data = data[thresh_mask]
            tags = '_'.join([f'{k}_{v}' for k, v in zip(self.threshold_params.keys(), iter_list)])
            param_group_data = subset_data.groupby(self.params)  # main group is what gets the different plots
            for param_name, param_data in param_group_data:
                plot_num = 0
                counter = 0
                ncols, nrows = [6, 3]
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                for _, row in param_data.iterrows():
                    sess_matrix = row['confusion_matrix']
                    sess_key = row['session_id']
                    vmax = 4  # default to 5 vmax probability/chance
                    if row['feature'] in ['choice', 'turn_type']:
                        vmax = 2

                    if (counter % (ncols * nrows) == 0) and (counter != 0):
                        fig.suptitle(f'Confusion matrices - all sessions - {row["results_tags"]}')
                        param_tags = '_'.join([f'{p}_{n}' for p, n in zip(self.params, param_name)])
                        add_tags = f'{tags}_{"_".join(["".join(n) for n in name])}_{param_tags}_plot{plot_num}'
                        self.results_io.save_fig(fig=fig, axes=axes, filename=f'all_confusion_matrices',
                                                 additional_tags=add_tags)

                        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                        counter = 0
                        plot_num += 1
                    else:
                        row_id = int(np.floor(counter / ncols))
                        col_id = counter - row_id * ncols

                        # plot confusion matrix
                        if hasattr(row['bins'], 'astype'):  # if the matrix exists
                            matrix = np.vstack(sess_matrix) * row['encoder_bin_num']  # scale to be probability/chance
                            locations = row['virtual_track'].cue_end_locations.get(row['feature'],
                                                                                   dict())  # don't annotate graph if no locations indicated
                            limits = [np.min(row['bins'].astype(int)), np.max(row['bins'].astype(int))]
                            im = axes[row_id][col_id].imshow(matrix, cmap=self.colors['cmap'], origin='lower',
                                                             # aspect='auto',
                                                             vmin=0, vmax=vmax,
                                                             extent=[limits[0], limits[1], limits[0], limits[1]])

                            # plot annotation lines
                            for key, value in locations.items():
                                axes[row_id][col_id].plot([value, value], [limits[0], limits[1]], linestyle='dashed',
                                                          color=[0, 0, 0, 0.5])
                                axes[row_id][col_id].plot([limits[0], limits[1]], [value, value], linestyle='dashed',
                                                          color=[0, 0, 0, 0.5])

                            # add labels
                            axes[row_id][col_id].text(0.6, 0.2, f'{row["num_trials"]} trials {new_line}'
                                                                f'{row["num_units"]} units',
                                                      transform=axes[row_id][col_id].transAxes, verticalalignment='top',
                                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                            axes[row_id][col_id].set_title(f'{sess_key}')
                            axes[row_id][col_id].set_xlim(limits)
                            axes[row_id][col_id].set_ylim(limits)
                            if row_id == (nrows - 1):
                                axes[row_id][col_id].set_xlabel(f'Actual')
                            if col_id == 0:
                                axes[row_id][col_id].set_ylabel(f'Decoded')
                            if col_id == (ncols - 1):
                                plt.colorbar(im, ax=axes[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                                             label='probability / chance')

                        counter += 1

                # wrap up last plot after loop finished
                fig.suptitle(f'Confusion matrices - all sessions - {row["results_tags"]}')
                param_tags = '_'.join([f'{p}_{n}' for p, n in zip(self.params, param_name)])
                add_tags = f'{tags}_{"_".join(["".join(n) for n in name])}_{param_tags}_plot{plot_num}'
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'all_confusion_matrices',
                                         additional_tags=add_tags)

    def plot_group_confusion_matrices(self, data, name):
        print('Plotting group confusion matrices...')

        # loop through all the parameters and plot one confusion matrix for all sessions for each
        plot_num, counter = (0, 0)
        ncols, nrows = (4, 1)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        sorted_df = self._sort_group_confusion_matrices(data)

        for _, sorted_data in sorted_df.iterrows():
            # plotting info
            new_line = '\n'
            title_params = ''.join([f'{p}: {n} {new_line}' for p, n in zip(self.params, sorted_data['param_values'])])
            title_thresh = ''.join([f'{p}: {n} {new_line}' for p, n in zip(self.threshold_params.keys(),
                                                                           sorted_data['thresh_values'])])
            title = f'{title_params}{title_thresh}'

            # plot the data
            if (counter % (ncols * nrows) == 0) and (counter != 0):
                fig.suptitle(f'Confusion matrices - all parameters - {name}')
                tags = f'{"_".join(["".join(n) for n in name])}_plot{plot_num}'
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_confusion_matrices', additional_tags=tags)

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                counter = 0
                plot_num += 1
            else:
                row_id = int(np.floor(counter / ncols))
                col_id = counter - row_id * ncols

                # plot confusion matrix
                if np.size(sorted_data['confusion_matrix']):
                    matrix = np.vstack(sorted_data['confusion_matrix']) * len(
                        sorted_data['bins']) - 1  # scale to be probability/chance
                    if isinstance(sorted_data['bins'], list):
                        limits = [np.min(np.array(sorted_data['bins'])), np.max(np.array(sorted_data['bins']))]
                    else:
                        limits = [np.round(np.min(sorted_data['bins']), 2), np.round(np.max(sorted_data['bins']), 2)]
                    im = axes[row_id][col_id].imshow(matrix, cmap=self.colors['cmap'], origin='lower',  # aspect='auto',
                                                     vmin=0, vmax=sorted_data['vmax'],
                                                     extent=[limits[0], limits[1], limits[0], limits[1]])

                # plot annotation lines
                for key, value in sorted_data['locations'].items():
                    axes[row_id][col_id].plot([value, value], [limits[0], limits[1]], linestyle='dashed',
                                              color=[0, 0, 0, 0.5])
                    axes[row_id][col_id].plot([limits[0], limits[1]], [value, value], linestyle='dashed',
                                              color=[0, 0, 0, 0.5])

                    # add labels
                    axes[row_id][col_id].text(0.6, 0.2, f'sum: {sorted_data["confusion_matrix_sum"]:.2f}',
                                              transform=axes[row_id][col_id].transAxes, verticalalignment='top',
                                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    axes[row_id][col_id].set_title(f'{title}', fontsize=10)
                    axes[row_id][col_id].set_xlim(limits)
                    axes[row_id][col_id].set_ylim(limits)

                if row_id == (nrows - 1):
                    axes[row_id][col_id].set_xlabel(f'Actual')
                if col_id == 0:
                    axes[row_id][col_id].set_ylabel(f'Decoded')
                plt.colorbar(im, ax=axes[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                             label='probability / chance')

                counter += 1

        # wrap up last plot after loop finished
        fig.suptitle(f'Confusion matrices - all parameters - {name}')
        tags = f'{"_".join(["".join(n) for n in name])}_plot{plot_num}'
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_confusion_matrices', additional_tags=tags)

    def plot_region_interaction_data(self, param_data, title, plot_groups=None, tags=''):
        interaction_data = self.aggregator.calc_region_interactions(param_data, plot_groups)
        if np.size(interaction_data):
            for g_name, g_data in interaction_data.groupby(['a_vs_b']):
                corr_maps = (g_data.groupby(['time_label', 'choice'])['corr_sliding'].apply(lambda x: np.nanmean(np.stack(x), axis=0)))
                times_sliding = g_data.groupby(['time_label', 'choice'])['times_sliding'].mean()
                lags_sliding = g_data.groupby(['time_label', 'choice'])['lags_sliding'].mean()
                g_data_by_time = (g_data
                                  .drop(['corr', 'corr_sliding', 'corr_lags', 'lags_sliding'], axis=1)
                                  .explode(['corr_coeff_sliding', 'times_sliding']))

                ncols, nrows = (len(self.aggregator.align_times), 3)
                fig = plt.figure(figsize=(11, 8.5), constrained_layout=True)
                sfigs = fig.subfigures(nrows, 2, height_ratios=[4, 2, 2], width_ratios=[10, 1])

                axes = sfigs[0][0].subplots(2, ncols, sharey='row', sharex='col')
                caxes = sfigs[0][1].subplots(2, 1)
                for t_name, t_data in corr_maps.groupby('time_label', sort=False):
                    col = np.argwhere(g_data['time_label'].unique() == t_name)[0][0]
                    times = times_sliding[t_name]['initial_stay']  # should be same for switch and stay
                    lags = lags_sliding[t_name]['initial_stay']

                    im1 = axes[0][col].imshow(t_data[t_name]['initial_stay'].T, aspect='auto',
                                              cmap=self.colors['div_cmap'], vmin=-0.015, vmax=0.015,
                                              origin='lower', extent=[times[0], times[-1], lags[0], lags[-1]])
                    axes[0][col].set(xlim=(times[0], times[-1]), ylim=(lags[0], lags[-1]), title=t_name,
                                     ylabel='corr lags')
                    axes[0][col].invert_yaxis()
                    im2 = axes[1][col].imshow(t_data[t_name]['switch'].T, aspect='auto',
                                              cmap=self.colors['div_cmap'], vmin=-0.015, vmax=0.015,
                                              origin='lower', extent=[times[0], times[-1], lags[0], lags[-1]])
                    axes[1][col].set(xlim=(times[0], times[-1]), ylim=(lags[0], lags[-1]),
                                     xlabel='Time around cue', ylabel='corr lags')
                    axes[1][col].invert_yaxis()

                plt.colorbar(im1, cax=caxes[0], label='initial corr', pad=0.01, fraction=0.04,)
                plt.colorbar(im2, cax=caxes[1], label='switch corr', pad=0.01, fraction=0.04,)

                (  # plot corr coeffs over time
                    so.Plot(g_data_by_time, x='times_sliding', y='corr_coeff_sliding', color='choice')
                        .facet(col='time_label')
                        .add(so.Band(), so.Est(errorbar='se'), )
                        .add(so.Line(linewidth=2), so.Agg(), )
                        .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                        .limit(x=(np.min(interaction_data['times'].values[-1]), np.max(interaction_data['times'].values[1])))
                        .theme(rcparams)
                        .layout(engine='constrained')
                        .on(sfigs[1][0])
                        .plot()
                )

                g_data['corr_coeff'][g_data['corr_coeff'].isna()] = 0
                (  # plot initial - switch difference over time (averages with bar)
                    so.Plot(g_data, x='corr_coeff', color='choice')
                        .facet(col='time_label')
                        .add(so.Bars(alpha=0.5, edgealpha=0.5), so.Hist(stat='proportion', binrange=(-1, 1), binwidth=0.1),)
                        .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                        .limit(x=(-1, 1))
                        .label(y='proportion')
                        .theme(rcparams)
                        .layout(engine='constrained')
                        .on(sfigs[2][0])
                        .plot()
                )
                add_lines = [[a.axvline(0, color='k', linestyle='dashed') for a in sf.axes]
                             for sf in [sfigs[2][0], sfigs[1][0], sfigs[0][0]]]
                leg = fig.legends.pop(0)
                sfigs[1][0].legend(leg.legendHandles, [t.get_text() for t in leg.texts], loc='upper right', fontsize='large')

                # save figures
                fig.suptitle(f'{title}_{g_name}')
                self.results_io.save_fig(fig=fig, filename=f'region_interactions', additional_tags=f'{tags}_{g_name}',
                                         tight_layout=False)

    def plot_group_aligned_data(self, param_data, title, plot_groups=None, tags=''):
        # load up data
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(param_data, plot_groups)
        # reaction_data = self.aggregator.calc_movement_reaction_times(param_data, plot_groups)
        aligned_data = self.aggregator.select_group_aligned_data(param_data, plot_groups, ret_df=True)
        bounds = trial_data['bound_values'].unique()
        spaces = getattr(param_data['virtual_track'].values[0], 'edge_spacing', [])
        prob_maps = aligned_data.groupby('time_label').apply(lambda x: np.nanmean(np.stack(x['probability']), axis=0))
        n_bins = np.shape(prob_maps.loc['start_time'])[0]
        prob_lims = np.linspace(aligned_data['feature'].apply(np.nanmin).min(),
                                aligned_data['feature'].apply(np.nanmax).max(), n_bins)
        time_lims = (aligned_data['times'].apply(np.nanmin).min(), aligned_data['times'].apply(np.nanmax).max())

        # make figure
        ncols, nrows = (len(self.aggregator.align_times), 6)  #heatmap, traces, probheatmaps, velocity, error
        fig, axes = plt.subplots(nrows, ncols, sharex='col', sharey='row', figsize=(20, 20), constrained_layout=True)

        for name, data in trial_data.groupby(['time_label'], sort=False):
            col = np.argwhere(trial_data['time_label'].unique() == name)[0][0]
            trial_mat = data.pivot(index=['choice', 'trial_index'], columns='times', values='prob_over_chance')
            switch_mat = trial_mat.query('choice == "switch"').to_numpy()
            stay_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
            times = trial_mat.columns.to_numpy()
            im_times = (times[0] - np.diff(times)[0] / 2, times[-1] + np.diff(times)[0] / 2)

            im_prob = axes[0][col].imshow(prob_maps.loc[name] * n_bins, cmap=self.colors['cmap'], aspect='auto',
                                          origin='lower', vmin=0.6, vmax=2.8,
                                          extent=[im_times[0], im_times[-1], prob_lims[0], prob_lims[-1]])
            axes[0][col].invert_yaxis()
            for b in bounds:
                axes[0][col].axhline(b[0], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)
                axes[0][col].axhline(b[1], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)
            for s in spaces:
                xmin, xmax = 0, 1
                if (times[0], times[-1]) == (0, time_lims[-1]):
                    xmin, xmax = 0.5, 1
                elif (times[0], times[-1]) == (time_lims[0], 0):
                    xmin, xmax = 0, 0.5
                axes[0][col].axhspan(*s, color='#DDDDDD', edgecolor=None, xmin=xmin, xmax=xmax)

            im_goal1 = axes[1][col].imshow(stay_mat, cmap=self.colors['stay_cmap'], aspect='auto', vmin=0, vmax=2.5,
                                           origin='lower', extent=[im_times[0], im_times[-1], 0, np.shape(stay_mat)[0]])
            im_goal2 = axes[2][col].imshow(switch_mat, cmap=self.colors['switch_cmap'], aspect='auto',
                                           vmin=0, vmax=2.5, origin='lower',
                                           extent=[im_times[0], im_times[-1], 0, np.shape(switch_mat)[0]], )

            axes[3][col].plot(times, np.nanmean(switch_mat, axis=0), color=self.colors['switch'], label='switch')
            axes[3][col].fill_between(times, np.nanmean(switch_mat, axis=0) + sem(switch_mat),
                                      np.nanmean(switch_mat, axis=0) - sem(switch_mat), color=self.colors['switch'],
                                      alpha=0.2)
            axes[3][col].plot(times, np.nanmean(stay_mat, axis=0), color=self.colors['initial_stay'], label='initial')
            axes[3][col].fill_between(times, np.nanmean(stay_mat, axis=0) + sem(stay_mat),
                                      np.nanmean(stay_mat, axis=0) - sem(stay_mat), color=self.colors['initial_stay'],
                                      alpha=0.2)
            axes[3][col].axhline(1, linestyle='dashed', color='k', alpha=0.5)
            axes[0][col].set(ylim=(prob_lims[0], prob_lims[-1]), ylabel=param_data['feature_name'].values[0],
                             title=name, xlim=time_lims)
            axes[1][col].set(ylim=(0, np.shape(stay_mat)[0]), ylabel='trials', title=name, xlim=time_lims)
            axes[2][col].set(ylim=(0, np.shape(switch_mat)[0]), ylabel='trials', title=name, xlim=time_lims)
            axes[3][col].set(ylabel='prob / chance', title=name, xlim=time_lims)

        plt.colorbar(im_prob, ax=axes[0][col], label='prob / chance', pad=0.01, fraction=0.046, location='right')
        plt.colorbar(im_goal1, ax=axes[1][col], label='prob / chance', pad=0.01, fraction=0.046, location='right')
        plt.colorbar(im_goal2, ax=axes[2][col], label='prob / chance', pad=0.01, fraction=0.046, location='right')

        for name, data in aligned_data.groupby(['time_label'], sort=False):
            col = np.argwhere(trial_data['time_label'].unique() == name)[0][0]
            veloc = np.stack(data['rotational_velocity'])
            error = np.stack(data['error'])
            axes[4][col].plot(data['times'].values[0], np.nanmean(veloc, axis=0), color=self.colors['control'],
                              label='rotational velocity')
            axes[4][col].fill_between(data['times'].values[0], np.nanmean(veloc, axis=0) + sem(veloc),
                                      np.nanmean(veloc, axis=0) - sem(veloc), color=self.colors['control'], alpha=0.2)
            axes[5][col].plot(data['times'].values[0], np.nanmean(error, axis=0), color=self.colors['error'],
                              label='decoding error')
            axes[5][col].fill_between(data['times'].values[0], np.nanmean(error, axis=0) + sem(error),
                                      np.nanmean(error, axis=0) - sem(error), color=self.colors['error'], alpha=0.2)
            axes[4][col].set(ylabel='velocity', title=name, xlim=time_lims)
            axes[5][col].set(ylabel='decoding error', title=name, xlim=time_lims)

        # medians = reaction_data.groupby('time_label', sort=False)['reaction_time'].median()
        for r in range(nrows):
            axes[r][col].legend(fontsize='large')
            for c in range(ncols):
                # med = medians.get(axes[r][c].title.get_text(), np.nan)
                axes[r][c].axvline(0, color='k', linestyle='dashed', alpha=0.5)
                # axes[r][c].axvline(med, color='purple', linestyle='dashed')

        # save figure
        fig.suptitle(title)
        self.results_io.save_fig(fig=fig, filename=f'group_aligned_data', additional_tags=tags,
                                 tight_layout=False)

    def plot_scatter_dists_around_update(self, param_data, title, plot_groups=None, tags=''):
        ncols, nrows = (len(self.aggregator.align_times) * 2, 4)
        plt_kwargs = dict(palette=sns.diverging_palette(240, 10, s=75, l=60, n=5, center='dark', as_cmap=True),
                          alpha=0.5)  # RdBu diverging palette but goes through black as center vs. white

        trial_data, old_vs_new_data = self.aggregator.calc_trial_by_trial_quant_data(param_data, plot_groups)
        if np.size(trial_data):
            # loop through plotting scatter and kde plots across different levels
            for kind, level in itertools.product(['scatter', 'kde'], ['session_id', 'trial_index', 'animal']):
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(25, 10))
                sfigs = fig.subfigures(int(nrows / 2), int(ncols / 2), hspace=0.1, wspace=0.1)
                for i, label in enumerate(['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made']):
                    data = (old_vs_new_data
                            .query(f"time_label == '{label}'")
                            .groupby([level, 'times_binned'])
                            .mean().reset_index())
                    if np.size(data):
                        plot_scatter_with_distributions(data=data,
                                                        x='prob_sum_initial_stay', y='prob_sum_switch', hue='times_binned',
                                                        fig=sfigs[0][i], title=label, kind=kind, plt_kwargs=plt_kwargs)

                        plot_scatter_with_distributions(data=data, hue='times_binned',
                                                        x='diff_baseline_initial_stay', y='diff_baseline_switch',
                                                        fig=sfigs[1][i], title=label, kind=kind, plt_kwargs=plt_kwargs)
                fig.subplots_adjust(right=0.95, top=0.95)

                # save figure
                fig.suptitle(title)
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'aligned_data_dists_{kind}_{level}',
                                         additional_tags=tags, tight_layout=False)

    def plot_trial_by_trial_around_update(self, param_data, title, plot_groups=None, tags=''):
        # make plots for aligned data (1 row for each plot, 1 col for each align time)
        ncols, nrows = (len(self.aggregator.align_times), 11)
        trial_data, old_vs_new_data = self.aggregator.calc_trial_by_trial_quant_data(param_data, plot_groups)
        if np.size(trial_data):
            fig = plt.figure(figsize=(11, 20), constrained_layout=True)
            sfigs = fig.subfigures(nrows, 2, height_ratios=[0.23, *([0.11] * 10)], width_ratios=[0.99, 0.01])

            # time traces
            (  # plot initial - switch difference over time (averages with bar)
                so.Plot(trial_data, x='times', color='choice')
                    .facet(col='time_label')
                    .pair(y=['prob_sum', 'diff_baseline'])
                    .add(so.Band(), so.Est(errorbar='se'),)
                    .add(so.Line(linewidth=2), so.Agg(),)
                    .scale(color=[self.colors[c] for c in trial_data['choice'].unique()])
                    .theme(rcparams)
                    .layout(engine='constrained')
                    .on(sfigs[0][0])
                    .plot()
            )
            sfigs[0][0].suptitle(title)
            (  # plot diff over time (averages with error bar)
                so.Plot(old_vs_new_data, x='times', y='diff_switch_stay')
                    .facet(col='time_label',
                           order=self.aggregator.align_times)
                    .add(so.Line(linewidth=2), so.Agg())
                    .add(so.Band(), so.Est(errorbar='se'))
                    .scale(color=self.colors['control'])
                    .theme(rcparams)
                    .layout(engine='constrained')
                    .on(sfigs[7][0])
                    .plot()
            )

            # heatmaps of data by session and by trial
            locations = dict(prob_sum=dict(sfig=1, nrows=2, groupby=['time_label', 'choice'], df=trial_data),
                             diff_baseline=dict(sfig=4, nrows=2, groupby=['time_label', 'choice'], df=trial_data),
                             diff_switch_stay=dict(sfig=8, nrows=1, groupby=['time_label'], df=old_vs_new_data))
            levels = ['animal', 'session_id', 'trial_index']
            for kind, level in itertools.product(['prob_sum', 'diff_baseline', 'diff_switch_stay'], levels):
                sfig_ind = locations[kind]['sfig'] + np.argwhere(np.array(levels) == level)[0][0]
                axes = sfigs[sfig_ind][0].subplots(locations[kind]['nrows'], ncols, sharey='row', squeeze=False)
                cax = sfigs[sfig_ind][1].subplots(locations[kind]['nrows'], 1, squeeze=False)
                group_list = locations[kind]['groupby']
                for name, data in locations[kind]['df'].groupby(group_list, sort=False):
                    if len(group_list) > 1:
                        col = np.argwhere(locations[kind]['df'][group_list[0]].unique() == name[0])[0][0]
                        row = np.argwhere(locations[kind]['df'][group_list[1]].unique() == name[1])[0][0]  # which row to plot
                        cmap = self.colors[f'{name[1]}_cmap']
                    else:
                        row, col = (0, np.argwhere(trial_data[group_list[0]].unique() == name)[0][0])
                        cmap = self.colors['cmap']

                    matrix = data.groupby([level, 'times'])[kind].mean().unstack().to_numpy()
                    im = axes[row][col].imshow(matrix, cmap=cmap, vmin=0, vmax=0.4, aspect='auto',
                                               origin='lower', extent=[data['times'].min(), data['times'].max(),
                                                                       0, np.shape(matrix)[0]],)
                    if col == 0:
                        axes[row][col].set_ylabel(level)
                    elif col == ncols - 1:
                        plt.colorbar(im, cax=cax[row][0], label='integral prob')
                sfigs[sfig_ind][0].supylabel(kind)
                sfigs[sfig_ind][0].set_facecolor('none')

            # save figure
            add_lines = [[a.axvline(0, color='k', linestyle='dashed') for a in sf[0].axes] for sf in sfigs]
            sfigs[-1][0].supxlabel(fig.axes[-1].get_xlabel())
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'aligned_data_by_trial',
                                     additional_tags=tags, tight_layout=False)

    def plot_group_aligned_comparisons(self, param_data, plot_groups=None, tags=''):
        print('Plotting group aligned comparisons...')

        feat = param_data['feature_name'].values[0]
        compiled_data = []
        for plot_types in list(itertools.product(*plot_groups.values())):
            plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
            for ind, time_label in enumerate(self.aggregator.align_times):
                filter_dict = dict(time_label=[time_label], **plot_group_dict)
                group_aligned_data = self.aggregator.select_group_aligned_data(param_data, filter_dict)
                if np.size(group_aligned_data) and group_aligned_data is not None:
                    quant_aligned_data = self.aggregator.quantify_aligned_data(param_data, group_aligned_data)
                    compiled_data.append(dict(data=group_aligned_data, quant=quant_aligned_data,
                                              **{k: v[0] for k, v in
                                                 filter_dict.items()}))  # TODO - separate out the compilation

        # compile data for comparisons
        compiled_data_df = pd.DataFrame(compiled_data)
        filter_dict = dict(time_label='t_update', correct=1)
        update_mask = pd.concat([compiled_data_df[k] == v for k, v in filter_dict.items()], axis=1).all(axis=1)
        filter_dict = dict(time_label='t_update', update_type='switch')
        correct_mask = pd.concat([compiled_data_df[k] == v for k, v in filter_dict.items()], axis=1).all(axis=1)
        compare_df = dict(update_type=dict(data=compiled_data_df[update_mask],
                                           comparison=['switch', 'stay']),
                          correct=dict(data=compiled_data_df[correct_mask],
                                       comparison=[1, 0]))

        # plot the data and accompanying stats
        for comp, data_dict in compare_df.items():
            stats_data, stats_plot_data = self.aggregator.get_aligned_stats(comp, data_dict, quant_aligned_data, tags)
            self.plot_aligned_comparison(comp, data_dict, stats_plot_data, feat, filter_dict, tags)
            self.plot_group_aligned_stats(stats_data, tags=tags)

    def plot_aligned_comparison(self, comp, data_dict, stats_data, feat, filter_dict, tags):
        ncols, nrows = (2, 6)  # one col for each comparison, 1 col for difference between two
        fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row')
        comparison_data = []
        for ind, v in enumerate(data_dict['comparison']):
            # plot comparison data
            title = f'{v}_{"".join([f"{k}{v}" for k, v in filter_dict.items()])}'
            data = data_dict['data'][data_dict['data'][comp] == v]['data'].values[0]
            quant = data_dict['data'][data_dict['data'][comp] == v]['quant'].values[0]
            bounds = [v['bound_values'] for v in quant.values()]
            self.plot_1d_around_update(data, quant, title, feat, axes, row_id=np.arange(nrows), col_id=[ind] * nrows,
                                       feature_name=feat, prob_map_axis=0, bounds=bounds)

            # compile comparison data for differential
            all_data = dict(data['stats']['probability'], bound='all', comparison=comp, group=v)
            initial_data = dict(**quant['left']['stats']['prob_sum'], bound='initial', comparison=comp, group=v)
            new_data = dict(**quant['right']['stats']['prob_sum'], bound='new', comparison=comp, group=v)
            comparison_data.extend([all_data, initial_data, new_data])
        data_df = pd.DataFrame(comparison_data)
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'compare_{comp}_aligned_data', additional_tags=tags)

        # calculate difference
        ncols, nrows = (1, 2)  # one col for each comparison, 1 col for difference between two
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, sharey='row')
        groups = data_df.groupby('group')
        g_names = data_dict['comparison']
        diff_mean = groups.get_group(g_names[0])['mean'].to_numpy() - groups.get_group(g_names[1])['mean'].to_numpy()
        diff_labels = groups.get_group(g_names[0]).bound.values
        diff_data = {k: v for k, v in zip(diff_labels, diff_mean)}
        times = data['times']
        n_position_bins = np.shape(diff_data['all'])[0]
        feat_bins = np.linspace(np.nanmin(data['decoding']), np.nanmax(data['decoding']), n_position_bins)

        im = axes[0][0].imshow(diff_data['all'], cmap='Greys', origin='lower', aspect='auto',
                               vmin=0.25 * np.nanmin(diff_data['all']), vmax=0.75 * np.nanmax(diff_data['all']),
                               extent=[times[0], times[-1], feat_bins[0], feat_bins[-1]])
        axes[0][0].axvline(0, linestyle='dashed', color='k', zorder=0)
        axes[0][0].invert_yaxis()
        axes[0][0].set(xlabel='Time around update (s)', ylabel=f'diff in probability', xlim=[times[0], times[-1]],
                       ylim=[feat_bins[0], feat_bins[-1]])
        axes[0][0].set_title(f'{title} trials - probability density - {g_names[0]} - {g_names[1]}', fontsize=14)
        plt.colorbar(im, ax=axes[0][0], label='probability density', pad=0.01, location='right', fraction=0.046)
        for b in bounds:
            axes[0][0].axhline(b[0], linestyle='dashed', linewidth=0.5, zorder=0)
            axes[0][0].axhline(b[1], linestyle='dashed', linewidth=0.5, zorder=0)

        # add labels for bound_values, threshold
        axes[1][0].plot(times, diff_data['initial'], color=self.colors['stay'], label='initial_stay')
        axes[1][0].plot(times, diff_data['new'], color=self.colors['switch'], label='switch')
        axes[1][0].axvline(0, linestyle='dashed', color='k', zorder=0)
        axes[1][0].set(xlabel='Time around update (s)', ylabel=f'diff in probability', xlim=[times[0], times[-1]])
        axes[1][0].set_title(f'{title} trials - probability density - {g_names[0]} - {g_names[1]}', fontsize=14)
        axes[1][0].legend(loc='upper left')

        # add significance stars
        bound_plot_info = dict(initial=dict(height=10, color=self.colors['stay']),
                               new=dict(height=5, color=self.colors['switch']))
        for key, val in bound_plot_info.items():
            stars_height = [axes[1][0].get_ylim()[1] / (val['height'])] * len(stats_data['sig'][key])
            blanks_height = [axes[1][0].get_ylim()[1] / (val['height'])] * len(stats_data['ns'][key])
            axes[1][0].plot(stats_data['ns'][key], blanks_height, marker="o", linestyle='', markerfacecolor='k',
                            markersize=5, label='n.s.')
            axes[1][0].plot(stats_data['sig'][key], stars_height, marker="*", linestyle='',
                            markerfacecolor=val['color'],
                            markersize=10, label=f'{key} sig.')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'compare_{comp}_aligned_data_diff', additional_tags=tags)

    def plot_group_aligned_stats(self, data_for_stats, tags=''):
        # grab data from the first second after the update occurs
        prob_sum_df = pd.DataFrame(data_for_stats)
        prob_sum_df = prob_sum_df.explode('prob_sum').reset_index(drop=True)
        bins_to_grab = np.floor([len(prob_sum_df['prob_sum'].values[0]) / 2,
                                 5 * len(prob_sum_df['prob_sum'].values[0]) / 8]).astype(int)
        prob_sum_df['data'] = prob_sum_df['prob_sum'].apply(lambda x: np.nansum(x[bins_to_grab[0]:bins_to_grab[1]]))
        temp_df = prob_sum_df[['bound', 'comparison', 'group', 'data']].explode('data').reset_index(drop=True)
        df = pd.DataFrame(temp_df.to_dict())  # fix weird object error for violin plots

        # plot figure
        nrows = 3  # 1 row for each plot type (cum fract, hist, violin)
        ncols = 4  # 1 column for switch vs. stay, correct vs. incorrect switch * 3 for left, right
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
        count = 0
        for n, group in df.groupby(['comparison', 'bound']):
            plot_distributions(group, axes=axes, column_name='data', group='group', row_ids=[0, 1, 2],
                               col_ids=[count] * 3, xlabel='probability sum', title=n, stripplot=False)
            count += 1

        # get stats
        for name, group in df.groupby(['comparison', 'bound']):
            data_to_compare = {'_'.join((*name, str(v))): group[group['group'] == v]['data'].values for v in
                               list(group['group'].unique())}
            self.results_io.export_statistics(data_to_compare, f'aligned_data_{"_".join(name)}_stats_{tags}')

    def plot_tuning_curves(self, data, name):
        print('Plotting tuning curves...')

        feat = data['feature'].values[0]
        locations = data.virtual_track.values[0].cue_end_locations.get(feat, dict())
        tuning_curve_params = [p for p in self.params if p not in ['decoder_bins']]
        data_const_decoding = data[data['decoder_bins'] == data['decoder_bins'].values[0]]
        param_group_data = data_const_decoding.groupby(tuning_curve_params)
        plot_num, counter = (0, 0)
        nrows, ncols = (1, 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for param_name, param_data in param_group_data:
            group_tuning_curve_df = self.aggregator.get_tuning_data(param_data)
            tuning_curve_mat = np.stack(group_tuning_curve_df['tuning_curve'].values)
            tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
            tuning_curve_bins = group_tuning_curve_df['bins'].values[0]
            sort_index = np.argsort(np.argmax(tuning_curve_scaled, axis=1))

            # plotting info
            param_name = param_name if isinstance(param_name, list) else [param_name]
            new_line = '\n'
            tags = f'{"_".join(["".join(n) for n in name])}' \
                   f'{"_".join([f"{p}_{n}" for p, n in zip(tuning_curve_params, param_name)])}'
            title = ''.join([f'{p}: {n} {new_line}' for p, n in zip(tuning_curve_params, param_name)])

            # plot the data
            if (counter % (ncols * nrows) == 0) and (counter != 0):
                fig.suptitle(f'Feature tuning curves - {name}')
                tags = f'{tags}_plot{plot_num}'
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves', additional_tags=tags)

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                counter = 0
                plot_num += 1
            else:
                row_id = int(np.floor(counter / ncols))
                col_id = counter - row_id * ncols

                # plot heatmaps
                y_limits = [0, np.shape(tuning_curve_scaled)[0]]
                x_limits = [np.round(np.min(tuning_curve_bins), 2), np.round(np.max(tuning_curve_bins), 2)]
                im = axes[row_id][col_id].imshow(tuning_curve_scaled[sort_index, :], cmap=self.colors['cmap'],
                                                 origin='lower',
                                                 vmin=0.1,
                                                 aspect='auto',
                                                 vmax=0.9, extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]])

                # plot annotation lines
                for key, value in locations.items():
                    axes[row_id][col_id].plot([value, value], [y_limits[0], y_limits[1]], linestyle='dashed',
                                              color=[1, 1, 1, 0.5])

                # add limits
                axes[row_id][col_id].set_title(f'{title}', fontsize=10)
                axes[row_id][col_id].set_xlim(x_limits)
                axes[row_id][col_id].set_ylim(y_limits)
                axes[row_id][col_id].set_ylim(y_limits)
                if row_id == (nrows - 1):
                    axes[row_id][col_id].set_xlabel(f'{feat}')
                if col_id == 0:
                    axes[row_id][col_id].set_ylabel(f'Units')
                plt.colorbar(im, ax=axes[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                             label='Normalized firing rate')
                counter += 1

        # save figure
        fig.suptitle(f'Feature tuning curves - {name}')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves', additional_tags=tags)

    def plot_phase_modulation_around_update(self, param_data, title, plot_groups=None, tags=''):
        # make plots for aligned data (1 row for heatmap, 1 row for modulation index, 1 col for each align time)

        print('Plotting phase modulation around update...')
        feat = param_data['feature_name'].values[0]
        ncols, nrows = (len(self.aggregator.align_times), 4)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey='row', sharex='col')
        for ind, time_label in enumerate(self.aggregator.align_times):
            filter_dict = dict(time_label=[time_label], **plot_groups)
            theta_phase_data = self.aggregator.calc_theta_phase_data(param_data, filter_dict, time_bins=21)
            scaling_dict = dict(switch=(0.1, 0.3), initial_stay=(0.1, 0.3), home=(0.7, 0.9),
                                theta_amplitude=(-100, 100))
            if np.size(theta_phase_data):
                theta_phase_data_full = theta_phase_data[theta_phase_data['bin_name'] == 'full']
                for row_ind, loc in enumerate(['switch', 'initial_stay', 'home', 'theta_amplitude']):
                    cmap = self.colors.get(f'{loc}_cmap', self.colors['home_cmap'])
                    mod_map = theta_phase_data_full.pivot(columns='time_mid', index='phase_mid', values=loc)
                    phases = mod_map.index.to_numpy() / np.pi
                    times = mod_map.columns.to_numpy()
                    im = axes[row_ind][ind].imshow(mod_map, cmap=cmap, origin='lower', aspect='auto',
                                                   vmin=scaling_dict[loc][0], vmax=scaling_dict[loc][1],
                                                   extent=[times[0], times[-1], phases[0], phases[-1]])
                    axes[row_ind][ind].yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
                    axes[row_ind][ind].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
                    axes[row_ind][ind].axvline(0, linestyle='dashed', color='k', zorder=0)
                    if loc == 'theta_amplitude':
                        plt.colorbar(im, ax=axes[row_ind][ind], label='theta amplitude', pad=0.01,
                                     location='right',
                                     fraction=0.046)
                    else:
                        plt.colorbar(im, ax=axes[row_ind][ind], label=f'{loc} prob density', pad=0.01, location='right',
                                     fraction=0.046)
                    if row_ind == nrows - 1:
                        axes[row_ind][ind].set(xlabel='Time around update')
                    if row_ind == 0:
                        axes[row_ind][ind].set_title(time_label)

        # save figure
        fig.suptitle(f'{feat}_{title}')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'theta_mod_around_update',
                                 additional_tags=tags)

    def plot_theta_phase_histogram(self, param_data, title, plot_groups=None, tags=''):
        print('Plotting theta phase histograms...')

        # make plots for aligned data (1 row for hists, half-cycle data, 1 col for each align time)
        feat = param_data['feature_name'].values[0]
        ncols, nrows = (len(self.aggregator.align_times), 6)
        fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row', sharex='col')
        for ind, time_label in enumerate(self.aggregator.align_times):
            filter_dict = dict(time_label=[time_label], **plot_groups)
            theta_phase_data = self.aggregator.calc_theta_phase_data(param_data, filter_dict)
            rows = dict(full=0, half=3, theta_amplitude=0, initial_stay=1, switch=1, home=2)
            for g_name, group in theta_phase_data.groupby(['bin_name', 'times']):
                row_ind = rows[g_name[0]]  # full or half
                if g_name[1] == 'post' and time_label == 't_choice_made':
                    pass  # don't plot post-choice made data bc really messy
                else:
                    for loc in ['switch', 'initial_stay', 'home', 'theta_amplitude']:
                        lstyle = ['dashed' if g_name[1] == 'pre' else 'solid'][0]
                        color = [self.colors[loc] if loc in ['switch', 'initial_stay'] else 'k'][0]
                        axes[rows[loc] + row_ind][ind].plot(group['phase_mid'] / np.pi, group[f'{loc}'], color=color,
                                                            linestyle=lstyle,
                                                            label=f'{loc}_{g_name[1]}')
                        axes[rows[loc] + row_ind][ind].fill_between(group['phase_mid'] / np.pi,
                                                                    group[f'{loc}_err_lower'],
                                                                    group[f'{loc}_err_upper'], alpha=0.2, color=color, )
                        axes[rows[loc] + row_ind][ind].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
                        axes[rows[loc] + row_ind][ind].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
                        axes[rows[loc] + row_ind][ind].relim()
                        if ind == 0 and rows[loc] + row_ind != 0:
                            axes[rows[loc] + row_ind][ind].legend()
                            axes[rows[loc] + row_ind][ind].set_ylabel('prob_density')
                        elif ind == 0 and loc == 'theta_amplitude':
                            axes[rows[loc] + row_ind][ind].legend()
                            axes[rows[loc] + row_ind][ind].set_ylabel('theta_amplitude')

                    if rows[loc] + row_ind == nrows - 1:
                        axes[rows[loc] + row_ind][ind].set(xlabel='theta phase')
                    if rows[loc] + row_ind == 0:
                        axes[rows[loc] + row_ind][ind].set_title(time_label)

        # save figure
        fig.suptitle(f'{feat}_{title}')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'theta_phase_hist', additional_tags=tags)

    def plot_theta_phase_comparisons(self, param_data, plot_groups=None, tags=''):
        print('Plotting theta phase comparisons...')

        compiled_data = []
        for plot_types in list(itertools.product(*plot_groups.values())):
            plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
            filter_dict = dict(time_label=['t_update'], **plot_group_dict)
            theta_phase_data = self.aggregator.calc_theta_phase_data(param_data, filter_dict)
            compiled_data.append(dict(data=theta_phase_data, **{k: v[0] for k, v in filter_dict.items()}))

        # compile data for comparisons
        compiled_data_df = pd.DataFrame(compiled_data)
        update_mask = pd.concat([compiled_data_df[k] == v for k, v in dict(correct=1).items()], axis=1).all(axis=1)
        correct_mask = pd.concat([compiled_data_df[k] == v for k, v in dict(update_type='switch').items()], axis=1).all(
            axis=1)
        compare_df = dict(update_type=dict(data=compiled_data_df[update_mask], comparison=['switch', 'stay']),
                          correct=dict(data=compiled_data_df[correct_mask], comparison=[1, 0]))

        # plot the data and accompanying stats
        for comp, data_dict in compare_df.items():
            # get difference between groups
            comparison_data = []
            for ind, v in enumerate(data_dict['comparison']):
                d = data_dict['data'][data_dict['data'][comp] == v]['data'].values[0]
                d = d[d['bin_name'] == 'full']
                d['comparison'] = comp
                d['group'] = v
                comparison_data.append(d)
            comparison_df = pd.concat(comparison_data, axis=0)

            # calculate difference
            ncols, nrows = (6, 2)  # cols for pre/post, g12 g_diff, row for each value
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, sharey='row')

            for t_ind, time in enumerate(['pre', 'post']):
                groups = comparison_df[comparison_df['times'] == time].groupby('group')
                g_names = data_dict['comparison']
                for loc in ['initial_stay', 'switch', 'home']:
                    diff_mean = np.vstack(groups.get_group(g_names[0])[loc].to_numpy()) - \
                                np.vstack(groups.get_group(g_names[1])[loc].to_numpy())
                    diff_labels = groups.get_group(g_names[0])['times'].values
                    diff_data = {k: v for k, v in zip(diff_labels, diff_mean)}

                    lstyle = ['dashed' if time == 'pre' else 'solid'][0]
                    color = [self.colors[loc] if loc in ['switch', 'initial_stay'] else 'k'][0]
                    row_ind = [0 if loc in ['initial_stay', 'switch'] else 1][0]
                    axes[row_ind][0 + t_ind].plot(np.array(comparison_df['phase_mid'][0]) / np.pi,
                                                  groups.get_group(g_names[0])[loc].to_numpy()[0], color=color,
                                                  linestyle=lstyle,
                                                  label=f'{loc}')
                    axes[row_ind][2 + t_ind].plot(np.array(comparison_df['phase_mid'][0]) / np.pi,
                                                  groups.get_group(g_names[1])[loc].to_numpy()[0], color=color,
                                                  linestyle=lstyle,
                                                  label=f'{loc} {g_names[1]}')
                    axes[row_ind][4 + t_ind].plot(np.array(comparison_df['phase_mid'][0]) / np.pi, diff_data[time],
                                                  color=color,
                                                  linestyle=lstyle, label=f'{loc}')
                    axes[row_ind][0 + t_ind].set(title=f'{g_names[0]} - {time}', ylabel='mean probability')
                    axes[row_ind][2 + t_ind].set(title=f'{g_names[1]} - {time}')
                    axes[row_ind][4 + t_ind].set(title=f'{g_names[0]} - {g_names[1]} - {time}')
                    axes[row_ind][4 + t_ind].set(ylabel='diff probability')

            ax_list = axes.flat
            for ax in ax_list:
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
                ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
                ax.set(xlabel='theta phase')
                ax.legend()
            fig.suptitle(f'Theta phase histogram comparison - {g_names[0]} - {g_names[1]}')
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'compare_{comp}_theta_phase_hist',
                                     additional_tags=tags)

    def plot_performance_comparisons(self, param_data, plot_groups=None, tags=''):
        # load up data
        plot_groups = dict(update_type=['non_update', 'switch', 'stay'],
                           turn_type=[1, 2],
                           correct=[0, 1],
                           time_label=['t_update'])  # use as default if no value given
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(param_data, plot_groups=plot_groups, n_time_bins=6)
        trial_data.dropna(subset='times_binned', inplace=True)

        # make figure
        time_bins = trial_data['times_binned'].unique()
        for t_ind, t in enumerate(time_bins):
            time_included = list(time_bins[:(t_ind + 1)])  # get all data up to this point in time
            data_subset = trial_data.query(f'times_binned.isin({time_included})')
            fig = plt.figure(figsize=(8.5, 11), constrained_layout=True)
            sfigs = fig.subfigures(3, 3, hspace=0.1, wspace=0.1)
            for g_name, g_data in data_subset.groupby('update_type', sort=False):
                # bin trials for percent correct calculations
                col = np.argwhere(trial_data['update_type'].unique() == g_name)[0][0]
                trial_block = 40
                bins = np.hstack([g_data['index'].unique()[::trial_block], g_data['index'].unique()[-1]])
                g_data['trials_binned'] = pd.cut(g_data['index'], bins=bins, include_lowest=True, labels=False, duplicates='drop')

                for row, level in enumerate(['trials_binned', 'session_id', 'animal']):
                    plot_data = g_data.groupby([level, 'choice', 'region']).mean().reset_index()

                    (  # plot scatters with estimates
                        so.Plot(plot_data, x='correct', color='choice')
                            .pair(y=['diff_baseline', 'prob_over_chance'])
                            .add(so.Dot(alpha=0.3))
                            .add(so.Line(), so.PolyFit(order=1),)
                            .share(y='row')
                            .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                            .theme(rcparams)
                            .layout(engine='constrained')
                            .limit(x=(0, 1))
                            .label(title=g_name, x='proportion correct')
                            .on(sfigs[row][col])
                            .plot()
                        )
                    for ax in sfigs[row][col].axes:
                        metric = ax.get_ylabel().replace(' ', '_')
                        stats = plot_data.groupby('choice').apply(lambda x: pearsonr(x['correct'], x[metric])).to_dict()
                        text = ''.join([f'{k}: {v[0]:.2f}, {v[1]:.2g} {new_line}' for k, v in stats.items()])
                        ax.text(0.05, 0.65, text, transform=ax.transAxes)
                    sfigs[row][1].suptitle(f'Performance by {level}')

            # save figure
            times = f'up_to_{np.max(time_included)}s'
            self.results_io.save_fig(fig=fig, filename=f'performance_comparisons', additional_tags=f'{tags}_{times}',
                                     tight_layout=False)

    def plot_1d_around_update(self, data_around_update, quantification_data, title, label, axes,
                              row_id, col_id, feature_name=None, prob_map_axis=0, bounds=[]):
        # get color mappings
        prob_name = 'prob_over_chance'  # was prob sum
        data_split = dict(initial_stay=dict(data=quantification_data['left'],
                                            color=self.colors['stay'],
                                            cmap=self.colors['stay_cmap']),
                          switch=dict(data=quantification_data['right'],
                                      color=self.colors['switch'],
                                      cmap=self.colors['switch_cmap']))

        feature_name = feature_name or self.data.feature_names[0]
        error_bars = ''
        if feature_name in ['x_position', 'view_angle', 'choice', 'turn_type', 'dynamic_choice',
                            'cue_bias']:  # divergent color maps for div data
            cmap_pos = self.colors['left_right_cmap_div']
            balanced = True
        elif feature_name in ['y_position']:  # sequential color map for seq data
            cmap_pos = 'Greys'
            balanced = False
        if np.size(data_around_update['probability']):  # skip if there is no data
            prob_map = np.nanmean(data_around_update['probability'], axis=prob_map_axis)
            if prob_map_axis == 1:
                prob_map = prob_map.T

            stats = data_around_update['stats']
            limits = get_limits_from_data([data_around_update['feature']], balanced=balanced)
            err_limits = get_limits_from_data(
                [(v['data']['stats'][prob_name][f'{error_bars}lower'],
                  v['data']['stats'][prob_name][f'{error_bars}upper'])
                 for k, v in data_split.items()], balanced=False)
            err_limits[0] = 0
            all_limits_balanced = get_limits_from_data([v['data'][prob_name] for v in data_split.values()])
            times = data_around_update['times']
            time_tick_values = times.astype(int)
            n_position_bins = np.shape(prob_map)[0]
            data_values = np.linspace(np.nanmin(data_around_update['decoding']),
                                      np.nanmax(data_around_update['decoding']),
                                      n_position_bins)

            pos_values_after_update = np.nansum(
                data_around_update['feature'][int(len(time_tick_values) / 2):int(len(time_tick_values) / 2) + 10],
                axis=0)
            sort_index = np.argsort(pos_values_after_update)
            # TODO - add rotational velocity onset calculations and replace feature plotting with these

            im = axes[row_id[0]][col_id[0]].imshow(prob_map*n_position_bins, cmap=self.colors['cmap'], origin='lower',
                                                   aspect='auto',
                                                   # vmin=0.25 * np.nanmin(prob_map), vmax=0.75 * np.nanmax(prob_map),
                                                   vmin=0.6, vmax=2.8,  # other options are (0.01, 0.25 to 0.1()
                                                   extent=[times[0], times[-1], data_values[0], data_values[-1]])
            axes[row_id[0]][col_id[0]].plot([0, 0], [data_values[0], data_values[-1]], linestyle='dashed',
                                            color=[0, 0, 0, 0.5])
            axes[row_id[0]][col_id[0]].invert_yaxis()
            axes[row_id[0]][col_id[0]].set(xlabel='Time around update (s)', ylabel=f'{label}',
                                           xlim=[times[0], times[-1]],
                                           ylim=[data_values[0], data_values[-1]])
            axes[row_id[0]][col_id[0]].set_title(f'{title} trials - probability density - {label}', fontsize=14)
            plt.colorbar(im, ax=axes[row_id[0]][col_id[0]], label=prob_name, pad=0.01, location='right',
                         fraction=0.046)
            for b in bounds:
                axes[row_id[0]][col_id[0]].plot([times[0], times[-1]], [b[0], b[0]], linestyle='dashed',
                                                color=[0, 0, 0, 0.5],
                                                linewidth=0.5)
                axes[row_id[0]][col_id[0]].plot([times[0], times[-1]], [b[1], b[1]], linestyle='dashed',
                                                color=[0, 0, 0, 0.5],
                                                linewidth=0.5)

            axes[row_id[1]][col_id[1]].plot(times, stats['error']['mean'], color=self.colors['error'],
                                            label='|True - decoded|')
            axes[row_id[1]][col_id[1]].fill_between(times, stats['error']['lower'], stats['error']['upper'], alpha=0.2,
                                                    color=self.colors['error'],
                                                    label='95% CI')
            axes[row_id[1]][col_id[1]].plot([0, 0], [0, np.max(stats['error']['upper'])], linestyle='dashed', color='k',
                                            alpha=0.25)
            axes[row_id[1]][col_id[1]].set(xlim=[times[0], times[-1]], ylim=[0, np.nanmax(stats['error']['upper'])],
                                           xlabel='Time around update(s)', ylabel=label)
            axes[row_id[1]][col_id[1]].set_title(f'{title} trials - decoding error {label}', fontsize=14)
            axes[row_id[1]][col_id[1]].legend(loc='upper left')

            im = axes[row_id[2]][col_id[2]].imshow(data_around_update['feature'][:, sort_index], cmap=cmap_pos,
                                                   origin='lower',
                                                   vmin=data_values[0], vmax=data_values[-1],
                                                   extent=[times[0], times[-1], 0, len(sort_index)], aspect='auto')
            axes[row_id[2]][col_id[2]].plot([0, 0], [0, len(sort_index)], linestyle='dashed',
                                            color=[0, 0, 0, 0.5])
            axes[row_id[2]][col_id[2]].set(ylabel='Trials')
            axes[row_id[2]][col_id[2]].set_title(f'{title} trials - true {label}', fontsize=14)
            plt.colorbar(im, ax=axes[row_id[2]][col_id[2]], label=f'{label} fraction', pad=0.01, location='right',
                         fraction=0.046)

            # add labels for bound_values, threshold
            bound_ind = 0
            for key, value in data_split.items():
                stats = value['data']['stats']

                # line plots
                axes[row_id[3]][col_id[3]].plot(times, stats[prob_name]['mean'], color=value['color'], label=key)
                axes[row_id[3]][col_id[3]].fill_between(times, stats[prob_name][f'{error_bars}lower'],
                                                        stats[prob_name][f'{error_bars}upper'],
                                                        alpha=0.2, color=value['color'], label='95% CI')
                axes[row_id[3]][col_id[3]].axvline(0, linestyle='dashed', color='k', alpha=0.25)
                axes[row_id[3]][col_id[3]].axhline(1, linestyle='dashed', color='k', alpha=0.25)
                axes[row_id[3]][col_id[3]].set(xlim=[times[0], times[-1]], ylim=err_limits, ylabel=key)
                axes[row_id[3]][col_id[3]].legend(loc='upper left')
                axes[row_id[3]][col_id[3]].set_title(f'{title} trials - {label} - {prob_name}', fontsize=14)

                # heat maps by trial
                im = axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].imshow(value['data'][prob_name],
                                                                               cmap=value['cmap'], origin='lower',
                                                                               aspect='auto',
                                                                               extent=[times[0], times[-1], 0,
                                                                                       len(value['data'][prob_name])],
                                                                               vmin=0,
                                                                               vmax=2.75)
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].invert_yaxis()
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].plot([0, 0], [0, len(value['data'][prob_name])],
                                                                        linestyle='dashed', color=[0, 0, 0, 0.5])
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].set(xlim=[times[0], times[-1]], ylabel=f'trials')
                plt.colorbar(im, ax=axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]], label=prob_name, pad=0.01,
                             location='right',
                             fraction=0.046)

                if (4 + bound_ind) == len(row_id):
                    axes[row_id[4 + bound_ind][col_id[4 + bound_ind]]].set(xlabel='Time (s)')

                bound_ind = + 1

    def plot_2d_around_update(self, data_around_update, time_bin, times, title, color, axes, ax_dict):
        stats = data_around_update['stats']
        prob_map = np.nanmean(data_around_update['probability'], axis=0)
        if title == 'switch':
            correct_multiplier = -1
        elif title == 'stay':
            correct_multiplier = 1
        xlims = [-30, 30]
        ylims = [5, 285]
        track_bounds_xs, track_bounds_ys = self.data.virtual_track.get_track_boundaries()

        if np.size(data_around_update['probability']):  # skip if there is no data
            positions_y = stats['feature']['mean'][:time_bin + 1]
            positions_x = stats['feature']['mean'][:time_bin + 1]

            axes[ax_dict[0]].plot(positions_x, positions_y, color='k', label='True position')
            axes[ax_dict[0]].plot(positions_x[-1], positions_y[-1], color='k', marker='o', markersize='10',
                                  label='Current true position')
            axes[ax_dict[0]].plot(stats['decoding_x']['mean'][:time_bin + 1],
                                  stats['decoding_y']['mean'][:time_bin + 1],
                                  color=color, label='Decoded position')
            axes[ax_dict[0]].plot(stats['decoding_x']['mean'][time_bin], stats['decoding_y']['mean'][time_bin],
                                  color=color,
                                  marker='o', markersize='10', label='Current decoded position')
            axes[ax_dict[0]].plot(track_bounds_xs, track_bounds_ys, color='black')
            axes[ax_dict[0]].set(xlim=[-25, 25], ylim=ylims, xlabel='X position', ylabel='Y position')
            axes[ax_dict[0]].legend(loc='lower left')
            axes[ax_dict[0]].text(0.65, 0.1, f'Time to update: {np.round(times[time_bin], 2):.2f} s',
                                  transform=axes[ax_dict[0]].transAxes, fontsize=14,
                                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.25))
            axes[ax_dict[0]].annotate('update cue on here', (2, stats['position_y']['mean'][int(len(times) / 2)]),
                                      xycoords='data', xytext=(5, stats['position_y']['mean'][int(len(times) / 2)]),
                                      textcoords='data', va='center', arrowprops=dict(arrowstyle='->'))
            axes[ax_dict[0]].annotate('correct side', (18 * correct_multiplier, 250), textcoords='data', va='center')
            axes[ax_dict[0]].set_title(f'{title} trials - decoded vs. true position', fontsize=14)

            im = axes[ax_dict[1]].imshow(prob_map[:, :, time_bin], cmap=self.colors['cmap'], origin='lower',
                                         aspect='auto',
                                         vmin=0, vmax=0.6 * np.nanmax(prob_map),
                                         extent=[xlims[0], xlims[1], ylims[0], ylims[1]])
            axes[ax_dict[1]].plot(positions_x, positions_y, color='k', label='True position')
            axes[ax_dict[1]].plot(positions_x[-1], positions_y[-1], color='k', marker='o', markersize='10',
                                  label='Current true position')
            axes[ax_dict[1]].plot(track_bounds_xs, track_bounds_ys, color='black')
            axes[ax_dict[1]].annotate('update cue on here', (2, stats['position_y']['mean'][int(len(times) / 2)]),
                                      xycoords='data', xytext=(5, stats['position_y']['mean'][int(len(times) / 2)]),
                                      textcoords='data', va='center', arrowprops=dict(arrowstyle='->'))
            axes[ax_dict[1]].annotate('correct side', (18 * correct_multiplier, 250), textcoords='data', va='center')
            axes[ax_dict[1]].set(xlim=[-25, 25], ylim=ylims, xlabel='X position',
                                 ylabel='Y position')  # cutoff some bc lo
            axes[ax_dict[1]].set_title(f'{title} trials - probability density', fontsize=14)
            plt.colorbar(im, ax=axes[ax_dict[1]], label='Probability density', pad=0.04, location='right',
                         fraction=0.046)
