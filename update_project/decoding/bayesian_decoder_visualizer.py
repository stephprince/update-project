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
from statannotations.Annotator import Annotator

from update_project.decoding.bayesian_decoder_aggregator import BayesianDecoderAggregator
from update_project.general.results_io import ResultsIO
from update_project.general.plots import plot_distributions, plot_scatter_with_distributions, rainbow_text, clean_box_plot
from update_project.statistics.statistics import Stats
from update_project.base_visualization_class import BaseVisualizationClass


class BayesianDecoderVisualizer(BaseVisualizationClass):
    def __init__(self, data, exclusion_criteria=None, params=None, threshold_params=None):
        super().__init__(data)
        self.exclusion_criteria = exclusion_criteria or dict(units=20, trials=50)
        self.params = params
        self.threshold_params = threshold_params or dict(num_units=[self.exclusion_criteria['units']],
                                                         num_trials=[self.exclusion_criteria['trials']])
        self.aggregator = BayesianDecoderAggregator(exclusion_criteria=self.exclusion_criteria)
        self.aggregator.run_df_aggregation(data, overwrite=True, window=2.5)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path(__file__).parent.stem)

    def plot(self, group_by=None):
        group_names = list(group_by.keys())

        # make plots for different parameters, groups, features
        groups = [g if g != 'feature' else 'feature_name' for g in [*group_names, *self.params]]
        for g_name, data in self.aggregator.group_aligned_df.groupby(groups):
            tags = "_".join([str(n) for n in g_name])
            kwargs = dict(plot_groups=self.plot_group_comparisons, tags=tags)
            # self.plot_theta_data(data, kwargs)
            # self.plot_group_aligned_stats(data, **kwargs)
            self.plot_group_aligned_comparisons(data, **kwargs)
            # self.plot_performance_comparisons(data, tags=tags)

            # make plots for individual plot groups (e.g. correct/incorrect, left/right, update/non-update)
            for plot_types in list(itertools.product(*self.plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
                title = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
                kwargs = dict(title=title, plot_groups=plot_group_dict, tags=f'{tags}_{title}')
                self.plot_group_aligned_data(data, **kwargs)
                # self.plot_scatter_dists_around_update(data, **kwargs)
                # self.plot_trial_by_trial_around_update(data, **kwargs)

        # plot model metrics (comparing parameters across brain regions and features)
        for name, data in self.aggregator.group_df.groupby(group_names):  # TODO - regions/features in same plot?
            self.plot_group_confusion_matrices(data, name, )
            self.plot_tuning_curves(data, name, )
            self.plot_all_confusion_matrices(data, name)
            self.plot_parameter_comparison(data, name, thresh_params=True)

        #  make plots with both brain regions
        self.plot_multiregional_data()
        self.plot_all_groups_error(main_group=group_names[0], sub_group=group_names[1])
        self.plot_all_groups_error(main_group=group_names, sub_group='animal')
        self.plot_all_groups_error(main_group='feature', sub_group='num_units', thresh_params=True)
        self.plot_all_groups_error(main_group='feature', sub_group='num_trials', thresh_params=True)

    def plot_decoding_output_heatmap(self, sfig):
        # load up data
        plot_groups = dict(update_type=['switch'], turn_type=[1, 2], correct=[1], time_label=['t_update'])
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups)
        aligned_data = self.aggregator.select_group_aligned_data(self.aggregator.group_aligned_df, plot_groups, ret_df=True)

        prob_map = np.nanmean(np.stack(aligned_data['probability']), axis=0)
        true_feat = np.nanmean(np.stack(aligned_data['feature']), axis=0)
        time_bins = np.linspace(aligned_data['times'].apply(np.nanmin).min(),
                                aligned_data['times'].apply(np.nanmax).max(),
                                np.shape(prob_map)[1])
        feat_bins = np.linspace(aligned_data['bins'].apply(np.nanmin).min(),
                                aligned_data['bins'].apply(np.nanmax).max(),
                                np.shape(prob_map)[0])
        track_fraction = (feat_bins - np.min(feat_bins)) / (np.max(feat_bins) - np.min(feat_bins))
        true_feat = (true_feat - np.min(feat_bins)) / (np.max(feat_bins) - np.min(feat_bins))
        bounds = [(b - np.min(feat_bins)) / (np.max(feat_bins) - np.min(feat_bins))
                  for b in trial_data['bound_values'].unique()]
        if aligned_data['feature_name'].values[0] == 'choice':  # if bounds are on ends of track, make home between
            bounds = [(bounds[0][1], bounds[1][0]), *bounds]
            ylabel = 'p(new choice)'
        else:
            bounds = [(track_fraction[0], bounds[0][0]), *bounds]  # else put at start
            ylabel = 'fraction of track'

        # plot figure
        ax = sfig.subplots(nrows=1, ncols=1)
        im_times = (time_bins[0] - np.diff(time_bins)[0] / 2, time_bins[-1] + np.diff(time_bins)[0] / 2)
        clipping_masks, images = dict(), dict()
        for b, arm in zip(bounds, ['home', 'initial', 'new']):
            mask = mpl.patches.Rectangle(xy=(0, b[0]), width=1, height=b[1] - b[0],
                                         facecolor='white', alpha=0, transform=ax.get_yaxis_transform())
            clipping_masks[arm] = mask
            images[arm] = ax.imshow(prob_map, cmap=self.colors[f'{arm}_cmap'], aspect='auto',
                                    origin='lower', vmin=0.02, vmax=0.056,
                                    extent=[im_times[0], im_times[-1], track_fraction[0], track_fraction[-1]])
            images[arm].set_clip_path(clipping_masks[arm])
            ticks = None if arm == 'home' else []
            label = f'prob.{self.new_line}density' if arm == 'home' else ''
            cbar = plt.colorbar(images[arm], ax=ax, pad=0.01, fraction=0.046, shrink=0.5,  aspect=12,
                                ticks=ticks)
            cbar.ax.set_title(label, fontsize=8, ha='center')
            for line in b:
                ax.axhline(line, color=self.colors['phase_dividers'], alpha=0.5, linewidth=0.75,)
        ax.plot(time_bins, true_feat, color=self.colors['switch'], linestyle='dotted', linewidth=1,
                label='actual position')

        # plot edge spaces in track
        spaces = getattr(self.aggregator.group_aligned_df['virtual_track'].values[0], 'edge_spacing', [])
        for s in spaces:
            ax.axhspan(*s, color='white', edgecolor=None,)
        ax.axvline(0, color='k', linestyle='dashed', alpha=0.5)

        ax.set(ylim=(track_fraction[0], track_fraction[-1]), ylabel=ylabel,
               xlim=(time_bins[0], time_bins[-1]), xlabel='time around update (s)')
        ax.set_title('switch', color=self.colors['switch'])
        ax.legend(loc='lower right', labelcolor='linecolor')
        sfig.suptitle('Decoded position', fontsize=12)

        return sfig

    def plot_goal_coding(self, sfig, comparison='update_type'):
        # load up data
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value='prob_sum')

        # plot figure
        ax = sfig.subplots(nrows=1, ncols=len(label_map.keys()), sharey=True)
        for update, data in trial_data.groupby('update_type'):
            trial_mat = data.pivot(index=['choice', 'trial_index'], columns='times', values='prob_over_chance')
            new_mat = trial_mat.query('choice == "switch"').to_numpy()
            initial_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
            time_bins = trial_mat.columns.to_numpy()

            ax_id = np.argwhere(np.array(list(label_map.keys())) == update)[0][0]
            ax[ax_id].plot(time_bins, np.nanmean(new_mat, axis=0), color=self.colors['new'], label='new')
            ax[ax_id].fill_between(time_bins, np.nanmean(new_mat, axis=0) + sem(new_mat),
                                   np.nanmean(new_mat, axis=0) - sem(new_mat), color=self.colors['new'], alpha=0.2)
            ax[ax_id].plot(time_bins, np.nanmean(initial_mat, axis=0), color=self.colors['initial'], label='initial')
            ax[ax_id].fill_between(time_bins, np.nanmean(initial_mat, axis=0) + sem(initial_mat),
                                   np.nanmean(initial_mat, axis=0) - sem(initial_mat), color=self.colors['initial'],
                                   alpha=0.2)
            ax[ax_id].axvline(0, color='k', linestyle='dashed', alpha=0.5)
            ax[ax_id].set_title(label_map[update], color=self.colors[update])
            # ax[ax_id].set(ylabel='prob. density', xlim=(time_bins[0], time_bins[-1]), xlabel='time around update (s)')
        sfig.suptitle('Goal arm representations', fontsize=12)
        sfig.supylabel('prob. density', fontsize=10)
        sfig.supxlabel('time around update (s)', ha='center', fontsize=10)

        colors = [self.colors[t] for t in ['initial', 'new']]
        rainbow_text(0.05, 0.85, ['initial', 'new'], colors, ax=ax[0], size=10)

        return sfig

    def plot_goal_coding_stats(self, sfig, comparison='update_type'):
        # get data
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       n_time_bins=11, time_window=(-2.5, 2.5),
                                                                       prob_value='prob_sum')
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'feature_name',
                        'choice']
        time_window = (0, 1.5)
        data_for_stats = (trial_data
                          .query(f'times_binned > {time_window[0]} & times_binned < {time_window[-1]}')  # only look at first 1.5 seconds
                          .groupby(groupby_cols)[['prob_over_chance', 'diff_baseline']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        data_for_stats.reset_index(inplace=True)
        data_for_stats['choice'] = data_for_stats['choice'].map({'initial_stay': 'initial', 'switch': 'new'})
        diff_data = (data_for_stats.pivot(index=groupby_cols[:-1], columns=['choice'],
                                          values=['prob_over_chance_mean'])
                     .reset_index())
        diff_data[('initial_vs_new','')] = diff_data[('prob_over_chance_mean', 'initial')] - \
                                           diff_data[('prob_over_chance_mean', 'new')]
        diff_data = diff_data.droplevel(1, axis=1)
        diff_data['update_type'] = diff_data['update_type'].map(label_map)

        # setup stats - group variables, pairs to compare, and levels of hierarchical data
        var = 'diff_baseline_mean'
        group = 'choice'
        group_list = data_for_stats['choice'].unique()
        combos = list(itertools.combinations(plot_groups[comparison], r=2))
        pairs = [((g, c[0]), (g, c[1], )) for c in combos for g in group_list]
        stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                      approaches=['traditional'], tests=['mann-whitney'])
        stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice', comparison],
                  pairs=pairs, filename=f'goal_coding_stats_{comparison}')

        # plot data TODO - make this mixed effects models
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[2, 1]))
        colors = [self.colors[t] for t in list(label_map.keys())]
        sns.boxplot(data=data_for_stats, x=group, y=var, hue=comparison, ax=ax[0], width=0.5, showfliers=False,
                    palette=colors, medianprops={'color': 'white'}, hue_order=list(label_map.values()))
        ax[0] = clean_box_plot(ax[0], labelcolors=[self.colors['initial'], self.colors['new']])
        ax[0].set(xlabel='goal location', ylabel=f'Δ prob. density from t={time_window[0]} to t={time_window[-1]}')
        ax[0].get_legend().remove()
        rainbow_text(0.5, 0.9, list(label_map.keys()), colors, ax=ax[0], size=8)

        # add stats annotations
        stats_data = stats.stats_df.query(f'approach == "traditional" & test == "mann-whitney"'
                                          f'& variable == "{var}"')
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]
        annot = Annotator(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                          hue_order=list(label_map.keys()))
        annot.new_plot(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison)
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))

        # plot the difference between initial and new probabilities
        sns.boxplot(data=diff_data, x=comparison, y='initial_vs_new', ax=ax[1], width=0.5, showfliers=False,
                    palette=colors, medianprops={'color': 'white'}, order=list(label_map.values()))
        ax[1] = clean_box_plot(ax[1])
        ax[1].set(xlabel=comparison, ylabel=f'initial - new prob. density')

        combos = list(itertools.combinations(list(label_map.values()), r=2))
        diff_pairs = [((c[0],), (c[1],)) for c in combos]
        stats.run(diff_data, dependent_vars=['initial_vs_new'], group_vars=[comparison], pairs=diff_pairs, filename='goal_diff_stats')
        stats_data = stats.stats_df.query(f'approach == "traditional" & test == "mann-whitney"'
                                          f'& variable == "initial_vs_new"')
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in diff_pairs]
        annot = Annotator(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
                          order=list(label_map.values()))
        annot.new_plot(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new')
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))

        sfig.suptitle('Goal representation quantification', fontsize=12)
        return sfig

    def plot_decoding_validation(self, sfig):

        return sfig

    def plot_multiregional_data(self):
        if len(self.aggregator.group_aligned_df['region'].unique()) > 1:
            for name, data in self.aggregator.group_aligned_df.groupby(self.params):
                tags = "_".join([str(n) for n in name])
                kwargs = dict(plot_groups=self.plot_group_comparisons, tags=tags)
                self.plot_region_interaction_stats(data, **kwargs)

                for plot_types in list(itertools.product(*self.plot_groups.values())):
                    plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
                    tags = "_".join([str(n) for n in name])
                    title = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
                    kwargs = dict(title=title, plot_groups=plot_group_dict, tags=f'{tags}_{title}')
                    self.plot_region_interaction_data(data, **kwargs)

    def plot_theta_data(self, data, kwargs):
        unique_bins = data['decoder_bin_size'].unique()
        any_small_bins = np.size([b for b in unique_bins if b < 0.05])

        if any_small_bins:
            self.plot_theta_phase_stats(data, **kwargs)
            self.plot_theta_phase_comparisons(data, **kwargs)

            for plot_types in list(itertools.product(*self.plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
                title = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
                new_kwargs = dict(title=title, plot_groups=plot_group_dict, tags=kwargs["tags"])

                self.plot_phase_modulation_around_update(data, **new_kwargs)
                self.plot_theta_phase_histogram(data, **new_kwargs)

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
            param_tags = '_'.join([f'{p}_{n}' for p, n in zip(self.params, sorted_data['param_values'])])

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
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_confusion_matrices_{param_tags}',
                                 additional_tags=tags)

    def plot_region_interaction_stats(self, param_data, plot_groups=None, tags=''):
        plot_group_dict = {k: [i[0] for i in v] if k != 'turn_type' else v[0] for k, v in plot_groups.items()}
        plot_group_dict.update(time_label=['t_update'])
        interaction_data = self.aggregator.calc_region_interactions(param_data, plot_group_dict)
        interaction_data = interaction_data.query('a_vs_b == "CA1_pos_PFC_choice" & time_label == "t_update"')
        groupby_cols = ['session_id', 'animal', 'trial_id', 'update_type', 'choice', 'correct']
        data_for_stats = (interaction_data
                          .drop(['corr_coeff_sliding', 'corr_coeff', 'corr', 'corr_lags', 'times'], axis=1)
                          .explode(['corr_sliding', 'times_sliding'])
                          .query('times_sliding > 0 & times_sliding < 1')
                          .explode(['corr_sliding', 'lags_sliding'])
                          .groupby(groupby_cols)['corr_sliding']
                          .agg(['mean', 'max'])
                          .add_prefix('corr_sliding_')
                          .reset_index())
        data_for_stats['choice'] = data_for_stats['choice'].map({'initial_stay': 'initial', 'switch': 'new'})
        dependent_vars = ['corr_sliding_mean', 'corr_sliding_max']

        # loop through each comparison to get stats output
        stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io)
        stats_tests = [('bootstrap', 'direct_prob'), ('traditional', 'mann-whitney')]
        group = 'choice'
        for comp, filters in self.data_comparisons.items():
            # define group variables, pairs to compare, and levels of hierarchical data
            mask = pd.concat([data_for_stats[k].isin(v) for k, v in filters.items()], axis=1).all(axis=1)
            comp_data = data_for_stats[mask]
            combos = list(itertools.combinations(filters[comp], r=2))
            pairs = [((c[0], g), (c[1], g)) for c in combos for g in comp_data[group].unique()]

            # filter data based on comparison
            stats.run(comp_data, dependent_vars=dependent_vars, group_vars=[comp, group], pairs=pairs,
                      filename=f'interactions_{comp}_{tags}')

            for var in dependent_vars:
                fig, axes = plt.subplots(2, 2)
                for col, (approach, test) in enumerate(stats_tests):
                    stats_data = stats.stats_df.query(f'approach == "{approach}" & test == "{test}" '
                                                      f'& variable == "{var}"')
                    pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]
                    axes[0][col] = sns.violinplot(data=comp_data, x=comp, y=var, hue=group, ax=axes[0][col])
                    annot = Annotator(axes[0][col], pairs=pairs, data=comp_data, x=comp, y=var, hue=group)
                    (annot
                     .configure(test=None, test_short_name=test, text_format='simple')
                     .set_pvalues(pvalues=pvalues)
                     .annotate())

                    axes[1][col] = sns.boxplot(data=comp_data, x=comp, y=var, hue=group, ax=axes[1][col],
                                               width=0.5, showfliers=False)
                    annot.new_plot(axes[1][col], pairs=pairs, data=comp_data, x=comp, y=var, hue=group)
                    (annot
                     .configure(test=None, test_short_name=test, text_format='simple')
                     .set_pvalues(pvalues=pvalues)
                     .annotate())

                fig.suptitle(f'{comp} - {var}')
                self.results_io.save_fig(fig=fig, axes=axes,
                                         filename=f'compare_{comp}_interactions_{var}',
                                         additional_tags=tags, tight_layout=False)

    def plot_region_interaction_data(self, param_data, title, plot_groups=None, tags=''):
        interaction_data = self.aggregator.calc_region_interactions(param_data, plot_groups)
        if np.size(interaction_data):
            for g_name, g_data in interaction_data.groupby(['a_vs_b']):
                corr_maps = (g_data.groupby(['time_label', 'choice'])['corr_sliding'].apply(
                    lambda x: np.nanmean(np.stack(x), axis=0)))
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

                plt.colorbar(im1, cax=caxes[0], label='initial corr', pad=0.01, fraction=0.04, )
                plt.colorbar(im2, cax=caxes[1], label='switch corr', pad=0.01, fraction=0.04, )

                (  # plot corr coeffs over time
                    so.Plot(g_data_by_time, x='times_sliding', y='corr_coeff_sliding', color='choice')
                        .facet(col='time_label')
                        .add(so.Band(), so.Est(errorbar='se'), )
                        .add(so.Line(linewidth=2), so.Agg(), )
                        .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                        .limit(
                        x=(np.min(interaction_data['times'].values[-1]), np.max(interaction_data['times'].values[1])))
                        .theme(rcparams)
                        .layout(engine='constrained')
                        .on(sfigs[1][0])
                        .plot()
                )

                g_data['corr_coeff'][g_data['corr_coeff'].isna()] = 0
                (  # plot initial - switch difference over time (averages with bar)
                    so.Plot(g_data, x='corr_coeff', color='choice')
                        .facet(col='time_label')
                        .add(so.Bars(alpha=0.5, edgealpha=0.5),
                             so.Hist(stat='proportion', binrange=(-1, 1), binwidth=0.1), )
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
                sfigs[1][0].legend(leg.legendHandles, [t.get_text() for t in leg.texts], loc='upper right',
                                   fontsize='large')

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
        true_feat = aligned_data.groupby('time_label').apply(lambda x: np.nanmean(np.stack(x['feature']), axis=0))
        n_bins = np.shape(prob_maps.loc['start_time'])[0]
        prob_lims = np.linspace(aligned_data['feature'].apply(np.nanmin).min(),
                                aligned_data['feature'].apply(np.nanmax).max(), n_bins)
        time_lims = (aligned_data['times'].apply(np.nanmin).min(), aligned_data['times'].apply(np.nanmax).max())

        # make figure
        ncols, nrows = (len(self.aggregator.align_times), 6)  # heatmap, traces, probheatmaps, velocity, error
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
            axes[0][col].plot(times, true_feat.loc[name], color='w', linestyle='dashed')
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

            im_goal1 = axes[1][col].imshow(stay_mat, cmap=self.colors['initial_cmap'], aspect='auto', vmin=0, vmax=2.5,
                                           origin='lower', extent=[im_times[0], im_times[-1], 0, np.shape(stay_mat)[0]])
            im_goal2 = axes[2][col].imshow(switch_mat, cmap=self.colors['new_cmap'], aspect='auto',
                                           vmin=0, vmax=2.5, origin='lower',
                                           extent=[im_times[0], im_times[-1], 0, np.shape(switch_mat)[0]], )

            axes[3][col].plot(times, np.nanmean(switch_mat, axis=0), color=self.colors['new'], label='new')
            axes[3][col].fill_between(times, np.nanmean(switch_mat, axis=0) + sem(switch_mat),
                                      np.nanmean(switch_mat, axis=0) - sem(switch_mat), color=self.colors['new'],
                                      alpha=0.2)
            axes[3][col].plot(times, np.nanmean(stay_mat, axis=0), color=self.colors['initial'], label='initial')
            axes[3][col].fill_between(times, np.nanmean(stay_mat, axis=0) + sem(stay_mat),
                                      np.nanmean(stay_mat, axis=0) - sem(stay_mat), color=self.colors['initial'],
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
                                                        x='prob_sum_initial_stay', y='prob_sum_switch',
                                                        hue='times_binned',
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
                    .add(so.Band(), so.Est(errorbar='se'), )
                    .add(so.Line(linewidth=2), so.Agg(), )
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
                        row = np.argwhere(locations[kind]['df'][group_list[1]].unique() == name[1])[0][
                            0]  # which row to plot
                        cmap = self.colors[f'{name[1]}_cmap']
                    else:
                        row, col = (0, np.argwhere(trial_data[group_list[0]].unique() == name)[0][0])
                        cmap = self.colors['cmap']

                    matrix = data.groupby([level, 'times'])[kind].mean().unstack().to_numpy()
                    im = axes[row][col].imshow(matrix, cmap=cmap, vmin=0, vmax=0.4, aspect='auto',
                                               origin='lower', extent=[data['times'].min(), data['times'].max(),
                                                                       0, np.shape(matrix)[0]], )
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
        # compile data for comparisons
        compiled_data = []
        for plot_types in list(itertools.product(*plot_groups.values())):
            plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
            filter_dict = dict(time_label=['t_update'], **plot_group_dict)
            trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(param_data, filter_dict)
            aligned_data = self.aggregator.select_group_aligned_data(param_data, filter_dict, ret_df=True)
            compiled_data.append(dict(aligned_data=aligned_data, trial_data=trial_data,
                                      **{k: v[0] for k, v in
                                         filter_dict.items()}))  # TODO - separate out the compilation
        compiled_data_df = pd.DataFrame(compiled_data)

        # plot the data
        for comp, filters in self.data_comparisons.items():
            mask = pd.concat([compiled_data_df[k].isin(v) for k, v in filters.items()], axis=1).all(axis=1)
            self.plot_aligned_comparison(comp, compiled_data_df[mask], filters, tags)

    def plot_aligned_comparison(self, comp, data, filters, tags):
        ncols, nrows = (np.shape(data)[0], 6)  # one col for each comparison, 1 col for difference between two
        fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row',
                                 constrained_layout=True)
        data.reset_index(drop=True, inplace=True)  # reset index so iteration through rows for each column
        trial_lims = []
        for ind, v in data.iterrows():
            # load up data
            title = f'{"".join([f"{k}{f}" for k, f in filters.items()])}'
            bounds = v['trial_data']['bound_values'].unique()
            spaces = getattr(self.aggregator.group_df['virtual_track'].values[0], 'edge_spacing', [])
            prob_maps = v['aligned_data'].groupby('time_label').apply(
                lambda x: np.nanmean(np.stack(x['probability']), axis=0))
            true_feat = v['aligned_data'].groupby('time_label').apply(
                lambda x: np.nanmean(np.stack(x['feature']), axis=0))
            n_bins = np.shape(prob_maps.loc['t_update'])[0]
            prob_lims = np.linspace(v['aligned_data']['feature'].apply(np.nanmin).min(),
                                    v['aligned_data']['feature'].apply(np.nanmax).max(), n_bins)
            time_lims = (v['aligned_data']['times'].apply(np.nanmin).min(),
                         v['aligned_data']['times'].apply(np.nanmax).max())

            trial_mat = v['trial_data'].pivot(index=['choice', 'trial_index'], columns='times',
                                              values='prob_over_chance')
            switch_mat = trial_mat.query('choice == "switch"').to_numpy()
            stay_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
            trial_lims.append(np.shape(switch_mat)[0])
            times = trial_mat.columns.to_numpy()
            im_times = (times[0] - np.diff(times)[0] / 2, times[-1] + np.diff(times)[0] / 2)

            im_prob = axes[0][ind].imshow(prob_maps.loc['t_update'] * n_bins, cmap=self.colors['cmap'], aspect='auto',
                                          origin='lower', vmin=0.6, vmax=2.8,
                                          extent=[im_times[0], im_times[-1], prob_lims[0], prob_lims[-1]])
            axes[0][ind].invert_yaxis()
            axes[0][ind].plot(times, true_feat.loc['t_update'], color='w', linestyle='dashed')
            for b in bounds:
                axes[0][ind].axhline(b[0], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)
                axes[0][ind].axhline(b[1], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)
            for s in spaces:
                xmin, xmax = 0, 1
                if (times[0], times[-1]) == (0, time_lims[-1]):
                    xmin, xmax = 0.5, 1
                elif (times[0], times[-1]) == (time_lims[0], 0):
                    xmin, xmax = 0, 0.5
                axes[0][ind].axhspan(*s, color='#DDDDDD', edgecolor=None, xmin=xmin, xmax=xmax)

            im_goal1 = axes[1][ind].imshow(stay_mat, cmap=self.colors['initial_cmap'], aspect='auto', vmin=0,
                                           vmax=2.5,
                                           origin='lower',
                                           extent=[im_times[0], im_times[-1], 0, np.shape(stay_mat)[0]])
            im_goal2 = axes[2][ind].imshow(switch_mat, cmap=self.colors['new_cmap'], aspect='auto',
                                           vmin=0, vmax=2.5, origin='lower',
                                           extent=[im_times[0], im_times[-1], 0, np.shape(switch_mat)[0]], )

            axes[3][ind].plot(times, np.nanmean(switch_mat, axis=0), color=self.colors['new'],
                              label='new')
            axes[3][ind].fill_between(times, np.nanmean(switch_mat, axis=0) + sem(switch_mat),
                                      np.nanmean(switch_mat, axis=0) - sem(switch_mat),
                                      color=self.colors['new'],
                                      alpha=0.2)
            axes[3][ind].plot(times, np.nanmean(stay_mat, axis=0), color=self.colors['initial'],
                              label='initial')
            axes[3][ind].fill_between(times, np.nanmean(stay_mat, axis=0) + sem(stay_mat),
                                      np.nanmean(stay_mat, axis=0) - sem(stay_mat),
                                      color=self.colors['initial'],
                                      alpha=0.2)
            axes[3][ind].axhline(1, linestyle='dashed', color='k', alpha=0.5)
            axes[0][ind].set(ylim=(prob_lims[0], prob_lims[-1]), ylabel=self.aggregator.group_df['feature'].values[0],
                             title=v[comp], xlim=time_lims)
            axes[1][ind].set(ylim=(0, np.shape(stay_mat)[0]), ylabel='trials', title=v[comp], xlim=time_lims)
            axes[2][ind].set(ylim=(0, np.shape(switch_mat)[0]), ylabel='trials', title=v[comp], xlim=time_lims)
            axes[3][ind].set(ylabel='prob / chance', title=v[comp], xlim=time_lims)

            veloc = np.stack(v['aligned_data']['rotational_velocity'])
            error = np.stack(v['aligned_data']['error'])
            axes[4][ind].plot(v['aligned_data']['times'].values[0], np.nanmean(veloc, axis=0),
                              color=self.colors['control'],
                              label='rotational velocity')
            axes[4][ind].fill_between(v['aligned_data']['times'].values[0], np.nanmean(veloc, axis=0) + sem(veloc),
                                      np.nanmean(veloc, axis=0) - sem(veloc), color=self.colors['control'],
                                      alpha=0.2)
            axes[5][ind].plot(v['aligned_data']['times'].values[0], np.nanmean(error, axis=0),
                              color=self.colors['error'],
                              label='decoding error')
            axes[5][ind].fill_between(v['aligned_data']['times'].values[0], np.nanmean(error, axis=0) + sem(error),
                                      np.nanmean(error, axis=0) - sem(error), color=self.colors['error'],
                                      alpha=0.2)
            axes[4][ind].set(ylabel='velocity', title=v[comp], xlim=time_lims)
            axes[5][ind].set(ylabel='decoding error', title=v[comp], xlim=time_lims)

        plt.colorbar(im_prob, ax=axes[0][ind], label='prob / chance', pad=0.01, fraction=0.046,
                     location='right')
        plt.colorbar(im_goal1, ax=axes[1][ind], label='prob / chance', pad=0.01, fraction=0.046,
                     location='right')
        plt.colorbar(im_goal2, ax=axes[2][ind], label='prob / chance', pad=0.01, fraction=0.046,
                     location='right')

        for r in range(nrows):
            axes[r][ind].legend(fontsize='large')
            for c in range(ncols):
                axes[r][c].axvline(0, color='k', linestyle='dashed', alpha=0.5)
                axes[1][c].set(ylim=(0, np.max(np.array(trial_lims))))
                axes[2][c].set(ylim=(0, np.max(np.array(trial_lims))))

        fig.suptitle(title)
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'compare_{comp}_aligned_data', additional_tags=tags,
                                 tight_layout=False)

    def plot_group_aligned_stats(self, param_data, plot_groups, tags=''):
        plot_group_dict = {k: [i[0] for i in v] if k != 'turn_type' else v[0] for k, v in plot_groups.items()}
        plot_group_dict.update(time_label=['t_update'])
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(param_data, plot_group_dict, n_time_bins=11,
                                                                       time_window=(-2.5, 2.5))
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'feature_name',
                        'choice']

        windows = [(0, 1.5), (0, 1), (0, 2)]
        for win in windows:
            data_for_stats = (trial_data
                              .query(
                f'times_binned > {win[0]} & times_binned < {win[1]}')  # only look at first two seconds, could do 1.75 too
                              .groupby(groupby_cols)[['prob_over_chance', 'diff_baseline']]  # group by trial/trial type
                              .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                              .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
            dependent_vars = data_for_stats.columns.to_list()
            data_for_stats.reset_index(inplace=True)
            data_for_stats['choice'] = data_for_stats['choice'].map({'initial_stay': 'initial', 'switch': 'new'})

            # loop through each comparison to get stats output
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io)
            group = 'choice'
            for comp, filters in self.data_comparisons.items():
                # define group variables, pairs to compare, and levels of hierarchical data
                mask = pd.concat([data_for_stats[k].isin(v) for k, v in filters.items()], axis=1).all(axis=1)
                group_list = data_for_stats[mask][group].unique()
                combos = list(itertools.combinations(filters[comp], r=2))
                pairs = [((c[0], g), (c[1], g)) for c in combos for g in group_list]

                # filter data based on comparison
                stats.run(data_for_stats[mask], dependent_vars=dependent_vars, group_vars=[comp, group], pairs=pairs,
                          filename=f'aligned_decoding_{comp}_{tags}')

                for var in dependent_vars:
                    fig, axes = plt.subplots(2, 2)
                    for col, (approach, test) in enumerate(
                            [('bootstrap', 'direct_prob'), ('traditional', 'mann-whitney')]):
                        stats_data = stats.stats_df.query(f'approach == "{approach}" & test == "{test}" '
                                                          f'& variable == "{var}"')
                        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]

                        axes[0][col] = sns.violinplot(data=data_for_stats[mask], x=comp, y=var, hue=group,
                                                      ax=axes[0][col])
                        annot = Annotator(axes[0][col], pairs=pairs, data=data_for_stats[mask], x=comp, y=var,
                                          hue=group)
                        (annot
                         .configure(test=None, test_short_name=test, text_format='simple')
                         .set_pvalues(pvalues=pvalues)
                         .annotate())

                        axes[1][col] = sns.boxplot(data=data_for_stats[mask], x=comp, y=var, hue=group, ax=axes[1][col],
                                                   width=0.5, showfliers=False)
                        annot.new_plot(axes[1][col], pairs=pairs, data=data_for_stats[mask], x=comp, y=var, hue=group)
                        (annot
                         .configure(test=None, test_short_name=test, text_format='simple')
                         .set_pvalues(pvalues=pvalues)
                         .annotate())

                    fig.suptitle(f'{comp} - {var}')
                    self.results_io.save_fig(fig=fig, axes=axes,
                                             filename=f'compare_{comp}_aligned_data_{var}_window_{win}',
                                             additional_tags=tags, tight_layout=False)

    def plot_tuning_curves(self, data, name):
        print('Plotting tuning curves...')

        feat = data['feature'].values[0]
        locations = data.virtual_track.values[0].cue_end_locations.get(feat, dict())
        tuning_curve_params = [p for p in self.params if p not in ['decoder_bins']]
        data_const_decoding = data[data['decoder_bin_size'] == data['decoder_bin_size'].values[0]]
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
                im = axes[row_id][col_id].imshow(tuning_curve_scaled[sort_index, :], cmap=self.colors['cmap_r'],
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
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'theta_mod_around_update_{title}',
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
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'theta_phase_hist_{title}', additional_tags=tags)

    def plot_theta_phase_comparisons(self, param_data, plot_groups=None, tags=''):
        print('Plotting theta phase comparisons...')

        compiled_data = []
        for plot_types in list(itertools.product(*plot_groups.values())):
            plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
            filter_dict = dict(time_label=['t_update'], **plot_group_dict)
            theta_phase_data = self.aggregator.calc_theta_phase_data(param_data, filter_dict, ret_by_trial=True)
            compiled_data.append(dict(data=theta_phase_data, **{k: v[0] for k, v in filter_dict.items()}))

        # compile data for comparisons
        comparisons = dict(update_type=dict(update_type=['switch', 'stay', 'non_update'],
                                            correct=[1]),
                           correct=dict(update_type=['switch'], correct=[0, 1]))
        compiled_data_df = pd.DataFrame(compiled_data)

        # plot the data and accompanying stats
        for comp, filters in comparisons.items():
            mask = pd.concat([compiled_data_df[k].isin(v) for k, v in filters.items()], axis=1).all(axis=1)
            data = compiled_data_df[mask].reset_index(
                drop=True)  # reset index so iteration through rows for each column

            # calculate difference
            ncols, nrows = (np.shape(data)[0] * 2, 2)  # cols for pre/post, g12 g_diff, row for each value
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5), squeeze=False, sharey='row')
            for col_add, v in data.iterrows():
                r_data = v['data'].query('bin_name == "full"')
                for time, d_time in r_data.groupby('times'):
                    t_ind = np.argwhere(r_data['times'].unique() == time)[0][0]
                    for loc in ['initial_stay', 'switch', 'home', 'theta_amplitude']:
                        lstyle = ['dashed' if time == 'pre' else 'solid'][0]
                        color = [self.colors[loc] if loc in ['switch', 'initial_stay'] else 'k'][0]
                        row_ind = [0 if loc == 'theta_amplitude' else 1][0]
                        mean_loc = d_time.groupby('phase_mid')[loc].mean()
                        err_loc = d_time.groupby('phase_mid')[loc].apply(sem)
                        axes[row_ind][t_ind + 2 * col_add].plot(mean_loc.index.to_numpy() / np.pi,
                                                                mean_loc.to_numpy(), color=color,
                                                                linestyle=lstyle,
                                                                label=f'{loc}')
                        axes[row_ind][t_ind + 2 * col_add].fill_between(err_loc.index.to_numpy() / np.pi,
                                                                        mean_loc.to_numpy() - err_loc.to_numpy(),
                                                                        mean_loc.to_numpy() + err_loc.to_numpy(),
                                                                        alpha=0.2, color=color)
                        # NOTE err_lower/upper is STD for visualization purposes
                        axes[row_ind][t_ind + 2 * col_add].set(title=f'{v[comp]} - {time}', ylabel='mean probability')

                ax_list = axes.flat
                for ax in ax_list:
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
                    ax.set(xlabel='theta phase')
                    ax.legend()
                fig.suptitle(f'Theta phase histogram comparison')
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'compare_{comp}_theta_phase_hist',
                                     additional_tags=tags)

    def plot_theta_phase_stats(self, param_data, plot_groups=None, tags=''):
        plot_group_dict = {k: [i[0] for i in v] if k != 'turn_type' else v[0] for k, v in plot_groups.items()}
        plot_group_dict.update(time_label=['t_update'])
        theta_data = self.aggregator.calc_theta_phase_data(param_data, plot_group_dict, ret_by_trial=True)
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'feature_name', 'times',
                        'phase_mid']
        data_for_stats = (theta_data
                          .query('times == "post" & bin_name == "quarter"')  # only look at post update
                          .groupby(groupby_cols)[['initial_stay', 'switch', 'home']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, had max, or peak latency for each trial
                          .melt(ignore_index=False)
                          .rename(columns={'variable_0': 'location'})
                          .set_index('location', append=True)
                          .pivot(columns='variable_1', values='value')
                          .add_prefix('decoding_')
                          .reset_index())
        data_for_stats['location'] = data_for_stats['location'].map(
            {'initial_stay': 'initial', 'switch': 'new', 'home': 'home'})
        stats_tests = [('traditional', 'mann-whitney')]

        # loop through each comparison to get stats output
        stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                      approaches=[s[0] for s in stats_tests], tests=[s[1] for s in stats_tests])
        group = ['location', 'phase_mid']
        dependent_vars = ['decoding_mean']

        for comp, filters in self.data_comparisons.items():
            # define group variables, pairs to compare, and levels of hierarchical data
            mask = pd.concat([data_for_stats[k].isin(v) for k, v in filters.items()], axis=1).all(axis=1)
            data_to_plot = data_for_stats[mask]
            group_list = list(data_to_plot.groupby(group).groups.keys())
            combos = list(itertools.combinations(filters[comp], r=2))
            pairs = [((c[0], *g), (c[1], *g)) for c in combos for g in group_list]

            # run comparisons between theta phase
            for f in filters[comp]:
                data_to_comp = data_to_plot.query(f'{comp} == "{f}"')
                group_list = list(data_to_plot.groupby('phase_mid').groups.keys())
                combos = list(itertools.combinations(data_to_plot['location'].unique(), r=2))
                pairs = [((c[0], g), (c[1], g)) for c in combos for g in group_list]
                stats.run(data_to_comp, dependent_vars=dependent_vars, group_vars=['location', 'phase_mid'],
                          pairs=pairs, filename=f'theta_quadrant_{comp}_{tags}_{f}')

                for var in dependent_vars:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

                    for col, (approach, test) in enumerate(stats_tests):
                        stats_data = stats.stats_df.query(f'approach == "{approach}" & test == "{test}" '
                                                          f'& variable == "{var}"')
                        stats_data = stats_data[stats_data['pair'].isin(pairs)]
                        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]

                        axes[0][col] = sns.violinplot(data=data_to_comp, x='location', y=var, hue='phase_mid',
                                                      ax=axes[0][col])
                        annot = Annotator(axes[0][col], pairs=pairs, data=data_to_comp, x='location', y=var,
                                          hue='phase_mid')
                        (annot
                         .configure(test=None, test_short_name=test, text_format='simple')
                         .set_pvalues(pvalues=pvalues)
                         .annotate())

                        axes[1][col] = sns.boxplot(data=data_to_comp, x='location', y=var, hue='phase_mid',
                                                   ax=axes[1][col], width=0.5, showfliers=False)
                        annot.new_plot(axes[1][col], pairs=pairs, data=data_to_comp, x='location', y=var,
                                       hue='phase_mid')
                        (annot
                         .configure(test=None, test_short_name=test, text_format='simple')
                         .set_pvalues(pvalues=pvalues)
                         .annotate())

                    fig.suptitle(f'{f}')
                    self.results_io.save_fig(fig=fig, axes=axes,
                                             filename=f'compare_phase_{f}_theta_modulation_{var}_{comp}',
                                             additional_tags=tags, tight_layout=False)

            # run comparisons between trial types
            stats.run(data_to_plot, dependent_vars=dependent_vars, group_vars=[comp, *group], pairs=pairs,
                      filename=f'theta_modulation_{comp}_{tags}')
            for var in dependent_vars:
                for g_name, g_data in data_to_plot.groupby(group[0]):
                    pairs_subset = [p for p in pairs if p[0][1] == g_name]
                    fig, axes = plt.subplots(2, 2)
                    for col, (approach, test) in enumerate(stats_tests):
                        stats_data = stats.stats_df.query(f'approach == "{approach}" & test == "{test}" '
                                                          f'& variable == "{var}"')
                        stats_data = stats_data[stats_data['pair'].isin(pairs_subset)]
                        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs_subset]

                        if len(pairs_subset[0][0]) > 2:
                            g_data['hue'] = g_data[group].apply(lambda x: str(tuple(x)), axis=1)
                            annot_pairs = [((c[0], str(g)), (c[1], str(g))) for c in combos for g in group_list
                                           if g[0] == g_name]
                        else:
                            g_data['hue'] = g_data[group]
                            annot_pairs = pairs_subset

                        axes[0][col] = sns.violinplot(data=g_data, x=comp, y=var, hue='hue',
                                                      ax=axes[0][col])
                        annot = Annotator(axes[0][col], pairs=annot_pairs, data=g_data, x=comp, y=var, hue='hue')
                        (annot
                         .configure(test=None, test_short_name=test, text_format='simple')
                         .set_pvalues(pvalues=pvalues)
                         .annotate())

                        axes[1][col] = sns.boxplot(data=g_data, x=comp, y=var, hue='hue', ax=axes[1][col],
                                                   width=0.5, showfliers=False)
                        annot.new_plot(axes[1][col], pairs=annot_pairs, data=g_data, x=comp, y=var, hue='hue')
                        (annot
                         .configure(test=None, test_short_name=test, text_format='simple')
                         .set_pvalues(pvalues=pvalues)
                         .annotate())

                    fig.suptitle(f'{comp}')
                    self.results_io.save_fig(fig=fig, axes=axes,
                                             filename=f'compare_{comp}_theta_modulation_{var}_{g_name}',
                                             additional_tags=tags, tight_layout=False)

    def plot_performance_comparisons(self, param_data, plot_groups=None, tags=''):
        # load up data
        plot_groups = dict(update_type=['non_update', 'switch', 'stay'],
                           turn_type=[1, 2],
                           correct=[0, 1],
                           time_label=['t_update'])  # use as default if no value given
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(param_data, plot_groups=plot_groups,
                                                                       n_time_bins=6)
        trial_data.dropna(subset='times_binned', inplace=True)
        trial_block = 40

        # make figure
        time_bins = trial_data['times_binned'].unique()
        for t_ind, t in enumerate(time_bins):
            time_included = list(time_bins[(t_ind - 1):(t_ind + 1)])  # get all data up to this point in time
            data_subset = trial_data.query(f'times_binned.isin({time_included})')
            fig = plt.figure(figsize=(8.5, 11), constrained_layout=True)
            sfigs = fig.subfigures(3, 3, hspace=0.1, wspace=0.1)
            for g_name, g_data in data_subset.groupby('update_type', sort=False):
                # bin trials for percent correct calculations
                col = np.argwhere(trial_data['update_type'].unique() == g_name)[0][0]
                bins = np.hstack([g_data['index'].unique()[::trial_block], g_data['index'].unique()[-1]])
                g_data['trials_binned'] = pd.cut(g_data['index'], bins=bins, include_lowest=True, labels=False,
                                                 duplicates='drop')

                for row, level in enumerate(['trials_binned', 'session_id', 'animal']):
                    plot_data = g_data.groupby([level, 'choice', 'region']).mean().reset_index()

                    (  # plot scatters with estimates
                        so.Plot(plot_data, x='correct', color='choice')
                            .pair(y=['diff_baseline', ])  # 'prob_over_chance' remove for now
                            .add(so.Dot(alpha=0.3))
                            .add(so.Line(), so.PolyFit(order=1), )
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
            times = f'up_to_{time_included}s_{trial_block}trials'
            self.results_io.save_fig(fig=fig, filename=f'performance_comparisons', additional_tags=f'{tags}_{times}',
                                     tight_layout=False)
