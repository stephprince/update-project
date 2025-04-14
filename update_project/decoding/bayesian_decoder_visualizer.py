import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from pathlib import Path
from matplotlib import ticker
from scipy.stats import sem, pearsonr
from statannotations.Annotator import Annotator
from scipy import signal

from update_project.decoding.bayesian_decoder_aggregator import BayesianDecoderAggregator
from update_project.general.results_io import ResultsIO
from update_project.general.plots import plot_distributions, plot_scatter_with_distributions, rainbow_text, \
    clean_box_plot, clean_violin_plot, add_task_phase_lines
from update_project.general.place_cells import get_place_fields, get_largest_field_loc
from update_project.statistics.statistics import Stats
from update_project.base_visualization_class import BaseVisualizationClass


class BayesianDecoderVisualizer(BaseVisualizationClass):
    def __init__(self, data, exclusion_criteria=None, params=None, threshold_params=None, window=2.5, turn_to_flip=2):
        super().__init__(data)
        self.exclusion_criteria = exclusion_criteria or dict(units=20, trials=50)
        self.params = params
        self.threshold_params = threshold_params or dict(num_units=[self.exclusion_criteria['units']],
                                                         num_trials=[self.exclusion_criteria['trials']])
        self.aggregator = BayesianDecoderAggregator(exclusion_criteria=self.exclusion_criteria, turn_to_flip=turn_to_flip)
        self.aggregator.run_df_aggregation(data, overwrite=True, window=window)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path(__file__).parent.stem)

    def plot(self, group_by=None):
        group_names = list(group_by.keys())

        # make plots for different parameters, groups, features
        groups = [g if g != 'feature' else 'feature_name' for g in [*group_names, *self.params]]
        groups = [g for g in groups if g in self.aggregator.group_aligned_df.columns]
        for g_name, data in self.aggregator.group_aligned_df.groupby(groups):
            tags = "_".join([str(n) for n in g_name])
            kwargs = dict(plot_groups=self.plot_group_comparisons, tags=tags)
            self.plot_motor_controls(data, **kwargs)
            self.plot_theta_data(data, kwargs)
            self.plot_group_aligned_stats(data, **kwargs)
            self.plot_group_aligned_comparisons(data, **kwargs)
            self.plot_performance_comparisons(data, tags=tags)
        
    def plot_decoding_error_location(self, sfig, update_type=['non_update'], prob_value='prob_sum', time_label='t_update', other_zones=dict(), ylim=None):
        plot_groups = dict(update_type=update_type, turn_type=[1, 2], correct=[1], time_label=[time_label])
        #decoding_measures = 'zscore_prob' if use_zscores else prob_value
        choice_mapping = {z: z for z in other_zones}
        choice_mapping['initial_stay'] = 'initial'
        choice_mapping['switch'] = 'new'
        
        aligned_data = self.aggregator.select_group_aligned_data(self.aggregator.group_aligned_df, plot_groups,
                                                                 ret_df=True)
        
        #making a mask based on location of the animal. should limit to only the home arm between initial cue and choice cue
        error2 = np.stack(aligned_data['error'])#don't want probability, want error
        features2 = np.stack(aligned_data['feature'])
        mask = (features2 < 120.35) #& (features2 > 250.35)#should remove any time while intial cue is presented or arms themselves
        #probability2[expanded_mask] = np.nan
        aligned_data.error[mask] = np.nan
        
        #reshaping + manipulating array like calc_trial_by_trial_quant_data. calc also gets zscores, but should be available in tria_data if needed? if not, add zscore lines and uncomment below
        quant_df = self.aggregator.quantify_aligned_data(param_data, aligned_data, ret_df=True, other_zones=other_zones)#getting bounds and quantifying choice
        if np.size(quant_df) and quant_df is not None:
            # get diff from baseline
            prob_sum_mat = np.vstack(quant_df[prob_value])
            align_end = np.argwhere(quant_df['times'].values[0] >= 0)[0][0]
            align_start = np.argwhere(quant_df['times'].values[0] >= -1.5)[0][0]
            if align_end != 0:
                # align_end = align_end - 1  # get index immediately preceding 0 if not the first one
                prob_sum_diff = prob_sum_mat.T - np.nanmean(prob_sum_mat[:, align_start:align_end], axis=1)
            elif align_end == 0:
                prob_sum_diff = prob_sum_mat.T - prob_sum_mat[:, align_end]  # only use first bin if that's all there is
            quant_df['diff_baseline'] = list(prob_sum_diff.T)
            #quant_df['zscore_prob'] = list((prob_sum_mat - zscore_mean) / zscore_std)

            # get diff from left vs. right bounds
            quant_df['trial_index'] = quant_df.index
            quant_df = (quant_df
                        .explode(['times', 'prob_sum', 'prob_over_chance', 'diff_baseline', 'zscore_prob',
                                  'rotational_velocity', 'translational_velocity', 'error'])
                        .reset_index())
            quant_df['times_binned'] = pd.cut(quant_df['times'], np.linspace(*time_window, n_time_bins)).apply(
                lambda x: x.mid)
            quant_df = pd.DataFrame(quant_df.to_dict())  # fix to avoid object dtype errors in seaborn
            choice_df = quant_df.pivot(
                index=['session_id', 'animal', 'time_label', 'times', 'times_binned', 'trial_index'],
                columns=['choice'],
                values=[prob_value, 'diff_baseline']).reset_index()  # had times here before, , 'zscore_prob'
            choice_df['diff_switch_stay'] = choice_df[(prob_value, 'switch')] - choice_df[(prob_value, 'initial_stay')]
            #choice_df['zscore_diff_switch_stay'] = choice_df[('zscore_prob', 'switch')] - \
            #                                       choice_df[('zscore_prob', 'initial_stay')]
            choice_df.columns = ['_'.join(c) if c[1] != '' else c[0] for c in choice_df.columns.to_flat_index()]
            #array now reshaped, quant_df is my current array.
        trial_data = quant_df
        #now back to trial_data, should hopefully just be able to use regionchoice for the remaining
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'time_label', 'feature_name', 
                    'choice']
        data_for_stats = (trial_data
                      .query(
            f'times_binned > {time_window[0]} & times_binned < {time_window[-1]}')  # only look at first 1.5 seconds
                      .groupby(groupby_cols)[[decoding_measures, 'diff_baseline', 'error']]  # group by trial/trial type
                      .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                      .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        
        data_for_stats.reset_index(inplace=True)
        data_for_stats['choice'] = data_for_stats['choice'].map(choice_mapping)
        data_for_stats[comparison] = data_for_stats[comparison].map(label_map)#renaming non update to delay only
        diff_data = (data_for_stats.pivot(index=groupby_cols[:-1], columns=['choice'],
                                      values=['error_mean'])#f'{decoding_measures}_mean'
                     .reset_index())
        diff_data[('initial_vs_new', '')] = diff_data[('error_mean', 'initial')] - \
                                            diff_data[('error_mean', 'new')]##f'{decoding_measures}_mean' in both ()
        diff_data = diff_data.droplevel(1, axis=1)

        # setup stats - group variables, pairs to compare, and levels of hierarchical data
        var = 'error_mean'#f'{decoding_measures}_mean'
        group = 'choice'
        group_list = data_for_stats['choice'].unique()#zones that are being decoded (what i want to be compared)
        plot_groups['region'] = [('CA1',), ('PFC',)]
        combo_list = [label_map[g] for g in plot_groups[comparison]]#switch back to comparison in plot groups later
        combos = list(itertools.combinations(combo_list, r=2))
        pairs = [((g, c[0]), (g, c[1],)) for c in combos for g in group_list]
        if comparison == 'choice':
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice'],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')
        else:
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='manuscript')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice', comparison],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')
        # plot data
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[3, 1]))
        colors = [self.colors[t] for t in list(label_map.values())]

        sess_averages = data_for_stats.groupby(['session_id', group, comparison])[var].mean().reset_index()
        common_kwargs = dict(data=sess_averages, x=group, y=var, hue=comparison, ax=ax[0],
                             hue_order=list(label_map.values()), errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
        if comparison == 'representations':
            common_kwargs.update(hue=group)
        if stripplot:
            ax[0] = sns.stripplot(data=sess_averages, x=group, y=var, hue=comparison, ax=ax[0],
                                  hue_order=list(label_map.values()), zorder=1, jitter=True,
                                  palette=[self.colors['home_medium']] * len(colors), alpha=0.4, dodge=True )#legend=False,
            if comparison == 'representations':
                common_kwargs.update(hue=group)
        category_order = ['initial','new']#'local','central',
        ax[0] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5, order=category_order)
        ax[0] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None, order=category_order)
        ax[0].set(xlabel=f'goal location', ylabel=f'prob. density after update onset')
        ax[0].get_legend().remove()
        rainbow_text(0.5, 0.9, list(label_map.values()), colors, ax=ax[0], size=8)
        if ylim:
            ax[0].set(ylim=ylim)
        # add stats annotations
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "{var}"')
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]
        annot = Annotator(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                          hue_order=list(label_map.values()), order=category_order)
        annot.new_plot(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                       hue_order=list(label_map.values()), order=category_order)
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))
        bins = self.aggregator.group_aligned_df['bins'].values[0]
        data_for_stats['error_mean'] = (data_for_stats['error_mean'] - np.min(bins)) / (np.max(bins) - np.min(bins))
        data_for_stats = data_for_stats.query('choice == "initial"')
        stats.run(data_for_stats, dependent_vars=['error_mean'], group_vars=['choice', comparison],
                  pairs=pairs, filename=f'decoding_error_stats_{comparison}_{tags}')
        sfig.suptitle(f'{title} goal representation quantification', fontsize=12)
        return sfig

    def plot_decoding_output_heatmap(self, sfig, update_type='switch', prob_value='prob_sum', feat='position',start_nan=False, time_label='t_update', prev_turn=None):
        # load up data
        plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=[time_label])
        #if start_nan:
        #    plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=['start_time','t_update'])
        #else:
        #    plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=['t_update'])
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value, hm=True)
        aligned_data = self.aggregator.select_group_aligned_data(self.aggregator.group_aligned_df, plot_groups,
                                                                 ret_df=True)
        if prev_turn is not None:
            file_path = r"Y:\singer\Steph\Code\update-project\results\decoding\intermediate_data\combined_data.xlsx"
            df = pd.read_excel(file_path)#loading in the excel spreadsheet with all of the trial types, made from behavior data where all trials still present with info
            combined_data = df.sort_values(by=['animal','session_id','trial_id'])
            combined_data['prev_turn_type'] = combined_data.groupby(['animal','session_id'])['turn_type'].shift(1)#making prev_turn_type have the value of the previous 
            if prev_turn == 'diff':
                # Filter combined_data to get trials where the previous turn_type was 'right'
                right_right_turn_trials = combined_data[(combined_data['prev_turn_type'] == 'right')#since update switches
                                                        & (combined_data['turn_type'] == 'right')
                                                        | (combined_data['prev_turn_type'] == 'left')
                                                        & (combined_data['turn_type'] == 'left')]#turn_type = 2 in aligned_data, also the turn that is flipped
                aligned_data = aligned_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            elif prev_turn == 'same':#refers to if mouse is turning the same way prev and initially told to do the same turn. is flipped because these are switch trials
                # Filter combined_data to get trials where the previous turn_type was 'right'
                right_right_turn_trials = combined_data[(combined_data['prev_turn_type'] == 'right')
                                                        & (combined_data['turn_type'] == 'left')
                                                        | (combined_data['prev_turn_type'] == 'left')
                                                        & (combined_data['turn_type'] == 'right')]#turn_type = 2 in aligned_data, also the turn that is flipped
                aligned_data = aligned_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            elif prev_turn == 'check':
                right_right_turn_trials = combined_data[(combined_data['turn_type'] == 'left')]#turn_type = 2 in aligned_data, also the turn that is flipped
                aligned_data = aligned_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            elif prev_turn == 'second_check':
                aligned_data = aligned_data.query('turn_type == 1')
                

        #if aligned_data.time_label == 'start_time':
        #    aligned_data.probability[:] = np.nan
        #if trial_data.time_label == 'start_time':
        #    trial_data.probability[:] = np.nan
        
        clim = (0.02, 0.045) if feat == 'position' else (0.02, 0.04)
        #mask = np.array([np.any(np.isnan(f)) for f in aligned_data['feature']])
        probability2 = np.stack(aligned_data['probability'])
        features2 = np.stack(aligned_data['feature'])
        if start_nan:
            mask = np.isnan(features2)| (features2 < 6)#6 should remove any time from when the mouse is "sitting" at the start
            #mask = np.zeros_like(features2, dtype=bool)
            #print("The maximum value from features2[:, :15] is:", np.max(features2[:, :15]))
            #print("The minimum value from features2[:, :15] is:", np.min(features2[:, :15]))
            #mask[:, :15] = features2[:,:15] > 255
            expanded_mask = np.broadcast_to(mask[:, np.newaxis, :], probability2.shape)
            probability2[expanded_mask] = np.nan
    
        #nan out any probability when the mouse is before the 0.1 (assuming track starts at 0) if needed
        prob_map = np.nanmean(probability2, axis=0)
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
        if aligned_data['feature_name'].values[0] in ['choice', 'x_position']:  # if bounds are on ends of track, make home between
            bounds = [(bounds[0][1], bounds[1][0]), *bounds]
            ylabel = 'p(new choice)'
            print(ylabel,bounds)
        else:
            bounds = [(track_fraction[0], bounds[0][0]), *bounds]  # else put at start
            ylabel = 'fraction of track'
            print(ylabel,bounds)

        # plot figure
        ax = sfig.subplots(nrows=1, ncols=1)
        im_times = (time_bins[0] - np.diff(time_bins)[0] / 2, time_bins[-1] + np.diff(time_bins)[0] / 2)
        clipping_masks, images = dict(), dict()
        for b, arm in zip(bounds, ['home', 'initial', 'new']):
            mask = mpl.patches.Rectangle(xy=(0, b[0]), width=1, height=b[1] - b[0],
                                         facecolor='white', alpha=0, transform=ax.get_yaxis_transform())
            clipping_masks[arm] = mask
            images[arm] = ax.imshow(prob_map, cmap=self.colors[f'{arm}_cmap'], aspect='auto',
                                    origin='lower', vmin=clim[0], vmax=clim[1],
                                    extent=[im_times[0], im_times[-1], track_fraction[0], track_fraction[-1]])
            images[arm].set_clip_path(clipping_masks[arm])
            ticks = None if arm == 'home' else []
            label = f'prob.{self.new_line}density' if arm == 'home' else ''
            cbar = plt.colorbar(images[arm], ax=ax, pad=0.01, fraction=0.046, shrink=0.5, aspect=12,
                                ticks=ticks)
            cbar.ax.set_title(label, fontsize=8, ha='center')
            for line in b:
                ax.axhline(line, color=self.colors['phase_dividers'], alpha=0.5, linewidth=0.75, )
        ax.plot(time_bins, true_feat, color=self.colors[f'incorrect'], linestyle='dotted', linewidth=1.25,
                label='actual position')#self.colors[f'{update_type}_light']

        # plot edge spaces in track
        # spaces = getattr(self.aggregator.group_aligned_df['virtual_track'].values[0], 'edge_spacing', [])
        # for s in spaces:
        #     ax.axhspan(*s, color='white', edgecolor=None,)
        ax.axvline(0, color='k', linestyle='dashed', alpha=0.5)
        if prev_turn is None:
            ax.set(ylim=(track_fraction[0], track_fraction[-1]), ylabel=ylabel,
                xlim=(time_bins[0], time_bins[-1]), xlabel='time around update (s)')
            ax.set_title(self.label_maps['update_type'][update_type], color=self.colors[update_type])
        if prev_turn is not None:
            ax.set(ylim=(track_fraction[0], track_fraction[-1]), ylabel=ylabel,
                xlim=(time_bins[0], time_bins[-1]), xlabel='time from initial cue (s)')
            if prev_turn == 'same':
                ax.set_title('same initial goal as prior trial', color=self.colors[update_type])
            elif prev_turn == 'diff':
                ax.set_title('opposite initial goal from prior trial', color=self.colors[update_type])
        
        ax.legend(loc='lower right', labelcolor='linecolor')
        sfig.suptitle('Decoded position', fontsize=12)

        return sfig
    
    def plot_decoding_output_heatmap_lr(self, sfig, update_type='switch', prob_value='prob_sum', feat='position',start_nan=False, time_label='t_update'):
        # load up data
        plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=[time_label])
        #if start_nan:
        #    plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=['start_time','t_update'])
        #else:
        #    plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=['t_update'])
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value)
        aligned_data = self.aggregator.select_group_aligned_data(self.aggregator.group_aligned_df, plot_groups,
                                                                 ret_df=True)


        #if aligned_data.time_label == 'start_time':
        #    aligned_data.probability[:] = np.nan
        #if trial_data.time_label == 'start_time':
        #    trial_data.probability[:] = np.nan
        
        clim = (0.02, 0.045) if feat == 'position' else (0.02, 0.04)
        #mask = np.array([np.any(np.isnan(f)) for f in aligned_data['feature']])
        probability2 = np.stack(aligned_data['probability'])
        features2 = np.stack(aligned_data['feature'])
        if start_nan:
            mask = np.isnan(features2)| (features2 < 6)#6 should remove any time from when the mouse is "sitting" at the start
            #mask = np.zeros_like(features2, dtype=bool)
            print("The maximum value from features2[:, :15] is:", np.max(features2[:, :15]))
            print("The minimum value from features2[:, :15] is:", np.min(features2[:, :15]))
            #mask[:, :15] = features2[:,:15] > 255
            expanded_mask = np.broadcast_to(mask[:, np.newaxis, :], probability2.shape)
            probability2[expanded_mask] = np.nan
    
        #nan out any probability when the mouse is before the 0.1 (assuming track starts at 0) if needed
        prob_map = np.nanmean(probability2, axis=0)
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
        if aligned_data['feature_name'].values[0] in ['choice', 'x_position']:  # if bounds are on ends of track, make home between
            bounds = [(bounds[0][1], bounds[1][0]), *bounds]
            ylabel = 'p(new choice)'
            print(ylabel,bounds)
        else:
            bounds = [(track_fraction[0], bounds[0][0]), *bounds]  # else put at start
            ylabel = 'fraction of track'
            print(ylabel,bounds)

        # plot figure
        ax = sfig.subplots(nrows=1, ncols=1)
        im_times = (time_bins[0] - np.diff(time_bins)[0] / 2, time_bins[-1] + np.diff(time_bins)[0] / 2)
        clipping_masks, images = dict(), dict()
        for b, arm in zip(bounds, ['home', 'initial', 'new']):
            mask = mpl.patches.Rectangle(xy=(0, b[0]), width=1, height=b[1] - b[0],
                                         facecolor='white', alpha=0, transform=ax.get_yaxis_transform())
            clipping_masks[arm] = mask
            images[arm] = ax.imshow(prob_map, cmap=self.colors[f'{arm}_cmap'], aspect='auto',
                                    origin='lower', vmin=clim[0], vmax=clim[1],
                                    extent=[im_times[0], im_times[-1], track_fraction[0], track_fraction[-1]])
            images[arm].set_clip_path(clipping_masks[arm])
            ticks = None if arm == 'home' else []
            label = f'prob.{self.new_line}density' if arm == 'home' else ''
            cbar = plt.colorbar(images[arm], ax=ax, pad=0.01, fraction=0.046, shrink=0.5, aspect=12,
                                ticks=ticks)
            cbar.ax.set_title(label, fontsize=8, ha='center')
            for line in b:
                ax.axhline(line, color=self.colors['phase_dividers'], alpha=0.5, linewidth=0.75, )
        ax.plot(time_bins, true_feat, color=self.colors[f'{update_type}_light'], linestyle='dotted', linewidth=1.25,
                label='actual position')

        # plot edge spaces in track
        # spaces = getattr(self.aggregator.group_aligned_df['virtual_track'].values[0], 'edge_spacing', [])
        # for s in spaces:
        #     ax.axhspan(*s, color='white', edgecolor=None,)
        ax.axvline(0, color='k', linestyle='dashed', alpha=0.5)

        ax.set(ylim=(track_fraction[0], track_fraction[-1]), ylabel=ylabel,
               xlim=(time_bins[0], time_bins[-1]), xlabel='time around update (s)')
        ax.set_title(self.label_maps['update_type'][update_type], color=self.colors[update_type])
        ax.legend(loc='lower right', labelcolor='linecolor')
        sfig.suptitle('Decoded position', fontsize=12)

        return sfig

    def plot_goal_coding(self, sfig, comparison='update_type', groups=None, prob_value='prob_sum', heatmap=False,
                         ylim=None, with_velocity=False, tags=None, update_type=['switch', 'non_update'],
                         correct_type=[1, 0], use_residuals=False, other_zones=None, use_delay=False, time_label=['t_update']):
        # load up data
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        plot_groups.update(update_type=update_type)
        plot_groups.update(time_label=time_label)
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value, other_zones=other_zones)
        sub_groups = groups if groups else 'feature_name'  # set as default to not break data down more unless needed
        if use_delay:
            trial_data = trial_data.query('update_type == "non_update"')
        elif comparison == 'correct':
            trial_data = trial_data.query('update_type == "switch"') 
        else:
            trial_data
        trial_types = update_type if comparison == 'update_type' else correct_type
        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)  #TODO
            prob_value = 'resid'

        # plot figure
        nrows = 1  # default value if no other info provided
        height_ratios = [1]
        if heatmap:
            nrows = nrows + 1
            height_ratios = [1] * nrows
        elif groups and not heatmap:
            nrows = len(trial_data[groups].unique())
            height_ratios = [1] * nrows
        elif with_velocity:
            nrows = nrows + 1
            height_ratios = [2, 1]
        ax = sfig.subplots(nrows=nrows, ncols=len(update_type), sharex=True, sharey='row', squeeze=False,
                           height_ratios=height_ratios)
        for comp, data in trial_data.groupby(comparison):
            col_id = np.argwhere(np.array(list(trial_types)) == comp)[0][0]
            if not heatmap:
                for s_name, s_data in data.groupby(sub_groups):
                    i = 2 if with_velocity else 1
                    row_id = np.argwhere(data[sub_groups].unique() == s_name)[0][0] * i
                    trial_mat = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'], columns='times',
                                             values=prob_value)
                    new_mat = trial_mat.query('choice == "switch"').to_numpy()
                    initial_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
                    time_bins = trial_mat.columns.to_numpy()

                    ax[row_id][col_id].plot(time_bins, np.nanmean(initial_mat, axis=0), color=self.colors['initial'])
                    ax[row_id][col_id].plot(time_bins, np.nanmean(new_mat, axis=0), color=self.colors['new'])
                    ax[row_id][col_id].fill_between(time_bins,
                                                    np.nanmean(new_mat, axis=0) + sem(new_mat, nan_policy='omit'),
                                                    np.nanmean(new_mat, axis=0) - sem(new_mat, nan_policy='omit'),
                                                    color=self.colors['new'], alpha=0.2)
                    ax[row_id][col_id].fill_between(time_bins, np.nanmean(initial_mat, axis=0) + sem(initial_mat,
                                                                                                     nan_policy='omit'),
                                                    np.nanmean(initial_mat, axis=0) - sem(initial_mat,
                                                                                          nan_policy='omit'),
                                                    color=self.colors['initial'],
                                                    alpha=0.2)
                    if other_zones:
                        original_mat = trial_mat.query('choice == "original"').to_numpy()
                        central_mat = trial_mat.query('choice == "central"').to_numpy()
                        local_mat = trial_mat.query('choice == "local"').to_numpy()
                        ax[row_id][col_id].plot(time_bins, np.nanmean(central_mat, axis=0), color=self.colors['central'], linestyle='dashed')
                        ax[row_id][col_id].plot(time_bins, np.nanmean(original_mat, axis=0), color=self.colors['original'])
                        ax[row_id][col_id].plot(time_bins, np.nanmean(local_mat, axis=0), color=self.colors['local'])
                        ax[row_id][col_id].fill_between(time_bins,
                                                        np.nanmean(original_mat, axis=0) + sem(original_mat, nan_policy='omit'),
                                                        np.nanmean(original_mat, axis=0) - sem(original_mat, nan_policy='omit'),
                                                        color=self.colors['original'], alpha=0.2)
                        ax[row_id][col_id].fill_between(time_bins, np.nanmean(central_mat, axis=0) + sem(central_mat,
                                                                                                         nan_policy='omit'),
                                                        np.nanmean(central_mat, axis=0) - sem(central_mat,
                                                                                              nan_policy='omit'),
                                                        color=self.colors['central'],
                                                        alpha=0.2)
                        ax[row_id][col_id].fill_between(time_bins, np.nanmean(local_mat, axis=0) + sem(local_mat,
                                                                                                         nan_policy='omit'),
                                                        np.nanmean(local_mat, axis=0) - sem(local_mat,
                                                                                              nan_policy='omit'),
                                                        color=self.colors['central'],
                                                        alpha=0.2)

                    ax[row_id][col_id].axvline(0, color='k', linestyle='dashed', alpha=0.5)
                    if ylim:
                        ax[row_id][col_id].set(ylim=ylim)

                    if with_velocity:
                        veloc_data = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'],
                                                  columns='times',
                                                  values='rotational_velocity')
                        veloc_mat = veloc_data.query('choice == "switch"').to_numpy()  # velocity data same for both
                        ax[row_id + 1][col_id].plot(time_bins, np.mean(veloc_mat, axis=0), color=self.colors['home'])
                        ax[row_id + 1][col_id].fill_between(time_bins, np.nanmean(veloc_mat, axis=0) + sem(veloc_mat),
                                                            np.nanmean(veloc_mat, axis=0) - sem(veloc_mat),
                                                            color=self.colors['new'], alpha=0.2)
                        ax[row_id + 1][col_id].axvline(0, color='k', linestyle='dashed', alpha=0.5)

                        # add markers to indicate timepoints where data switched
                        diff_initial_switch = np.nanmean(initial_mat, axis=0) - np.nanmean(new_mat, axis=0)
                        diff_veloc = np.diff(np.diff(np.nanmean(veloc_mat, axis=0), prepend=np.nan), prepend=np.nan)
                        if np.where(diff_initial_switch < 0)[0].any():
                            post_update = int(len(time_bins) / 2)
                            neural_flip_moment = np.where(diff_initial_switch < 0)[0][0] - 1
                            veloc_flip_moment = np.where(diff_veloc[post_update:] < 0)[0][0] + post_update - 1
                            x_neural = time_bins[neural_flip_moment] + np.diff(time_bins)[0] / 2
                            y_neural = np.nanmean(initial_mat, axis=0)[neural_flip_moment]
                            x_veloc = time_bins[veloc_flip_moment] + np.diff(time_bins)[0] / 2
                            y_veloc = np.nanmean(veloc_mat, axis=0)[veloc_flip_moment]
                            ax[row_id][col_id].annotate("", arrowprops=dict(arrowstyle='simple', facecolor='black'),
                                                        xy=(x_neural, y_neural), xytext=(x_neural, y_neural*1.5),
                                                        xycoords='data', textcoords='data')
                            ax[row_id + 1][col_id].annotate("",  arrowprops=dict(arrowstyle='simple', facecolor='black'),
                                                            xy=(x_veloc, y_veloc), xytext=(x_veloc, y_veloc*2),
                                                            xycoords='data', textcoords='data')
            else:
                time_bins = data['times'].unique()
                color_map = {'switch': 'new', 'initial_stay': 'initial'}
                for row_id, c in enumerate(['initial_stay', 'switch']):
                    matrix = (data
                              .query(f'choice == "{c}"')
                              .groupby([groups, 'times'])['diff_baseline']
                              .mean().unstack().to_numpy())
                    im = ax[row_id][col_id].imshow(matrix, cmap=self.colors[f'{color_map[c]}_cmap'], vmin=0, vmax=0.2,
                                                   aspect='auto', origin='lower',
                                                   extent=[time_bins.min(), time_bins.max(), 0, np.shape(matrix)[0]], )
                    ax[row_id][col_id].axvline(0, color='k', linestyle='dashed')
                    if col_id == len(label_map.keys()) - 1:
                        plt.colorbar(im, ax=ax[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                                     label='Δ prob. density from t=0')
            ax[0][col_id].set_title(label_map[comp], color=self.colors[label_map[comp]])

        # add stats bars above each plot
        ylims = ax[row_id][col_id].get_ylim()
        sfig.supylabel('prob. density', fontsize=10)
        sfig.supxlabel('time around update (s)', ha='center', fontsize=10)

        colors = [self.colors[t] for t in ['initial', 'new']]
        rainbow_text(0.05, 0.05, ['initial', 'new'], colors, ax=ax[0][col_id], size=8)

        return sfig
    
    def subsample_trials(self, data):
        #currently have this set to create the same ratio of C to I trials for switch trials as delay only trials, could change if needed
        trial_types = data['update_type'].unique().tolist()
        
         # Count correct and incorrect trials for each type
        counts = data[data['update_type'] == 'non_update']['correct'].value_counts()
        # Assign counts to variables
        num_correct_non_update = counts.get(1, 0)  # Default to 0 if missing
        num_incorrect_non_update = counts.get(0, 0)  # Default to 0 if missing
        
        best_ratio = num_correct_non_update/num_incorrect_non_update#update this ratio so it is for pfc not hpc
        subsampled_data = []
        
        trial_type = 'switch'
        correct_trials = data[(data['update_type'] == trial_type) & (data['correct'] == 1)]
        correct_trial_index = correct_trials.iloc[0::2].index  # Select every second row, starting from index 0 (all even, so all initial)
        
        incorrect_trials = data[(data['update_type'] == trial_type) & (data['correct'] == 0)]
        incorrect_trial_index = incorrect_trials.iloc[0::2].index  # Select every second row, starting from index 0 (all even, so all initial)

        # Adjust sample sizes to match the best ratio
        max_correct_sample = int(len(incorrect_trial_index) * best_ratio)
        max_incorrect_sample = int(len(correct_trial_index) / best_ratio)

        correct_sample_ids = np.random.choice(correct_trial_index, size=min(len(correct_trial_index), max_correct_sample), replace=False)
        incorrect_sample_ids = incorrect_trial_index

        sampled_trial_inds = np.concatenate([correct_sample_ids, incorrect_sample_ids])
        sampled_trial_ids = np.concatenate([sampled_trial_inds, sampled_trial_inds + 1])#adding back on the new indices for each trial
        delay_trial_ids = data.loc[data['update_type'] == 'non_update'].index
        sampled_trial_ids = np.concatenate([delay_trial_ids, sampled_trial_ids])
        subsampled_data.append(data.loc[data.index.isin(sampled_trial_ids)])
        # Combine all subsets into one DataFrame
        subsampled_data = pd.concat(subsampled_data, ignore_index=True)
        #need to still append ALL non_update trials

        return subsampled_data.sort_index()
    
    def plot_goal_coding_indtrials(self, sfig, comparison='update_type', groups=None, prob_value='prob_sum', heatmap=False,
                         ylim=None, with_velocity=False, tags=None, update_type=['switch', 'non_update'],
                         correct_type=[1], use_residuals=False, other_zones=None, use_delay=False, ave_bins=False, time_window=(0,1.5)):
        # load up data
        plot_groups = self.plot_group_comparisons_full['update_type']
        label_map = self.label_maps['update_type']
        plot_groups.update(update_type=update_type)
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value, other_zones=other_zones, time_window=(0, 2.5))
        
        sub_groups = groups if groups else 'feature_name'  # set as default to not break data down more unless needed
        if use_delay:
            trial_data = trial_data.query('update_type == "non_update"')
        elif comparison == 'correct':
            trial_data = trial_data.query('update_type == "switch"') 
        else:
            trial_data
        trial_types = update_type if comparison == 'update_type' else correct_type
        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)  #TODO
            prob_value = 'resid'

        # plot figure
        nrows = 1  # default value if no other info provided
        height_ratios = [1]
        if heatmap:
            nrows = nrows + 1
            height_ratios = [1] * nrows
        elif groups and not heatmap:
            nrows = len(trial_data[groups].unique())
            height_ratios = [1] * nrows
        elif with_velocity:
            nrows = nrows + 1
            height_ratios = [2, 1]
        ax = sfig.subplots(nrows=nrows, ncols=len(update_type), sharex=True, sharey='row', squeeze=False,
                           height_ratios=height_ratios)
        for comp, data in trial_data.groupby(comparison):
            col_id = np.argwhere(np.array(list(trial_types)) == comp)[0][0]
            if not heatmap:
                for s_name, s_data in data.groupby(sub_groups):
                    i = 2 if with_velocity else 1
                    row_id = np.argwhere(data[sub_groups].unique() == s_name)[0][0] * i
                    trial_mat = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'], columns='times',
                                             values=prob_value)
                    time_filter = trial_mat.columns[trial_mat.columns >= 0]
                    trial_mat = trial_mat[time_filter]
                    new_mat = trial_mat.query('choice == "switch"').to_numpy()
                    initial_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
                    norm_diff_mat = (new_mat - initial_mat)/(new_mat + initial_mat)#should give a normalized representation of how much initial vs new coding is happening at each bin
                    time_bins = trial_mat.columns.to_numpy()
                    # Chosen the p-value below which will indicate significance
                    alpha = 0.05
                    if ave_bins:
                        flat_norm_diff_mat = np.nanmean(norm_diff_mat, axis=1)
                        ax[row_id][col_id].set_xlabel('bin average norm_diff_mat Values')
                        ax[row_id][col_id].set_ylabel('probability')
                        ax[row_id][col_id].set_title('Distribution of bin averaged norm_diff_mat Values')
                        count = np.sum((flat_norm_diff_mat <= 0.5) & (flat_norm_diff_mat >= -0.5))
                        print(f"Number of values equal to between 0.5: {count}")
                        count2 = len(flat_norm_diff_mat)
                        print(f"total number of values: {count2}")
                        
                    else:
                        flat_norm_diff_mat = norm_diff_mat.flatten()
                        ax[row_id][col_id].set_xlabel('norm_diff_mat Values')
                        ax[row_id][col_id].set_ylabel('probability')
                        ax[row_id][col_id].set_title('Distribution of norm_diff_mat Values')
                        # Logical condition to check for 1 or -1
                        count = np.sum((flat_norm_diff_mat == 1) & (flat_norm_diff_mat == -1))

                        print(f"Number of values equal to 1 or -1: {count}")
                        count = np.sum((flat_norm_diff_mat <= 0.5) | (flat_norm_diff_mat >= -0.5))
                        print(f"Number of values equal to between 0.5: {count}")
                        count2 = len(flat_norm_diff_mat)
                        print(f"total number of values: {count2}")
                        
                    sns.histplot(flat_norm_diff_mat, bins=30, kde=True, color='skyblue', stat='probability', ax=ax[row_id][col_id])
                    flat_norm_diff_mat = flat_norm_diff_mat[~np.isnan(flat_norm_diff_mat)]#needed for lillifors test
                    ksstat, pvalue = sm.stats.diagnostic.lilliefors(flat_norm_diff_mat)
                    if pvalue > alpha:
                        result = 'Normal'
                    else:
                        result = 'NOT Normal'
                    print(f'Lilliefors: {result:>21s}')
                    print(ksstat)
                    

                    ax[row_id][col_id].axvline(0, color='k', linestyle='dashed', alpha=0.5)
                    if ylim:
                        ax[row_id][col_id].set(ylim=ylim)

                    if with_velocity:
                        veloc_data = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'],
                                                  columns='times',
                                                  values='rotational_velocity')
                        veloc_mat = veloc_data.query('choice == "switch"').to_numpy()  # velocity data same for both
                        ax[row_id + 1][col_id].plot(time_bins, np.mean(veloc_mat, axis=0), color=self.colors['home'])
                        ax[row_id + 1][col_id].fill_between(time_bins, np.nanmean(veloc_mat, axis=0) + sem(veloc_mat),
                                                            np.nanmean(veloc_mat, axis=0) - sem(veloc_mat),
                                                            color=self.colors['new'], alpha=0.2)
                        ax[row_id + 1][col_id].axvline(0, color='k', linestyle='dashed', alpha=0.5)

                        # add markers to indicate timepoints where data switched
                        diff_initial_switch = np.nanmean(initial_mat, axis=0) - np.nanmean(new_mat, axis=0)
                        diff_veloc = np.diff(np.diff(np.nanmean(veloc_mat, axis=0), prepend=np.nan), prepend=np.nan)
                        if np.where(diff_initial_switch < 0)[0].any():
                            post_update = int(len(time_bins) / 2)
                            neural_flip_moment = np.where(diff_initial_switch < 0)[0][0] - 1
                            veloc_flip_moment = np.where(diff_veloc[post_update:] < 0)[0][0] + post_update - 1
                            x_neural = time_bins[neural_flip_moment] + np.diff(time_bins)[0] / 2
                            y_neural = np.nanmean(initial_mat, axis=0)[neural_flip_moment]
                            x_veloc = time_bins[veloc_flip_moment] + np.diff(time_bins)[0] / 2
                            y_veloc = np.nanmean(veloc_mat, axis=0)[veloc_flip_moment]
                            ax[row_id][col_id].annotate("", arrowprops=dict(arrowstyle='simple', facecolor='black'),
                                                        xy=(x_neural, y_neural), xytext=(x_neural, y_neural*1.5),
                                                        xycoords='data', textcoords='data')
                            ax[row_id + 1][col_id].annotate("",  arrowprops=dict(arrowstyle='simple', facecolor='black'),
                                                            xy=(x_veloc, y_veloc), xytext=(x_veloc, y_veloc*2),
                                                            xycoords='data', textcoords='data')
            else:
                time_bins = data['times'].unique()
                color_map = {'switch': 'new', 'initial_stay': 'initial'}
                for row_id, c in enumerate(['initial_stay', 'switch']):
                    matrix = (data
                              .query(f'choice == "{c}"')
                              .groupby([groups, 'times'])['diff_baseline']
                              .mean().unstack().to_numpy())
                    im = ax[row_id][col_id].imshow(matrix, cmap=self.colors[f'{color_map[c]}_cmap'], vmin=0, vmax=0.2,
                                                   aspect='auto', origin='lower',
                                                   extent=[time_bins.min(), time_bins.max(), 0, np.shape(matrix)[0]], )
                    ax[row_id][col_id].axvline(0, color='k', linestyle='dashed')
                    if col_id == len(label_map.keys()) - 1:
                        plt.colorbar(im, ax=ax[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                                     label='Δ prob. density from t=0')
            ax[0][col_id].set_title(label_map[comp], color=self.colors[label_map[comp]])

        # add stats bars above each plot
        #ylims = ax[row_id][col_id].get_ylim()
        #sfig.supylabel('prob. density', fontsize=10)
        #sfig.supxlabel('time around update (s)', ha='center', fontsize=10)

        colors = [self.colors[t] for t in ['initial', 'new']]
        rainbow_text(0.05, 0.05, ['initial', 'new'], colors, ax=ax[0][col_id], size=8)

        return sfig

    def plot_motor_timing(self, sfig, comparison='update_type', groups=None, prob_value='prob_sum', tags=None,
                          update_type=['switch', 'non_update'], use_residuals=False):
        # load up data
        plot_groups = self.plot_group_comparisons_full[comparison]
        plot_groups.update(update_type=update_type)
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value)
        sub_groups = groups if groups else 'feature_name'  # set as default to not break data down more unless needed
        trial_data = trial_data.query('update_type == "switch"') if comparison == 'correct' else trial_data
        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)
            prob_value = 'resid'

        # plot figure
        ax = sfig.subplots(nrows=2, ncols=1, squeeze=False)
        for comp, data in trial_data.groupby(comparison):
            for s_name, s_data in data.groupby(sub_groups):
                # get veloc data
                trial_mat = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'], columns='times',
                                         values=prob_value)
                new_mat = trial_mat.query('choice == "switch"').to_numpy()
                initial_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
                time_bins = trial_mat.columns.to_numpy()
                veloc_data = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'],
                          columns='times',
                          values='rotational_velocity')
                veloc_mat = veloc_data.query('choice == "switch"').to_numpy()  # velocity data same for both

                # get switch data to use for calculations if non_update trial
                switch_data = trial_data.query(f"{comparison} == 'switch' & {sub_groups} == '{s_name}'")
                switch_veloc_data = switch_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'],
                                                      columns='times',
                                                      values='rotational_velocity')
                switch_veloc_mat = switch_veloc_data.query('choice == "switch"').to_numpy()
                switch_veloc_flip_inds = []
                for v in switch_veloc_mat:
                    post_update = int(len(time_bins) / 2)
                    if np.where(v[post_update:] < 0)[0].any():
                        switch_veloc_flip_inds.append(np.where(v[post_update:] < 0)[0][0] + post_update)
                    else:
                        switch_veloc_flip_inds.append(np.nan)

                # on a trial-by-trial basis look at initial - new event triggered average
                diff_indiv_trials = initial_mat - new_mat
                pad_width = [(0, 0), (0, np.shape(veloc_mat)[1])]
                diff_pad = np.pad(diff_indiv_trials, pad_width=pad_width, constant_values=np.nan)
                initial_pad = np.pad(initial_mat, pad_width=pad_width, constant_values=np.nan)
                new_pad = np.pad(new_mat, pad_width=pad_width, constant_values=np.nan)

                veloc_flip_inds = []
                for v in veloc_mat:
                    veloc_diff = np.diff(v, prepend=v[0])[post_update:]
                    if np.where(veloc_diff < 0)[0].any():
                        veloc_flip_inds.append(np.where(veloc_diff < 0)[0][0] + post_update)
                    else:
                        veloc_flip_inds.append(np.nan)
                sample_ind = np.arange(26 + 13) - 13  # adjust so plotting before and after

                # use the switch veloc inds to get the non_update data for plotting
                if comp == 'switch':
                    ind = np.array(veloc_flip_inds)[:, None] + sample_ind[None, :]
                elif comp == 'non_update':
                    mean_flip_ind = np.round(np.nanmean(switch_veloc_flip_inds))
                    flip_inds = [mean_flip_ind] * len(veloc_flip_inds)  # use the average switch one instead
                    ind = np.array(flip_inds)[:, None] + sample_ind[None, :]
                d_chunks = np.array([n[i.astype(int)] for i, n in zip(ind, diff_pad) if not np.isnan(i).any()])
                i_chunks = np.array([n[i.astype(int)] for i, n in zip(ind, initial_pad) if not np.isnan(i).any()])
                n_chunks = np.array([n[i.astype(int)] for i, n in zip(ind, new_pad) if not np.isnan(i).any()])

                # plot the event-triggered average for all data
                if comp == 'switch':
                    for i, chunks in enumerate([d_chunks, i_chunks, n_chunks]):
                        name = ['difference', 'initial', 'new'][i]
                        ylabel = ['initial - new prob. density', 'prob. density', 'prob. density'][i]
                        # ylim = [(-0.1, 0.1), (0.06, 0.18), (0.06, 0.18)][i]
                        c = 0
                        r = 0 if name == 'difference' else 1
                        color = [self.colors['control'], self.colors['initial'], self.colors['new']][i]
                        times_to_plot = np.concatenate([time_bins,
                                                        time_bins + time_bins.max() - time_bins.min()])[:np.shape(chunks)[1]]
                        ax[r][c].plot(times_to_plot, np.nanmean(chunks, axis=0), color=color, label=name)
                        ax[r][c].fill_between(times_to_plot, np.nanmean(chunks, axis=0) + self.nan_sem(chunks),
                                                        np.nanmean(chunks, axis=0) - self.nan_sem(chunks),
                                                        color=color, alpha=0.2)
                        ax[r][c].axvline(0, color='k', linestyle='dashed', alpha=0.5)
                        ax[r][c].set(title=f'{comp} - motor response triggered average', ylabel=ylabel, # ylim=ylim,
                                     xlabel='time around motor response (s)')
                        ax[r][c].legend()

        return sfig

    def plot_nonlocal_coding_stats(self, sfig, comparison='choice', prob_value='prob_sum', title='', tags='',
                               time_window=(0, 1.5), use_zscores=False, other_zones=dict(), stripplot=False,
                               update_type=['switch'], use_residuals=False, use_delay=False, ylim=None,
                               time_label='t_update', prev_turn=None, prospective_reps=False, all_trials=False):
        decoding_measures = 'zscore_prob' if use_zscores else prob_value
        choice_mapping = {z: z for z in other_zones}
        choice_mapping['initial_stay'] = 'initial'
        choice_mapping['switch'] = 'new'

        # get data
        label_map = self.label_maps['choice']
        plot_groups = dict(update_type = update_type, turn_type = [1, 2], correct = [1], time_label = [time_label])
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       n_time_bins=11, time_window=(-2.5, 2.5),
                                                                       prob_value=prob_value, other_zones=other_zones, prospective_reps=prospective_reps)

        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)
            decoding_measures = 'resid'

        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'time_label', 'feature_name', 
                        'choice']
        data_for_stats = (trial_data
                          .query(
            f'times_binned > {time_window[0]} & times_binned < {time_window[-1]}')  # only look at first 1.5 seconds
                          .groupby(groupby_cols)[[decoding_measures, 'diff_baseline', 'error']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        data_for_stats.reset_index(inplace=True)
        data_for_stats['choice'] = data_for_stats['choice'].map(choice_mapping)
        #data_for_stats[comparison] = data_for_stats[comparison].map(label_map)#renaming non update to delay only
        
        # setup stats - group variables, pairs to compare, and levels of hierarchical data
        var = f'{decoding_measures}_mean'
        group = 'choice'
        group_list = data_for_stats['choice'].unique()#zones that are being decoded (what i want to be compared)
        
        #combo_list = [label_map[g] for g in plot_groups[comparison]]#switch back to comparison in plot groups later
        #combos = list(itertools.combinations(combo_list, r=2))
        pairs = [(('central',),('initial',)),(('central',),('new',)),(('initial',),('new',))]#check formatting
        sess_averages = data_for_stats.groupby(['session_id', comparison])[var].mean().reset_index()
        stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                    approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
        stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice'],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')

        # plot data
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[3, 1]))
        colors = [self.colors[t] for t in list(label_map.values())]

        if other_zones == ['central']:
            category_order = ['central','initial','new']#'local','central',
        else:
            category_order = ['initial','new']
        common_kwargs = dict(data=sess_averages, x=comparison, y=var, ax=ax[0], errwidth=3,
                             order=category_order, join=False, dodge=(0.8 - 0.8 / 3), )#make sess_averages
        ax[0] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)#, order=category_order
        # common_kwargs = dict(data=diff_data, x=group, y=var, hue=comparison, ax=ax[0],
        #                      hue_order=list(label_map.values()), errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
        # ax[0] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5, order=category_order)
        ax[0] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)#, order=category_order
        ax[0].set(xlabel=f'goal location', ylabel=f'prob. density after update onset')
        #ax[0].get_legend().remove()
        rainbow_text(0.5, 0.9, list(label_map.values()), colors, ax=ax[0], size=8)
        if ylim:
            ax[0].set(ylim=ylim)

        # add stats annotations
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "{var}"')
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]
        diff_pairs = [(a[0], b[0]) for a, b in pairs]

        # annot = Annotator(ax[0], pairs=diff_pairs, data=data_for_stats, x=group, y=var, hue=comparison,
        #                   hue_order=list(label_map.values()), order=category_order)
        # annot.new_plot(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
        #                hue_order=list(label_map.values()), order=category_order)
        # (annot
        #  .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
        #  .set_pvalues(pvalues=pvalues)
        #  .annotate(line_offset=0.1, line_offset_to_group=0.025))

        # get error differences for reporting purposes
        # bins = self.aggregator.group_aligned_df['bins'].values[0]
        # data_for_stats['error_mean'] = (data_for_stats['error_mean'] - np.min(bins)) / (np.max(bins) - np.min(bins))
        # data_for_stats = data_for_stats.query('choice == "initial"')
        # stats.run(data_for_stats, dependent_vars=['error_mean'], group_vars=['choice', comparison],
        #           pairs=pairs, filename=f'decoding_error_stats_{comparison}_{tags}')

        sfig.suptitle(f'{title} goal representation quantification', fontsize=12)
        return sfig
    
    def plot_goal_coding_stats(self, sfig, comparison='update_type', prob_value='prob_sum', title='', tags='',
                               time_window=(0, 1.5), use_zscores=False, other_zones=dict(), stripplot=False,
                               update_type=['switch', 'non_update'], use_residuals=False, use_delay=False, ylim=None, subsample=False,
                               time_label=['t_update'], prev_turn=None, prospective_reps=False, all_trials=False, half=None, rebaseline=False):
        decoding_measures = 'zscore_prob' if use_zscores else prob_value
        choice_mapping = {z: z for z in other_zones}
        choice_mapping['initial_stay'] = 'initial'
        choice_mapping['switch'] = 'new'

        # get data
        plot_groups = self.plot_group_comparisons_full[comparison]
        plot_groups.update(update_type=update_type)
        plot_groups.update(time_label=time_label)
        if all_trials==True: 
            plot_groups.update(correct=[1, 0]) 
        label_map = self.label_maps[comparison]
        label_map = {k: label_map[k] for k in plot_groups[comparison]}#need to have comparison have this variable, look at next
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       n_time_bins=11, time_window=(-2.5, 2.5),
                                                                       prob_value=prob_value, other_zones=other_zones, prospective_reps=prospective_reps, half=half)
        if use_delay:
            trial_data = trial_data.query('update_type == "non_update"')
        elif comparison == 'correct':
            trial_data = trial_data.query('update_type == "switch"')
         
        else:
            trial_data

        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)
            decoding_measures = 'resid'

        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'time_label', 'feature_name', 
                        'choice']
        if rebaseline:
            # Step 1: Define the time range for the pre-window (-1.5 to 0 seconds) for prob_sum
            pre_window_means = (trial_data
                .query('times_binned >= -1.5 & times_binned < 0')  # Pre-window time range
                .groupby(groupby_cols)[prob_value]  # Group by the relevant columns
                .mean()  # Calculate the mean over the pre-window time range
                .reset_index()
                .rename(columns={prob_value: 'prob_sum_pre'}))  # Rename to prob_sum_pre
            # Step 2: Merge the pre-window means back into the full dataset
            trial_data1 = trial_data.merge(pre_window_means, on=groupby_cols, how='left')
            # Step 3: Rename original prob_sum to og_prob_sum
            trial_data1 = trial_data1.rename(columns={prob_value: 'og_prob_sum'})
            # Step 4: Normalize the prob_sum by subtracting the pre-window mean and rename it to prob_sum
            trial_data1[prob_value] = trial_data1['og_prob_sum'] - trial_data1['prob_sum_pre']
            # Now the 'prob_sum_normalized' column contains the normalized values while all other data remains intact.
            trial_data = trial_data1
            
        data_for_stats = (trial_data
                          .query(
            f'times_binned > {time_window[0]} & times_binned < {time_window[-1]}')  # only look at first 1.5 seconds
                          .groupby(groupby_cols)[[decoding_measures, 'diff_baseline', 'error']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        data_for_stats.reset_index(inplace=True)
        if subsample:#subsampling trials to get equivalent correct/incorrect ratios, currently only works to make switch = delay_only
            data_for_stats = self.subsample_trials(data_for_stats)
        data_for_stats['choice'] = data_for_stats['choice'].map(choice_mapping)
        data_for_stats[comparison] = data_for_stats[comparison].map(label_map)
        diff_data = (data_for_stats.pivot(index=groupby_cols[:-1], columns=['choice'],
                                          values=[f'{decoding_measures}_mean'])
                     .reset_index())
        diff_data[('initial_vs_new', '')] = diff_data[(f'{decoding_measures}_mean', 'initial')] - \
                                            diff_data[(f'{decoding_measures}_mean', 'new')]
        diff_data = diff_data.droplevel(1, axis=1)

        # setup stats - group variables, pairs to compare, and levels of hierarchical data
        var = f'{decoding_measures}_mean'
        group = 'choice'
        group_list = data_for_stats['choice'].unique()
        combo_list = [label_map[g] for g in plot_groups[comparison]]
        combos = list(itertools.combinations(combo_list, r=2))
        pairs = [((g, c[0]), (g, c[1],)) for c in combos for g in group_list]
        if comparison == 'choice':
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice'],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')
        else:
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='manuscript')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice', comparison],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')

        # plot data
        #fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
        if prob_value == 'prob_over_chance':
            sfigs = sfig.subfigures(nrows=2, ncols=1, height_ratios=[1, 1])
            ax_top = sfigs[0].subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[3, 1]))
            ax_bottom = sfigs[1].subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[1, 1]))
            # Combine all axes into a single flat list
            ax = [ax_top[0], ax_top[1], ax_bottom[0], ax_bottom[1]]
        else:
            ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[3, 1]))
            # Flatten for consistency
            ax = [ax[0], ax[1]]
        colors = [self.colors[t] for t in list(label_map.values())]

        sess_averages = data_for_stats.groupby(['session_id', group, comparison])[var].mean().reset_index()
        common_kwargs = dict(data=sess_averages, x=group, y=var, hue=comparison, ax=ax[0],
                             hue_order=list(label_map.values()), errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
        if comparison == 'representations':
            common_kwargs.update(hue=group)
        if stripplot:
            ax[0] = sns.stripplot(data=sess_averages, x=group, y=var, hue=comparison, ax=ax[0],
                                  hue_order=list(label_map.values()), zorder=1, jitter=True,
                                  palette=[self.colors['home_medium']] * len(colors), alpha=0.4, dodge=True )#legend=False,
            if comparison == 'representations':
                common_kwargs.update(hue=group)
        if other_zones == ['central']:
            category_order = ['central','initial','new']#'local','central',
        else:
            category_order = ['initial','new']
        ax[0] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5, order=category_order)
        ax[0] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None, order=category_order)
        ax[0].set(xlabel=f'goal location', ylabel=f'prob. density after update onset')
        ax[0].get_legend().remove()
        rainbow_text(0.5, 0.9, list(label_map.values()), colors, ax=ax[0], size=8)
        if ylim:
            ax[0].set(ylim=ylim)

        # add stats annotations
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "{var}"')
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]
        annot = Annotator(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                          hue_order=list(label_map.values()), order=category_order)
        annot.new_plot(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                       hue_order=list(label_map.values()), order=category_order)
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))
        
        #saving source data
        df_export = sess_averages[[group, var, comparison]]
        save_dir = self.results_io.get_source_data_path()
        filename = f'pointplot_data_{tags}_a.xlsx'
        df_export.to_excel(save_dir / filename, index=False)

        sess_averages = diff_data.groupby(['session_id', comparison])['initial_vs_new'].mean().reset_index()
        if stripplot:
            ax[1] = sns.stripplot(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], zorder=1, 
                                  color=self.colors['home_medium'], order=list(label_map.values()), alpha=0.4)#legend=False,
        common_kwargs = dict(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], errwidth=3,
                             order=list(label_map.values()), join=False, dodge=(0.8 - 0.8 / 3), )
        ax[1] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
        ax[1] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)
        ax[1].set(ylabel=f'initial - new prob. density')
        ax[1].axhline(0, color='k', linestyle='dashed', alpha=0.5)

        combos = list(itertools.combinations(combo_list, r=2))
        diff_pairs = [((c[0],), (c[1],)) for c in combos]
        stats.run(diff_data, dependent_vars=['initial_vs_new'], group_vars=[comparison], pairs=diff_pairs,
                  filename=f'goal_diff_stats_{comparison}_{tags}')
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "initial_vs_new"')
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in diff_pairs]
        annot = Annotator(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
                          order=list(label_map.values()))
        annot.new_plot(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
                       order=list(label_map.values()),
                       )
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))
        
        #saving source data
        df_export = sess_averages[['initial_vs_new', comparison]]
        save_dir = self.results_io.get_source_data_path()
        filename = f'pointplot_data_{tags}_b.xlsx'
        df_export.to_excel(save_dir / filename, index=False)

        # get error differences for reporting purposes
        bins = self.aggregator.group_aligned_df['bins'].values[0]
        data_for_stats['error_mean'] = (data_for_stats['error_mean'] - np.min(bins)) / (np.max(bins) - np.min(bins))
        data_for_stats1 = data_for_stats.query('choice == "initial"')
        stats.run(data_for_stats1, dependent_vars=['error_mean'], group_vars=['choice', comparison],
                  pairs=pairs, filename=f'decoding_error_stats_{comparison}_{tags}')

        # get significant difference from zero for reporting purposes
        sess_averages = (diff_data
                         .groupby(['animal', 'session_id', 'update_type', 'correct', 'region'])
                         .mean()
                         .reset_index())
        pairs = [[[c]] for c in combo_list]
        stats = Stats(levels=['animal', 'session_id'], results_io=self.results_io,
                      approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='manuscript')
        stats.run(sess_averages, dependent_vars=['initial_vs_new'], group_vars=[comparison], pairs=pairs,
                  filename=f'goal_diff_stats_from_0_{comparison}_{tags}')
        
        if prob_value == 'prob_over_chance':#need to limit diff_data to switch only trials
            # sess_averages = (data_for_stats[data_for_stats['update_type'] == 'switch']
            #                  .groupby(['animal', 'session_id', 'update_type', 'correct', 'region','choice'])
            #                  .mean()
            #                  .reset_index())
            data_for_stats = data_for_stats[data_for_stats['update_type'] == 'switch']
            if not rebaseline: 
                data_for_stats['prob_over_chance_mean_shifted'] = data_for_stats['prob_over_chance_mean'] - 1
                var = 'prob_over_chance_mean_shifted'
            else:
                var = 'prob_over_chance_mean'
            pairs = [[[c]] for c in category_order]#should give zones instead of trial type
            stats = Stats(levels=['animal', 'session_id'], results_io=self.results_io,
                          approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='response', alternatives=['greater'])
            stats.run(data_for_stats, dependent_vars=[var], group_vars='choice', pairs=pairs,
                      filename=f'goal_diff_stats_from_1_wilc_{tags}')
            new_mat = data_for_stats.query('choice == "new"')[var].to_numpy()
            initial_mat = data_for_stats.query('choice == "initial"')[var].to_numpy()
            initial_ratio = np.sum((initial_mat <=0))/np.size(initial_mat)
            print(f"ratio of initial values below 0: {initial_ratio}")
            new_ratio = np.sum((new_mat <=0))/np.size(new_mat)
            print(f"ratio of new values below 0: {new_ratio}")
            # Plot the histogram for initial_mat
            ax[2].hist(initial_mat.flatten(), bins=30, color='#2459bd', alpha=0.7, edgecolor='black')
            ax[2].set_title('Histogram of Initial Mat')
            ax[2].set_xlabel('Values')
            ax[2].set_ylabel('Frequency')
            # Plot the histogram for new_mat
            ax[3].hist(new_mat.flatten(), bins=30, color='#b01e70', alpha=0.7, edgecolor='black')
            ax[3].set_title('Histogram of New Mat')
            ax[3].set_xlabel('Values')
            # Adjust layout
            plt.tight_layout()
            #trying zscores
            # Calculate mean and standard deviation
            mean_initial = np.nanmean(initial_mat)  # Use nanmean to ignore NaN values, if any
            std_initial = np.nanstd(initial_mat)    # Use nanstd to ignore NaN values, if any
            mean_new = np.nanmean(new_mat)  # Use nanmean to ignore NaN values, if any
            std_new = np.nanstd(new_mat)    # Use nanstd to ignore NaN values, if any
            # Z-score transformation
            # Assign the z-scored values to a new column in the DataFrame
            # data_for_stats['prob_over_chance_mean_shifted_zscore'] = np.nan
            # data_for_stats.loc[data_for_stats['choice'] == 'initial', 'prob_over_chance_mean_shifted_zscore'] = (
            #     (data_for_stats.loc[data_for_stats['choice'] == 'initial', 'prob_over_chance_mean_shifted'] - mean_initial)
            #     / std_initial
            # )
            # # Z-score for 'new'
            # data_for_stats.loc[data_for_stats['choice'] == 'new', 'prob_over_chance_mean_shifted_zscore'] = (
            #     (data_for_stats.loc[data_for_stats['choice'] == 'new', 'prob_over_chance_mean_shifted'] - mean_new)
            #     / std_new
            # )
            # stats = Stats(levels=['animal', 'session_id'], results_io=self.results_io,
            #               approaches=['traditional'], tests=['anova'], results_type='response')
            # stats.run(data_for_stats, dependent_vars=['prob_over_chance_mean_shifted_zscore'], group_vars='choice', pairs=pairs,
            #           filename=f'goal_diff_stats_from_1_ttest_{tags}')
        sfig.suptitle(f'{title} goal representation quantification', fontsize=12)
        return sfig
    
    def plot_goal_coding_stats_regionchoice(self, sfig, comparison='update_type', prob_value='prob_sum', title='', tags='',
                               time_window=(0, 1.5), use_zscores=False, other_zones=dict(), stripplot=False,
                               update_type=['switch', 'non_update'], use_residuals=False, use_delay=False, ylim=None,
                               time_label='t_update', prev_turn=None, prospective_reps=False):
        decoding_measures = 'zscore_prob' if use_zscores else prob_value
        choice_mapping = {z: z for z in other_zones}
        choice_mapping['initial_stay'] = 'initial'
        choice_mapping['switch'] = 'new'

        # get data
        if comparison == 'cue_type':
            plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=[time_label])
            #need to do something for label map potentially in here
        elif comparison == 'region':#can we combine with above? check how plot_groups is used. might have to make something to pull regions idk
            plot_groups = dict(update_type = update_type, turn_type = [1, 2], correct = [1], time_label = [time_label])
            label_map = {('CA1',):'CA1', ('PFC',):'PFC'}
        else:
            plot_groups = self.plot_group_comparisons_full[comparison]
            plot_groups.update(update_type=update_type)
            label_map = self.label_maps[comparison]
            label_map = {k: label_map[k] for k in plot_groups[comparison]}#need to have comparison have this variable, look at next
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       n_time_bins=11, time_window=(-2.5, 2.5),
                                                                       prob_value=prob_value, other_zones=other_zones, prospective_reps=prospective_reps)
        if use_delay:
            trial_data = trial_data.query('update_type == "non_update"')
        elif comparison == 'correct':
            trial_data = trial_data.query('update_type == "switch"')
         
        else:
            trial_data
            
        if prev_turn is not None:
            file_path = r"Y:\singer\Steph\Code\update-project\results\decoding\intermediate_data\combined_data.xlsx"
            df = pd.read_excel(file_path)#loading in the excel spreadsheet with all of the trial types, made from behavior data where all trials still present with info
            combined_data = df.sort_values(by=['animal','session_id','trial_id'])
            combined_data['prev_turn_type'] = combined_data.groupby(['animal','session_id'])['turn_type'].shift(1)#making prev_turn_type have the value of the previous 
            if prev_turn == 'diff':
                # Filter combined_data to get trials where the previous turn_type was 'right'
                right_right_turn_trials = combined_data[(combined_data['prev_turn_type'] == 'right')#since update switches
                                                        & (combined_data['turn_type'] == 'right')
                                                        | (combined_data['prev_turn_type'] == 'left')
                                                        & (combined_data['turn_type'] == 'left')]#turn_type = 2 in aligned_data, also the turn that is flipped
                trial_data = trial_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            elif prev_turn == 'same':#refers to if mouse is turning the same way prev and initially told to do the same turn. is flipped because these are switch trials
                # Filter combined_data to get trials where the previous turn_type was 'right'
                right_right_turn_trials = combined_data[(combined_data['prev_turn_type'] == 'right')
                                                        & (combined_data['turn_type'] == 'left')
                                                        | (combined_data['prev_turn_type'] == 'left')
                                                        & (combined_data['turn_type'] == 'right')]#turn_type = 2 in aligned_data, also the turn that is flipped
                trial_data = trial_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            
        #trial_data = trial_data.query('update_type == "switch"') if comparison == 'correct' else trial_data
        #trial_data = trial_data.query('update_type == "switch"') if comparison == 'choice' else trial_data
        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)
            decoding_measures = 'resid'

        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'time_label', 'feature_name', 
                        'choice']
        data_for_stats = (trial_data
                          .query(
            f'times_binned > {time_window[0]} & times_binned < {time_window[-1]}')  # only look at first 1.5 seconds
                          .groupby(groupby_cols)[[decoding_measures, 'diff_baseline', 'error']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        data_for_stats.reset_index(inplace=True)
        data_for_stats['choice'] = data_for_stats['choice'].map(choice_mapping)
        data_for_stats[comparison] = data_for_stats[comparison].map(label_map)#renaming non update to delay only
        diff_data = (data_for_stats.pivot(index=groupby_cols[:-1], columns=['choice'],
                                          values=['error_mean'])#f'{decoding_measures}_mean'
                     .reset_index())
        diff_data[('initial_vs_new', '')] = diff_data[('error_mean', 'initial')] - \
                                            diff_data[('error_mean', 'new')]##f'{decoding_measures}_mean' in both ()
        diff_data = diff_data.droplevel(1, axis=1)

        # setup stats - group variables, pairs to compare, and levels of hierarchical data
        var = 'error_mean'#f'{decoding_measures}_mean'
        group = 'choice'
        group_list = data_for_stats['choice'].unique()#zones that are being decoded (what i want to be compared)
        plot_groups['region'] = [('CA1',), ('PFC',)]
        combo_list = [label_map[g] for g in plot_groups[comparison]]#switch back to comparison in plot groups later
        combos = list(itertools.combinations(combo_list, r=2))
        pairs = [((g, c[0]), (g, c[1],)) for c in combos for g in group_list]
        if comparison == 'choice':
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice'],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')
        else:
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='manuscript')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice', comparison],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')

        # plot data
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[3, 1]))
        colors = [self.colors[t] for t in list(label_map.values())]

        sess_averages = data_for_stats.groupby(['session_id', group, comparison])[var].mean().reset_index()
        common_kwargs = dict(data=sess_averages, x=group, y=var, hue=comparison, ax=ax[0],
                             hue_order=list(label_map.values()), errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
        if comparison == 'representations':
            common_kwargs.update(hue=group)
        if stripplot:
            ax[0] = sns.stripplot(data=sess_averages, x=group, y=var, hue=comparison, ax=ax[0],
                                  hue_order=list(label_map.values()), zorder=1, jitter=True,
                                  palette=[self.colors['home_medium']] * len(colors), alpha=0.4, dodge=True )#legend=False,
            if comparison == 'representations':
                common_kwargs.update(hue=group)
        category_order = ['initial','new']#'local','central',
        ax[0] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5, order=category_order)
        ax[0] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None, order=category_order)
        ax[0].set(xlabel=f'goal location', ylabel=f'prob. density after update onset')
        ax[0].get_legend().remove()
        rainbow_text(0.5, 0.9, list(label_map.values()), colors, ax=ax[0], size=8)
        if ylim:
            ax[0].set(ylim=ylim)

        # add stats annotations
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "{var}"')
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]
        annot = Annotator(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                          hue_order=list(label_map.values()), order=category_order)
        annot.new_plot(ax[0], pairs=pairs, data=data_for_stats, x=group, y=var, hue=comparison,
                       hue_order=list(label_map.values()), order=category_order)
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))

        # sess_averages = diff_data.groupby(['session_id', comparison])['initial_vs_new'].mean().reset_index()
        # if stripplot:
        #     ax[1] = sns.stripplot(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], zorder=1, 
        #                           color=self.colors['home_medium'], order=list(label_map.values()), alpha=0.4)#legend=False,
        # common_kwargs = dict(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], errwidth=3,
        #                      order=list(label_map.values()), join=False, dodge=(0.8 - 0.8 / 3), )
        # ax[1] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
        # ax[1] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)
        # ax[1].set(ylabel=f'initial - new prob. density')
        # ax[1].axhline(0, color='k', linestyle='dashed', alpha=0.5)

        # combos = list(itertools.combinations(combo_list, r=2))
        # diff_pairs = [((c[0],), (c[1],)) for c in combos]
        # stats.run(diff_data, dependent_vars=['initial_vs_new'], group_vars=[comparison], pairs=diff_pairs,
        #           filename=f'goal_diff_stats_{comparison}_{tags}')
        # stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
        #                                   f'& variable == "initial_vs_new"')
        # stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        # pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in diff_pairs]
        # annot = Annotator(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
        #                   order=list(label_map.values()))
        # annot.new_plot(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
        #                order=list(label_map.values()),
        #                )
        # (annot
        #  .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
        #  .set_pvalues(pvalues=pvalues)
        #  .annotate(line_offset=0.1, line_offset_to_group=0.025))

        # get error differences for reporting purposes
        bins = self.aggregator.group_aligned_df['bins'].values[0]
        data_for_stats['error_mean'] = (data_for_stats['error_mean'] - np.min(bins)) / (np.max(bins) - np.min(bins))
        data_for_stats = data_for_stats.query('choice == "initial"')
        stats.run(data_for_stats, dependent_vars=['error_mean'], group_vars=['choice', comparison],
                  pairs=pairs, filename=f'decoding_error_stats_{comparison}_{tags}')

        # get significant difference from zero for reporting purposes
        # sess_averages = (diff_data
        #                  .groupby(['animal', 'session_id', 'update_type', 'correct', 'region'])
        #                  .mean()
        #                  .reset_index())
        # pairs = [[[c]] for c in combo_list]
        # stats = Stats(levels=['animal', 'session_id'], results_io=self.results_io,
        #               approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='manuscript')
        # stats.run(sess_averages, dependent_vars=['initial_vs_new'], group_vars=[comparison], pairs=pairs,
        #           filename=f'goal_diff_stats_from_0_{comparison}_{tags}')

        sfig.suptitle(f'{title} goal representation quantification', fontsize=12)
        return sfig
    
    def plot_goal_coding_stats_nomap(self, sfig, comparison='update_type', prob_value='prob_sum', title='', tags='',
                               time_window=(0, 1.5), use_zscores=False, other_zones=dict(), stripplot=False,
                               update_type=['switch', 'non_update'], use_residuals=False, use_delay=False, ylim=None,
                               time_label='t_update', prev_turn=None, half=None):
        decoding_measures = 'zscore_prob' if use_zscores else prob_value
        choice_mapping = {z: z for z in other_zones}
        choice_mapping['initial_stay'] = 'initial'
        choice_mapping['switch'] = 'new'
        if comparison == 'region':
            
            label_map = {('CA1',):'CA1', ('PFC',):'PFC'}
        else:  
            label_map = {'start_time':'start cue', 't_update':'update cue'}

        # get data
        if comparison == 'time_label':
            plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=time_label)
            #need to do something for label map potentially in here
            time_window=(0, 1.5)
        elif comparison == 'region':
            plot_groups = dict(update_type=update_type, turn_type=[1, 2], correct=[1], time_label=time_label)
            time_window=(-2.5, 0)
        else:
            plot_groups = self.plot_group_comparisons_full[comparison]
            plot_groups.update(update_type=update_type)
            
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       n_time_bins=11, time_window=(-2.5, 2.5),
                                                                       prob_value=prob_value, other_zones=other_zones, half=half)
        # if use_delay:
        #     trial_data = trial_data.query('update_type == "non_update"')
        # elif comparison == 'correct':
        #     trial_data = trial_data.query('update_type == "switch"')
         
        # else:
        #     trial_data
            
        if prev_turn is not None:
            file_path = r"Y:\singer\Steph\Code\update-project\results\decoding\intermediate_data\combined_data.xlsx"
            df = pd.read_excel(file_path)#loading in the excel spreadsheet with all of the trial types, made from behavior data where all trials still present with info
            combined_data = df.sort_values(by=['animal','session_id','trial_id'])
            combined_data['prev_turn_type'] = combined_data.groupby(['animal','session_id'])['turn_type'].shift(1)#making prev_turn_type have the value of the previous 
            if prev_turn == 'diff':
                # Filter combined_data to get trials where the previous turn_type was 'right'
                right_right_turn_trials = combined_data[(combined_data['prev_turn_type'] == 'right')#since update switches
                                                        & (combined_data['turn_type'] == 'right')
                                                        | (combined_data['prev_turn_type'] == 'left')
                                                        & (combined_data['turn_type'] == 'left')]#turn_type = 2 in aligned_data, also the turn that is flipped
                trial_data = trial_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            elif prev_turn == 'same':#refers to if mouse is turning the same way prev and initially told to do the same turn. is flipped because these are switch trials
                # Filter combined_data to get trials where the previous turn_type was 'right'
                right_right_turn_trials = combined_data[(combined_data['prev_turn_type'] == 'right')
                                                        & (combined_data['turn_type'] == 'left')
                                                        | (combined_data['prev_turn_type'] == 'left')
                                                        & (combined_data['turn_type'] == 'right')]#turn_type = 2 in aligned_data, also the turn that is flipped
                trial_data = trial_data.merge(right_right_turn_trials[['trial_id','session_id']],
                                                  on=['trial_id','session_id'],
                                                  how='inner')
            
        #trial_data = trial_data.query('update_type == "switch"') if comparison == 'correct' else trial_data
        #trial_data = trial_data.query('update_type == "switch"') if comparison == 'choice' else trial_data
        if use_residuals:
            trial_data = self.aggregator.get_residuals(trial_data, by_session=True)
            decoding_measures = 'resid'

        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'time_label', 'feature_name',
                        'choice']
        data_for_stats = (trial_data
                          .query(
            f'times_binned > {time_window[0]} & times_binned < {time_window[-1]}')  # only look at first 1.5 seconds
                          .groupby(groupby_cols)[[decoding_measures, 'diff_baseline', 'error']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        data_for_stats.reset_index(inplace=True)
        data_for_stats['choice'] = data_for_stats['choice'].map(choice_mapping)
        if comparison == 'region':
            data_for_stats['region'] = data_for_stats['region'].map(label_map)
            trial_data['region'] = trial_data['region'].map(label_map)
        
        diff_data = (data_for_stats.pivot(index=groupby_cols[:-1], columns=['choice'],
                                          values=[f'{decoding_measures}_mean'])
                     .reset_index())
        diff_data[('initial_vs_new', '')] = diff_data[(f'{decoding_measures}_mean', 'initial')] - \
                                            diff_data[(f'{decoding_measures}_mean', 'new')]
        diff_data = diff_data.droplevel(1, axis=1)
        #this is where I need to start checking things based on my region stats
        if comparison == 'region':
            diff_pairs = [(('initial', 'CA1'),('initial', 'PFC')),(('new', 'CA1'),('new', 'PFC'))]
            groups = ['choice','region']
            pairs = [('CA1', 'PFC')]
            hues = 'region'
        else:
            diff_pairs = [(('initial', 'start_time'),('initial', 't_update')),(('new', 'start_time'),('new', 't_update'))]
            groups = ['choice','time_label']
            pairs = [(('start_time',), ('t_update',))]
            hues = 'time_label'
        var = f'{decoding_measures}_mean'
        stats = Stats(levels=['animal', 'session_id','trial_id'], results_io=self.results_io,
                      approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
        stats.run(data_for_stats, dependent_vars=[var], group_vars=groups, pairs=diff_pairs,
                  filename=f'goal_coding_stats_{comparison}_{tags}')#pairs=diff_pairs,
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "{var}"')
        
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0] if isinstance(x, list) else x)#fix for the weird formatting issue
        pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in diff_pairs]
        #pvalues = [stats_data['p_val'].to_numpy()[0]]
        
        #plot data
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[3, 1]))
        if comparison == 'region':
            colors = [self.colors[t] for t in list(label_map.values())]
            sess_averages = data_for_stats.groupby(['session_id', 'choice', 'region'])[var].mean().reset_index()
            hue_ord = list(label_map.values())
        else:
            colors = [self.colors[t] for t in ('initial','new')]
            sess_averages = data_for_stats.groupby(['session_id', 'choice', 'time_label'])[var].mean().reset_index()
            hue_ord = list(label_map.keys())
            # diff_data = (diff_data
            #        .groupby(['animal', 'session_id', 'trial_id', 'time_label'])
            #        [var].mean()
            #        .reset_index())

        
        common_kwargs = dict(data=sess_averages, x='choice', y=var, hue=comparison, ax=ax[0],
                             hue_order=hue_ord, errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
        
        category_order = ['initial','new']#'local','central',
        sns.pointplot(**common_kwargs, palette=colors, order=category_order, scale=1.5)#both used to be ax[0] = 
        sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None, order=category_order)
        ax[0].set(xlabel=f'goal location', ylabel=f'prob. density before update onset')
        #ax[0].get_legend().remove()
        rainbow_text(0.5, 0.9, list(label_map.keys()), colors, ax=ax[0], size=8)
        if ylim:
            ax[0].set(ylim=ylim)
        decoding_measures = 'prob_sum_mean'
        ##stats annotations
        annot = Annotator(ax[0], pairs=diff_pairs, data=data_for_stats, x='choice', y=decoding_measures, hue=comparison,
                          hue_order=hue_ord)#pairs=combos, before data= #order=list(label_map.values())
        annot.new_plot(ax[0], pairs=diff_pairs, data=data_for_stats, x='choice', y=decoding_measures, hue=comparison,
                       hue_order=hue_ord, )
                       #pairs=combos, before data=order=list(label_map.values()
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))

        sfig.suptitle(f'{title} goal arm cue quantification', fontsize=12)
        if comparison == 'region':
            return sfig
        
        # setup stats - group variables, pairs to compare, and levels of hierarchical data
        # var = f'{decoding_measures}'
        group = comparison
        group_list = data_for_stats['choice'].unique()#zones that are being decoded (what i want to be compared)
        #pairs = [('CA1', 'PFC')]#make sure label map applied correctly or this will break
        if comparison == 'choice':
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
            stats.run(diff_data, dependent_vars=[var], group_vars=['choice'],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')
        elif comparison == 'time_label':
            #order = list(time_label)
            combos = [('start_time', 't_update')]
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
            stats.run(diff_data, dependent_vars=['initial_vs_new'], group_vars=['time_label'],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_collapse_stats_{comparison}_{tags}')
        elif comparison == 'region':
            order=list(label_map.values())
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
            stats.run(diff_data, dependent_vars=[var], group_vars=['choice', comparison],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')
        else:
            stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                        approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='manuscript')
            stats.run(data_for_stats, dependent_vars=[var], group_vars=['choice', comparison],#choice is zones (which i want) and comparison is trial type (which i am dropping and replacing with nothing)
                    pairs=pairs, filename=f'goal_coding_stats_{comparison}_{tags}')

        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "initial_vs_new"')
        #plot data initial-new
        sess_averages = diff_data.groupby(['session_id', 'region','time_label'])['initial_vs_new'].mean().reset_index()
        #comparison = 'region'
        if stripplot:
            ax[1] = sns.stripplot(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], zorder=1, 
                                  color=self.colors['home_medium'], order=list(label_map.keys()), alpha=0.4)#legend=False,
        common_kwargs = dict(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], errwidth=3,
                             order=['start_time', 't_update'], join=False, dodge=(0.8 - 0.8 / 3), )#order=list(label_map.keys())
        common_kwargs['order'] = ['start_time', 't_update']
        ax[1] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
        ax[1] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)
        ax[1].set(ylabel=f'initial - new prob. density')
        ax[1].axhline(0, color='k', linestyle='dashed', alpha=0.5)

        
        #diff_pairs = [((c[0],), (c[1],)) for c in combos]
        
        stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        pvalues = [stats_data['p_val'].to_numpy()[0]]
        #pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in diff_pairs]
        annot = Annotator(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
                          order=list(label_map.keys()))#order=list(label_map.keys())
        annot.new_plot(ax[1], pairs=combos, data=diff_data, x=comparison, y='initial_vs_new',
                       order=['start_time', 't_update'],
                       )#list(label_map.keys())
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))

        # get error differences for reporting purposes
        # bins = self.aggregator.group_aligned_df['bins'].values[0]
        # data_for_stats['error_mean'] = (data_for_stats['error_mean'] - np.min(bins)) / (np.max(bins) - np.min(bins))
        # data_for_stats = data_for_stats.query('choice == "initial"')
        # stats.run(data_for_stats, dependent_vars=['error_mean'], group_vars=['choice', 'region'],
        #           pairs=pairs, filename=f'decoding_error_stats_region_{tags}')

        # #get significant difference from zero for reporting purposes
        # sess_averages = (diff_data
        #                  .groupby(['animal', 'session_id', 'update_type', 'correct','time_label'])
        #                  .mean()
        #                  .reset_index())
        # #pairs = [[[c]] for c in combo_list]
        # stats = Stats(levels=['animal', 'session_id'], results_io=self.results_io,
        #               approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='response')
        # stats.run(sess_averages, dependent_vars=['initial_vs_new'], group_vars='region', pairs=pairs,
        #           filename=f'goal_diff_stats_from_0_{comparison}_{tags}')

        sfig.suptitle(f'{title} goal representation quantification', fontsize=12)
        return sfig

    def plot_goal_coding_by_commitment_single_trials(self, sfig, comparison='update_type', prob_value='prob_sum',
                                                     use_zscores=False, metric='view_angle', tags='',
                                                     update_type=["switch", "stay"], plot_full_breakdown=False, normalize=False):
        # load up data
        decoding_measures = 'zscore_prob' if use_zscores else prob_value
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        choice_data = self.aggregator.calc_choice_commitment_data(self.aggregator.group_aligned_df, plot_groups,
                                                                  prob_value=prob_value, quantiles=4)

        trial_data_raw, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df,
                                                                           plot_groups,
                                                                           prob_value=prob_value, n_time_bins=11,
                                                                           time_window=(-2.5, 2.5))
        merge_keys = ['session_id', 'trial_id', 'times', 'choice']
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'feature_name',
                        'choice']
        agg_dict = {k: (k, np.mean) for k in ['zscore_prob', prob_value, 'diff_baseline', 'choice_commitment_at_0',
                                              'view_angle_at_0']}
        commitment_cols = ['choice_commitment_quantile', 'view_angle_quantile',
                           'choice_commitment_pos_or_neg', 'view_angle_pos_or_neg']
        agg_dict.update({k: (k, lambda x: x.unique()[0]) for k in commitment_cols})
        choice_data = choice_data.merge(trial_data_raw[[*merge_keys, 'zscore_prob']], how='left', left_on=merge_keys,
                                        right_on=merge_keys, validate='one_to_one')
        if not normalize:
            trial_data = (choice_data
                        .query(f'times_binned > 0 & times_binned < 1.5')  # select window
                        .query(f'update_type in {update_type}')
                        .groupby(groupby_cols)[[a[0] for a in agg_dict.values()]]
                        .agg(**agg_dict)  # get mean, peak, or peak latency for each trial (np.argmax)
                        .reset_index()  # TODO - look into nan values
                        .dropna(axis=0, subset=['choice_commitment_at_0', 'view_angle_at_0'])
                        .assign(choice=lambda x: x['choice'].map(dict(initial_stay='initial', switch='new'))))
        elif normalize:
            # Step 1: Define the time range for the pre-window (-1.5 to 0 seconds) for prob_sum
            pre_window_means = (choice_data
                .query('times_binned >= -1.5 & times_binned < 0')  # Pre-window time range
                .groupby(groupby_cols)['prob_sum']  # Group by the relevant columns
                .mean()  # Calculate the mean over the pre-window time range
                .reset_index()
                .rename(columns={'prob_sum': 'prob_sum_pre'}))  # Rename to prob_sum_pre
            # Step 2: Merge the pre-window means back into the full dataset
            trial_data1 = choice_data.merge(pre_window_means, on=groupby_cols, how='left')
            # Step 3: Rename original prob_sum to og_prob_sum
            trial_data1 = trial_data1.rename(columns={'prob_sum': 'og_prob_sum'})
            # Step 4: Normalize the prob_sum by subtracting the pre-window mean and rename it to prob_sum
            trial_data1['prob_sum'] = trial_data1['og_prob_sum'] - trial_data1['prob_sum_pre']
            # Now the 'prob_sum_normalized' column contains the normalized values while all other data remains intact.
            trial_data = (trial_data1
                        .query(f'times_binned > 0 & times_binned < 1.5')  # select window
                        .query(f'update_type in {update_type}')
                        .groupby(groupby_cols)[[a[0] for a in agg_dict.values()]]
                        .agg(**agg_dict)  # get mean, peak, or peak latency for each trial (np.argmax)
                        .reset_index()  # TODO - look into nan values
                        .dropna(axis=0, subset=['choice_commitment_at_0', 'view_angle_at_0'])
                        .assign(choice=lambda x: x['choice'].map(dict(initial_stay='initial', switch='new'))))

        if plot_full_breakdown:
            diff_data = (trial_data
                         .pivot(index=[*groupby_cols[:-1], *commitment_cols,
                                       'view_angle_at_0', 'choice_commitment_at_0'],
                                columns=['choice'],
                                values=[decoding_measures])
                         .reset_index())
            diff_data['initial'] = diff_data[(decoding_measures, 'initial')]
            diff_data['new'] = diff_data[(decoding_measures, 'new')]
            diff_data[('initial_vs_new', '')] = diff_data['initial'] - diff_data['new']
            diff_data = diff_data.droplevel(1, axis=1)

            variables = ['initial', 'new', 'initial_vs_new']
            nrows = len(update_type) * 2
        else:
            variables = ['initial', 'new']
            nrows = len(update_type)

        ax = sfig.subplots(nrows=nrows, ncols=len(variables), sharey=True, squeeze=False)
        for col_id, var in enumerate(variables):
            data_to_plot = diff_data if var == 'initial_vs_new' else trial_data.query(f'choice == "{var}"')
            dependent_var = 'initial_vs_new' if var == 'initial_vs_new' else decoding_measures
            groupers = ['update_type']

            grouping = 'quantile'  # pos_or_neg is other option
            order = ['q3', 'q2', 'q1', 'q0']  # ['negative', 'positive']
            for comp, data in data_to_plot.groupby(groupers):
                row_name = comp[0] if len(groupers) > 1 else comp
                row_id = np.argwhere(np.array(update_type) == row_name)[0][0]
                palette = self.colors['home_quartiles'] if var == 'initial_vs_new' else self.colors[f'{var}_quartiles']

                # sess_averages = (data
                #                  .groupby(['animal', 'session_id', f'{metric}_{grouping}'])[dependent_var]
                #                  .mean()
                #                  .reset_index())
                common_kwargs = dict(data=data, x=f'{metric}_{grouping}', y=dependent_var, order=order,
                                     ax=ax[row_id][col_id], )#y plotting prob_sum from data_to_plot
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, scale=1.5, palette=palette)
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, scale=0.75, palette=['w'] * len(order),
                                                   errorbar=None)
                
                #exporting source data
                data_to_export = data[[f'{metric}_{grouping}', dependent_var]]
                save_path = self.results_io.get_source_data_path()
                filename = f'fig7_data_{tags}.xlsx'
                data_to_export.to_excel(save_path / filename, index=False)
                data_to_export.to_excel(save_path / filename, index=False)
                # arbitrary test for descriptive stats
                stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                              approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='manuscript')
                stats.run(data, dependent_vars=[dependent_var], group_vars=[f'{metric}_{grouping}'],
                          pairs=[[c] for c in list(itertools.combinations(order, r=2))],
                          filename=f'commitment_stats_{metric}_{tags}_{grouping}_{var}')

                stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                              approaches=['traditional'], tests=['spearman'], results_type='manuscript')
                stats.run(data, dependent_vars=[dependent_var], group_vars=[f'{metric}_at_0'], pairs=None,
                          filename=f'commitment_stats_{metric}_{tags}_{grouping}_correlation_{var}')

                stats_data = stats.stats_df.query(f'approach == "traditional" & test == "spearman"'
                                                  f'& variable == "{dependent_var}"')
                pvalues = np.round(stats_data['p_val'].to_numpy(), 4)
                correlations = np.round(stats_data['test_statistic'].to_numpy(), 4)
                rainbow_text(0.05, 0.85, [f'p = {p}, rho = {r}' for p, r in zip(pvalues, correlations)],
                             [self.colors['home']], ax=ax[row_id][col_id], size=8)
                ax[row_id][col_id].set(xlabel=metric, ylabel='probability density after update')
                ax[row_id][col_id].set_title(f'{row_name} trials - {var}', color=self.colors[row_name])

            if plot_full_breakdown:  # plot pos/neg for all trials
                grouping = 'pos_or_neg'  # pos_or_neg is other option
                order = ['negative', 'positive']
                for comp, data in data_to_plot.groupby(groupers):
                    row_name = comp[0] if len(groupers) > 1 else comp
                    row_id = np.argwhere(np.array(update_type) == row_name)[0][0] + len(update_type)
                    palette = self.colors['home_quartiles'] if var == 'initial_vs_new' else self.colors[
                        f'{var}_quartiles']

                    # sess_averages = (data
                    #                  .groupby(['animal', 'session_id', f'{metric}_{grouping}'])[dependent_var]
                    #                  .mean()
                    #                  .reset_index())
                    for n, sess in data.groupby(['animal']):
                        ax[row_id][col_id] = sns.pointplot(sess, x=f'{metric}_{grouping}', y=dependent_var, order=order,
                                                           color=self.colors['home'], errorbar=None,
                                                           ax=ax[row_id][col_id], join=True)
                    [c.set_alpha(0.25) for c in ax[row_id][col_id].collections]
                    [l.set_alpha(0.25) for l in ax[row_id][col_id].lines]
                    ax[row_id][col_id] = sns.pointplot(data, x=f'{metric}_{grouping}', y=dependent_var,
                                                       order=order, scale=1.5, palette=palette, ax=ax[row_id][col_id], )
                    # arbitrary test for descriptive stats
                    stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                                  approaches=['traditional'], tests=['wilcoxon_one_sample'], results_type='manuscript')
                    stats.run(data, dependent_vars=[dependent_var], group_vars=[f'{metric}_{grouping}'],
                              pairs=[[c] for c in list(itertools.combinations(order, r=2))],
                              filename=f'commitment_stats_{comp}_{metric}_{tags}_{grouping}_{var}')

                    stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                                  approaches=['mixed_effects'], tests=['anova'], results_type='manuscript')
                    stats.run(data, dependent_vars=[dependent_var], group_vars=[f'{metric}_{grouping}'], pairs=None,
                              filename=f'commitment_stats_pos_vs_neg_{comp}_{metric}_{tags}_{grouping}_{var}')
                    stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "anova"'
                                                      f'& variable == "{dependent_var}"')
                    pvalues = np.round(stats_data['p_val'].to_numpy(), 4)
                    rainbow_text(0.05, 0.85, [f'p = {p}' for p in pvalues],
                                 [self.colors['home']], ax=ax[row_id][col_id], size=8)
                    ax[row_id][col_id].set(xlabel=metric, ylabel='probability density after update (z-scored)')
                    ax[row_id][col_id].set_title(f'{label_map[row_name]} - {var}',
                                                 color=self.colors[label_map[row_name]])

        colors = [self.colors[t] for t in ['initial', 'new']]
        rainbow_text(0.85, 0.85, ['initial', 'new'], colors, ax=ax[0][col_id], size=8)

        return sfig

    def plot_goal_coding_by_commitment(self, sfig, comparison='update_type', prob_value='prob_sum',
                                       metric='view_angle', update_type=["switch", "stay"],
                                       behavior_only=False, normalize=False):
        # load up data
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        choice_data = self.aggregator.calc_choice_commitment_data(self.aggregator.group_aligned_df, plot_groups,
                                                                  prob_value=prob_value)
        initial_colors = self.colors['initial_quartiles']
        new_colors = self.colors['new_quartiles']
        behavior_colors = self.colors['home_quartiles']

        # plot figure of goal coding over time
        ncols = 1 if behavior_only else 2
        nrows = len(update_type)
        ax = sfig.subplots(nrows=nrows, ncols=ncols, sharey='row', sharex=True, squeeze=False)
        for comp, data in choice_data.query(f'update_type in {update_type}').groupby(comparison):
            row_id = np.argwhere(np.array(update_type) == comp)[0][0]
            sub_groups = f'{metric}_quantile'
            for s_name, s_data in data.groupby(sub_groups):
                color_id = np.argwhere(np.array(['q3', 'q2', 'q1', 'q0']) == s_name)[0][0]
                trial_mat = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'], columns='times',
                                         values=prob_value)
                new_mat = trial_mat.query('choice == "switch"').to_numpy()
                initial_mat = trial_mat.query('choice == "initial_stay"').to_numpy()                    
                time_bins = trial_mat.columns.to_numpy()
                
                if normalize:
                    # Step 1: Identify indices for columns where times < 0 and times >= 0
                    pre_zero_indices = np.where(trial_mat.columns < 0)[0]   # Columns where t < 0
                    # Step 2: Compute the mean of new_mat for the columns with t < 0 (for each row)
                    pre_zero_mean = np.mean(new_mat[:, pre_zero_indices], axis=1, keepdims=True)
                    pre_zero_mean2 = np.mean(initial_mat[:,pre_zero_indices], axis=1, keepdims=True)
                    new_mat -= pre_zero_mean
                    initial_mat -= pre_zero_mean2
                    
                behavior_df = s_data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'],
                                           columns='times', values=metric)
                behavior_mat = behavior_df.query(
                    'choice == "switch"').to_numpy()  # behavior values hsould be same for both

                if not behavior_only:
                    ax[row_id][0].plot(time_bins, np.nanmean(initial_mat, axis=0), color=initial_colors[color_id])
                    ax[row_id][0].fill_between(time_bins,
                                               np.nanmean(initial_mat, axis=0) + sem(initial_mat, nan_policy='omit'),
                                               np.nanmean(initial_mat, axis=0) - sem(initial_mat, nan_policy='omit'),
                                               color=initial_colors[color_id], alpha=0.3)

                    ax[row_id][1].plot(time_bins, np.nanmean(new_mat, axis=0), color=new_colors[color_id])
                    ax[row_id][1].fill_between(time_bins,
                                               np.nanmean(new_mat, axis=0) + sem(new_mat, nan_policy='omit'),
                                               np.nanmean(new_mat, axis=0) - sem(new_mat, nan_policy='omit'),
                                               color=new_colors[color_id], alpha=0.3)
                    ax[row_id][0].set(ylabel='prob density')
                    colors = [self.colors[t] for t in ['initial', 'new']]
                    rainbow_text(0.85, 0.85, ['initial', 'new'], colors, ax=ax[row_id][0], size=8)
                elif behavior_only:
                    ax[row_id][0].plot(time_bins, np.nanmean(behavior_mat, axis=0),
                                       color=behavior_colors[color_id])
                    ax[row_id][0].fill_between(time_bins, np.nanmean(behavior_mat, axis=0) + sem(behavior_mat),
                                               np.nanmean(behavior_mat, axis=0) - sem(behavior_mat),
                                               color=behavior_colors[color_id],
                                               alpha=0.3)
                    ax[0][0].set(ylabel=metric)

            ax[row_id][0].set_title(f'{label_map[comp]}', color=self.colors[label_map[comp]])

        for a in ax.flatten():
            a.axvline(0, color='k', linestyle='dashed', alpha=0.5)
        sfig.supxlabel('time around update (s)', ha='center', fontsize=10)

        return sfig

    @staticmethod
    def nan_sem(x):
        return sem(x, nan_policy='omit')

    def plot_decoding_validation(self, sfig, comparison='update_type', tags='', update_type=['delay only']):
        ax = sfig.subplots(nrows=1, ncols=len(update_type) + 1)
        label_map = self.label_maps[comparison]
        feat_map = dict(y_position='position', x_position='lateral position', dynamic_choice='choice', choice='choice')
        feat = self.aggregator.group_df['feature'].to_numpy()[0]
        feat_bins = self.aggregator.group_df['bins'].to_numpy()[0]
        bins = (feat_bins[1:] + feat_bins[:-1]) / 2

        # get confusion matrix and error
        summary_df = pd.concat(self.aggregator.group_df['summary_df'].to_list())
        summary_df['feature_bins'] = pd.cut(summary_df['actual_feature'], feat_bins)
        confusion_matrix = (summary_df
                            .groupby(['update_type', 'correct'])
                            .apply(lambda x: self.aggregator._get_confusion_matrix(x, feat_bins)))

        if feat == 'choice':
            error_df = (summary_df
                        .query('correct == 1')
                        .assign(error_fraction=lambda x: (x.decoding_error))  # leave error in true units
                        .reset_index())
            extent = [-0.5, 0.5, -0.5, 0.5]
            locations_fractions = self.virtual_track.cue_end_locations.get('dynamic_choice', dict())
        elif feat == 'x_position':
            error_df = (summary_df
                        .query('correct == 1')
                        .assign(error_fraction=lambda x: (x.decoding_error) / (np.max(bins) - np.min(bins)))
                        .reset_index())
            locations_fractions = {k: (v) / (np.max(bins) - np.min(bins))
                                   for k, v in self.virtual_track.cue_end_locations.get(feat, dict()).items()}
            extent = [-0.5, 0.5, -0.5, 0.5]
        else:
            error_df = (summary_df
                        .query('correct == 1')
                        .assign(
                error_fraction=lambda x: (x.decoding_error) / (np.max(bins) - np.min(bins)))
                        .reset_index())
            locations_fractions = {k: (v - np.min(bins)) / (np.max(bins) - np.min(bins))
                                   for k, v in self.virtual_track.cue_end_locations.get(feat, dict()).items()}
            locations_fractions_start = {k: (v - np.min(bins)) / (np.max(bins) - np.min(bins))
                                         for k, v in self.virtual_track.cue_start_locations.get(feat, dict()).items()}
            extent = [0, 1, 0, 1]

        for col_id, label in enumerate(update_type):
            # plot confusion matrix for each trial type
            index_id = (label, 1.0)
            im = ax[col_id].imshow(confusion_matrix.loc[index_id], cmap=self.colors['control_cmap'],
                                   origin='lower', vmin=.015, vmax=0.1, extent=extent)
            ax[col_id].set(ylabel=f'decoded {feat_map[feat]}', xlabel=f'true {feat_map[feat]}')
            ax[col_id].set_title(label, color=self.colors[label])
            if feat != 'choice':
                ax[col_id] = add_task_phase_lines(ax[col_id], cue_locations=locations_fractions_start,
                                                  text_brackets=True)
            # add dashed lines
            for key, value in locations_fractions.items():
                ax[col_id].axvline(value, linestyle='dashed', linewidth=0.5, color='k', alpha=0.5)
                ax[col_id].axhline(value, linestyle='dashed', linewidth=0.5, color='k', alpha=0.5)
        plt.colorbar(im, ax=ax[col_id], pad=0.04, location='right', fraction=0.046, label='prob. density')

        ax[col_id + 1] = sns.ecdfplot(data=error_df, x='error_fraction', hue='session', ax=ax[col_id + 1],
                                      palette=[self.colors['control']] * len(error_df['session'].unique()), alpha=0.25,
                                      legend=False)
        ax[col_id + 1].set(xlabel='decoding error (fraction of track)', ylabel='cumulative fraction',
                           xlim=(0, extent[-1] - extent[0]))
        ax[col_id + 1].set_aspect(1. / ax[col_id + 1].get_data_ratio(), adjustable='box')

        # get error averages for reporting purposes
        pairs = [[[1.0]]]
        error_df['session_id'] = error_df['session']
        stats = Stats(levels=['animal', 'session_id',], results_io=self.results_io,
                      approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='manuscript')
        stats.run(error_df, dependent_vars=['error_fraction'], group_vars=['update_type'],
                  pairs=pairs, filename=f'decoding_validation_{comparison}_{tags}')

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
    
    # def plot_region_error_dc(self, update_type='non_update', comparison=region, sub_group=['animal'], time_label=['t_update'], feature=position):
    #     plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=time_label)
    #     df = self.aggregator.select_group_aligned_data(self.aggregator.group_aligned_df, plot_groups,
    #                                                              ret_df=True)
    #     true_feat = np.nanmean(np.stack(aligned_data['feature']), axis=0)
    #     self.aggregator._summarize()
    
    def plot_region_error_distribution(self, sfig, update_type='non_update', error_meaure='decoding_error', prob_value='prob_sum', feat='position', tags=''):
        plot_groups = dict(update_type=[update_type], time_label=['t_update'], turn_type=[1, 2], correct=[0,1])#, time_label=['t_update']
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value)
        aligned_data = self.aggregator.select_group_aligned_data(self.aggregator.group_aligned_df, plot_groups,
                                                                 ret_df=True)
        clim = (0.02, 0.045) if feat == 'position' else (0.02, 0.04)
        #prob_map = np.nanmean(np.stack(aligned_data['probability']), axis=0)
        true_feat = np.nanmean(np.stack(aligned_data['feature']), axis=0)
        #time_bins = np.linspace(aligned_data['times'].apply(np.nanmin).min(),
        #                        aligned_data['times'].apply(np.nanmax).max(),
        #                        np.shape(prob_map)[1])
        #feat_bins = np.linspace(aligned_data['bins'].apply(np.nanmin).min(),
        #                        aligned_data['bins'].apply(np.nanmax).max(),
        #                        np.shape(prob_map)[0])
        #track_fraction = (feat_bins - np.min(feat_bins)) / (np.max(feat_bins) - np.min(feat_bins))
        #true_feat = (true_feat - np.min(feat_bins)) / (np.max(feat_bins) - np.min(feat_bins))
        
        masked_probabilities = []
        label_map = {('CA1',):'CA1', ('PFC',):'PFC'}
        for trial_idx, (trial_prob, trial_feature, location) in enumerate(zip(aligned_data['probability'], aligned_data['feature'], aligned_data['location'])):
            if feat=='position':
                mask = (trial_feature >= 120.35)
            else:
                mask = (location >= 120.35) & (location <= 254)
            #transposed_mask = mask.T  # or np.transpose(mask)
            # Repeat the mask along the location axis to match the shape of the probability distribution
            #expanded_mask = np.repeat(mask[:, np.newaxis], trial_prob.shape[1], axis=1)
            prob_mask = np.broadcast_to(mask, trial_prob.shape)
            # Apply the mask to the probability distribution (set masked values to NaN)
            trial_prob_masked = np.where(prob_mask, trial_prob, np.nan)
            # Store the masked probabilities
            masked_probabilities.append(trial_prob_masked)
        
        aligned_data['probability_f'] = masked_probabilities
        #step 2: get the location of peak decoding value, and the the spread. std of results? then the locations for where that std is?
        decoding_spreads = []
        decoding_ave_spreads = []
        
        for trial_idx, trial_prob in enumerate(aligned_data['probability_f']):
            trial_spreads = []  # Store spreads for this trial
            ave_trial_spreads = []
            # Convert trial_prob to a numpy array (if not already)
            trial_prob = np.array(trial_prob)  # Shape (n_location_bins, n_locations)
            
            for loc in range(trial_prob.shape[1]):  # Iterate over locations
                if np.all(np.isnan(trial_prob[:, loc])):
                    # Handle NaN cases
                    trial_spreads.append(np.nan)
                    continue
                
                # Get the peak value and index for this location
                peak_location = np.nanargmax(trial_prob[:, loc])#
                peak_decoding = trial_prob[peak_location, loc]#
                
                # Calculate the threshold
                decoding_std = np.nanstd(trial_prob[:, loc])
                threshold = peak_decoding - decoding_std
                
                # Find the first index before the peak where decoding meets/exceeds the threshold
                indices_before_peak = np.where(trial_prob[:peak_location + 1, loc] >= threshold)[0]
                index_before = indices_before_peak[0] if len(indices_before_peak) > 0 else np.nan
                
                # Find the first index after the peak where decoding meets/exceeds the threshold
                indices_after_peak = np.where(trial_prob[peak_location:, loc] >= threshold)[0]
                index_after = peak_location + indices_after_peak[-1] if len(indices_after_peak) > 0 else np.nan
                
                # Compute the spread (index_after - index_before)
                if not np.isnan(index_before) and not np.isnan(index_after):
                    spread = index_after - index_before
                else:
                    spread = np.nan  # If either index is NaN, spread is undefined

                trial_spreads.append(spread)  # Store the spread for this location
            ave_trial_spreads = np.nanmean(trial_spreads)
            decoding_spreads.append(trial_spreads)  # Store the spreads for the trial
            decoding_ave_spreads.append(ave_trial_spreads)
        
        # Add spreads to aligned_data if needed
        aligned_data['decoding_spread'] = decoding_ave_spreads
        #step 3: average this across time/trials/etc and plot the results somehow
        groupby_cols = ['session_id', 'animal', 'region','trial_id']#have no idea how this will work given that the df is a different shape I think
        data_for_stats = (aligned_data
                          .groupby(groupby_cols)[['decoding_spread']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1))) 
        data_for_stats.reset_index(inplace=True)
        
        data_for_stats['region'] = data_for_stats['region'].map(label_map)
        aligned_data['region'] = aligned_data['region'].map(label_map)
        sess_averages = data_for_stats.groupby(['session_id', 'region'])['decoding_spread_mean'].mean().reset_index()
        
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[1, 1]))
        colors = [self.colors[t] for t in list(label_map.values())]

        common_kwargs = dict(data=sess_averages, x='region', y='decoding_spread_mean', ax=ax[0], errwidth=3,
                             order=list(label_map.values()), join=False, dodge=(0.8 - 0.8 / 3), )
        ax[0] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
        ax[0] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)
        ax[0].set(ylabel=f'decoding error')
        ax[0].axhline(0, color='k', linestyle='dashed', alpha=0.5)
        
        diff_pairs = [('CA1', 'PFC')]#('CA1', 'PFC'), ('PFC', 'CA1')
        stats = Stats(levels=['animal', 'session_id','trial_id'], results_io=self.results_io,
                      approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
        stats.run(data_for_stats, dependent_vars=['decoding_spread_mean'], group_vars='region', pairs=diff_pairs,
                  filename=f'goal_coding_stats_region_{tags}')#pairs=diff_pairs,
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "anova"'
                                          f'& variable == "decoding_spread_mean"')
        pvalues = [stats_data['p_val'].to_numpy()[0]]
        #pairs = [(str(p[0]), str(p[1])) for p in pairs]
        annot = Annotator(ax, pairs=diff_pairs, data=data_for_stats, x='region', y='decoding_spread_mean'
                          )#pairs=combos, before data= #order=list(label_map.values())
        annot.new_plot(ax, pairs=diff_pairs, data=data_for_stats, x='region', y='decoding_spread_mean',
                       order=list(label_map.values()),
                       )#pairs=combos, before data=
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))
        sfig.suptitle(f'spread representation quantification', fontsize=12)
        return sfig
        
        

    def plot_all_groups_error(self, sfig, update_type='non_update', main_group=['region'], sub_group=['animal'], error_measures = 'decoding_error', time_label=['t_update'], thresh_params=False, title='', tags=''):
        print('Plotting group decoding error distributions...')
        #plot_groups = dict(update_type=[update_type], turn_type=[1, 2], correct=[1], time_label=time_label)
        data_list = []
        for sess_dict in self.data:
            if not sess_dict['excluded_session']:
                sess_summary = sess_dict['summary_df']
                temp_df = pd.DataFrame({
                    'decoded_feature': sess_summary.decoded_feature,
                    'decoding_error': sess_summary.decoding_error,
                    'actual_feature': sess_summary.actual_feature,
                    'update_type': sess_summary.update_type,
                    'region': [sess_dict['region']] * len(sess_summary),
                    'session_id': [sess_dict['session_id']] * len(sess_summary),
                    'animal': [sess_dict['animal']]*len(sess_summary),
                    'trial_id': sess_summary.trial_id
                })
                mask = (temp_df['update_type'] == 'delay only') & (temp_df['actual_feature'] >= 130.35)# & (temp_df['actual_feature'] <= 245)#eliminating arms + 10 and initial cue + 10 for buffer
                filt_df = temp_df[mask]
                filt_df['rmse_cont']=sqrt(mean_squared_error(filt_df['actual_feature'], filt_df['decoded_feature']))
            #temp_df_ave = data.groupby(['trial_id', 'update_type','animal','region','session_id'])[['decoding_error', 'rmse']].mean().reset_index()
                data_list.append(filt_df)
        df = pd.concat(data_list, ignore_index=True)
        label_map = {('CA1',):'CA1', ('PFC',):'PFC'}
        
        group_data = df#.groupby(main_group)  # main group is what gets the different plots
        
        groupby_cols = ['session_id', 'animal', 'region']#og had 'session_id','animal','trial_id','update_type','correct','feature_name','choice'. need to get animal
        data_for_stats = (group_data
                          .groupby(groupby_cols)[['decoding_error', 'rmse_cont']]  # group by trial/trial type
                          .agg(['mean'])  # get mean, peak, or peak latency for each trial (np.argmax)
                          .pipe(lambda x: x.set_axis(x.columns.map('_'.join), axis=1)))  # fix columns so flattened
        data_for_stats.reset_index(inplace=True)
        #data_for_stats['choice'] = data_for_stats['choice'].map(choice_mapping)
        data_for_stats['region'] = data_for_stats['region'].map(label_map)
        group_data['region'] = group_data['region'].map(label_map)

        sess_averages = data_for_stats.groupby(['session_id', 'region'])['rmse_cont_mean','decoding_error_mean'].mean().reset_index()#getting session averages

        # plot data
        ax = sfig.subplots(nrows=1, ncols=2, gridspec_kw=dict(width_ratios=[1, 1]))
        colors = [self.colors[t] for t in list(label_map.values())]
        # if stripplot:
        #     ax[1] = sns.stripplot(data=sess_averages, x=comparison, y='initial_vs_new', ax=ax[1], zorder=1, legend=False,
        #                           color=self.colors['home_medium'], order=list(label_map.values()), alpha=0.4)
        
        common_kwargs = dict(data=sess_averages, x='region', y='rmse_cont_mean', ax=ax[1], errwidth=3,
                             order=list(label_map.values()), join=False, dodge=(0.8 - 0.8 / 3), )
        ax[1] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
        ax[1] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)
        ax[1].set(ylabel=f'rmse_mean')
        ax[1].axhline(0, color='k', linestyle='dashed', alpha=0.5)

        #combos = list(itertools.combinations(combo_list, r=2))
        #diff_pairs = [(('CA1',), ('PFC',))]
        diff_pairs = [('CA1', 'PFC')]#('CA1', 'PFC'), ('PFC', 'CA1')
        stats = Stats(levels=['animal', 'session_id','trial_id'], results_io=self.results_io,
                      approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='response')
        stats.run(data_for_stats, dependent_vars=['rmse_cont_mean'], group_vars='region', pairs=diff_pairs,
                  filename=f'locattion_region_rmse_error')#pairs=diff_pairs,
        stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                          f'& variable == "rmse_cont_mean"')
        #stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
        #diff_pairs = ['CA1','PFC']#no idea how this will work but we will see
        #pvalues = [stats_data[stats_data['pair'] == tuple(p)]['p_val'].to_numpy()[0] for p in diff_pairs]
        #pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in diff_pairs]#DC change back
        pvalues = [stats_data['p_val'].to_numpy()[0]]
        #pairs = [(str(p[0]), str(p[1])) for p in pairs]
        annot = Annotator(ax[1], pairs=diff_pairs, data=data_for_stats, x='region', y='rmse_cont_mean'
                          )#pairs=combos, before data= #order=list(label_map.values())
        annot.new_plot(ax[1], pairs=diff_pairs, data=data_for_stats, x='region', y='rmse_cont_mean',
                       order=list(label_map.values()),
                       )#pairs=combos, before data=
        (annot
         .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
         .set_pvalues(pvalues=pvalues)
         .annotate(line_offset=0.1, line_offset_to_group=0.025))

        sfig.suptitle(f'{title} error representation quantification', fontsize=12)
        return sfig

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

                # (  # plot corr coeffs over time  # TODO - adapt to not use seaborn objects
                #     so.Plot(g_data_by_time, x='times_sliding', y='corr_coeff_sliding', color='choice')
                #         .facet(col='time_label')
                #         .add(so.Band(), so.Est(errorbar='se'), )
                #         .add(so.Line(linewidth=2), so.Agg(), )
                #         .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                #         .limit(
                #         x=(np.min(interaction_data['times'].values[-1]), np.max(interaction_data['times'].values[1])))
                #         .theme(rcparams)
                #         .layout(engine='constrained')
                #         .on(sfigs[1][0])
                #         .plot()
                # )

                g_data['corr_coeff'][g_data['corr_coeff'].isna()] = 0
                # (  # plot initial - switch difference over time (averages with bar)  # TODO - adapt to not use so
                #     so.Plot(g_data, x='corr_coeff', color='choice')
                #         .facet(col='time_label')
                #         .add(so.Bars(alpha=0.5, edgealpha=0.5),
                #              so.Hist(stat='proportion', binrange=(-1, 1), binwidth=0.1), )
                #         .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                #         .limit(x=(-1, 1))
                #         .label(y='proportion')
                #         .theme(rcparams)
                #         .layout(engine='constrained')
                #         .on(sfigs[2][0])
                #         .plot()
                # )
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

            # time traces  # TODO - adapt to not use seaborn objects so
            # (  # plot initial - switch difference over time (averages with bar)
            #     so.Plot(trial_data, x='times', color='choice')
            #         .facet(col='time_label')
            #         .pair(y=['prob_sum', 'diff_baseline'])
            #         .add(so.Band(), so.Est(errorbar='se'), )
            #         .add(so.Line(linewidth=2), so.Agg(), )
            #         .scale(color=[self.colors[c] for c in trial_data['choice'].unique()])
            #         .theme(rcparams)
            #         .layout(engine='constrained')
            #         .on(sfigs[0][0])
            #         .plot()
            # )
            sfigs[0][0].suptitle(title)  # TODO - adapt to not use seaborn objects so
            # (  # plot diff over time (averages with error bar)
            #     so.Plot(old_vs_new_data, x='times', y='diff_switch_stay')
            #         .facet(col='time_label',
            #                order=self.aggregator.align_times)
            #         .add(so.Line(linewidth=2), so.Agg())
            #         .add(so.Band(), so.Est(errorbar='se'))
            #         .scale(color=self.colors['control'])
            #         .theme(rcparams)
            #         .layout(engine='constrained')
            #         .on(sfigs[7][0])
            #         .plot()
            # )

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

    def plot_tuning_curves(self, sfig, title='hippocampal position tuning curves'):
        # get data
        label_map = dict(model='encoding model',)
                         # model_delay_only='decoded trials - delay only',
                         # model_update_only='decoded trials - update only')
        feat = self.aggregator.group_df['feature'].values[0]
        sort_index, tuning_curve_scaled = dict(), dict()
        for model in list(label_map.keys()):
            model_df = self.aggregator.get_tuning_data(self.aggregator.group_df, model_name=model)
            tuning_curves = pd.concat(model_df['tuning_curve'].to_list(), axis=1)

            place_fields = get_place_fields(tuning_curves=tuning_curves)
            place_field_peak_ind = place_fields.apply(lambda x: get_largest_field_loc(x), axis=0).reset_index(drop=True)
            sort_index[model] = place_field_peak_ind.sort_values(na_position='first').index  # to start at 0

            tuning_curve_mat = np.stack(tuning_curves.T.values)
            tuning_curve_scaled[model] = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]

        tuning_curve_bins = model_df['bins'].values[0]  # should all be the same, use the most recent
        if feat == 'choice':
            extent = [-0.5, 0.5]
            locations_fractions = self.virtual_track.cue_end_locations.get('dynamic_choice', dict())
        elif feat == 'x_position':
            extent = [-0.5, 0.5]
            locations_fractions = {k: (v) / (np.max(tuning_curve_bins) - np.min(tuning_curve_bins))
                                   for k, v in self.virtual_track.cue_end_locations.get(feat, dict()).items()}
        else:
            locations = self.aggregator.group_df.virtual_track.values[0].cue_end_locations.get(feat, dict())
            fraction_of_track = (tuning_curve_bins - np.min(tuning_curve_bins)) / (
                    np.max(tuning_curve_bins) - np.min(tuning_curve_bins))
            locations_fractions = {
                k: (v - np.min(tuning_curve_bins)) / (np.max(tuning_curve_bins) - np.min(tuning_curve_bins))
                for k, v in locations.items()}
            locations_fractions_start = {
                k: (v - np.min(tuning_curve_bins)) / (np.max(tuning_curve_bins) - np.min(tuning_curve_bins))
                for k, v in self.virtual_track.cue_start_locations.get(feat, dict()).items()}
            extent = [fraction_of_track[0], fraction_of_track[-1]]

        # plot data
        ax = sfig.subplots(nrows=1, ncols=len(list(label_map.keys())), sharex=True, sharey=True, squeeze=False)
        for ax_ind, model in enumerate(list(label_map.keys())):
            im = ax[0][ax_ind].imshow(tuning_curve_scaled[model][sort_index['model'], :],
                                   cmap=self.colors['control_cmap_r'],
                                   origin='lower', vmin=0.25, vmax=0.75, aspect='auto',
                                   extent=[*extent, 0, np.shape(tuning_curve_scaled[model])[0]])
            if feat not in ['choice', 'x_position']:
                ax[0][ax_ind] = add_task_phase_lines(ax[0][ax_ind], cue_locations=locations_fractions_start,
                                                  text_brackets=True)
            for key, value in locations_fractions.items():
                ax[0][ax_ind].axvline(value, linestyle='dashed', color='w', alpha=0.5, linewidth=0.5)
                ax[0][ax_ind].set(xlim=extent, ylim=(0, np.shape(tuning_curve_scaled[model])[0]))
            # ax[0][ax_ind].set_title(label_map[model], fontsize=10)

        plt.colorbar(im, ax=ax[0][ax_ind], pad=0.04, location='right', fraction=0.046, label='Normalized firing rate')

        sfig.supylabel(f'units sorted by {self.new_line} {label_map["model"]}', fontsize=10)
        sfig.suptitle(title, fontsize=12)

        return sfig

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

    def plot_theta_phase_info(self, param_data, plot_groups=None, tags='', update_type=['switch', 'non_update']):
        plot_group_dict = {k: [i[0] for i in v] if k != 'turn_type' else v[0] for k, v in plot_groups.items()}
        plot_group_dict.update(time_label=['t_update'])
        theta_data = self.aggregator.calc_theta_phase_data(param_data, plot_group_dict, ret_by_trial=True)
        groupby_cols = ['session_id', 'animal', 'region', 'trial_id', 'update_type', 'correct', 'feature_name', 'times',
                        'phase_mid']
        data_for_stats = (theta_data
                          .query(f'times == "post" & bin_name == "quarter" & update_type in {update_type}')  # only look at post update
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
                    #
                    # (  # plot scatters with estimates # TODO - adapt to not use seaborn objects so
                    #     so.Plot(plot_data, x='correct', color='choice')
                    #         .pair(y=['diff_baseline', ])  # 'prob_over_chance' remove for now
                    #         .add(so.Dot(alpha=0.3))
                    #         .add(so.Line(), so.PolyFit(order=1), )
                    #         .share(y='row')
                    #         .scale(color=[self.colors[c] for c in g_data['choice'].unique()])
                    #         .theme(rcparams)
                    #         .layout(engine='constrained')
                    #         .limit(x=(0, 1))
                    #         .label(title=g_name, x='proportion correct')
                    #         .on(sfigs[row][col])
                    #         .plot()
                    # )
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

    def plot_goal_coding_prediction(self, sfig, comparison='all_update', prob_value='prob_sum', tags=''):
        # load up data
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps
        # colors = [self.colors[t] for t in ['non_update', 'stay', 'switch']]

        # plot figure
        ax = sfig.subplots(nrows=1, ncols=2, sharex=True, squeeze=False)
        predicted_value = ['correct']  # 'choice_made'
        for ind, pred in enumerate(predicted_value):
            predict_df = self.aggregator.predict_trial_outcomes(self.aggregator.group_aligned_df, plot_groups,
                                                                prob_value, comparison=pred, iterations=1000,
                                                                results_io=self.results_io, tags=tags)
            groupers = ['pre_or_post', 'input', 'correct'] if pred == 'choice_made' else ['pre_or_post', 'input',
                                                                                          'region']
            for comp, data in predict_df.groupby(groupers):
                col_id = np.argwhere(predict_df['pre_or_post'].unique() == comp[0])[0][0]
                row_add = int(comp[2] * 2) if pred == 'choice_made' else 0
                row_id = np.argwhere(predict_df['input'].unique() == comp[1])[0][0] + row_add

                # plot data
                common_kwargs = dict(data=data.query('target == "shuffled"'), x='update_type', y='score',
                                     ax=ax[row_id][col_id], errwidth=3, join=False)
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=[self.colors['control']] * 3, scale=1.5, capsize=0.75)
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=['w'] * 3, scale=0.75, errorbar=None)
                # ax[0].get_legend().remove()
                # rainbow_text(0.5, 0.9, list(label_map.values()), colors, ax=ax[0], size=8)

                colors = [self.colors[t] for t in data.query('target == "shuffled"')['update_type'].unique()]
                common_kwargs = dict(data=data.query('target == "actual"'), x='update_type', y='score',
                                     ax=ax[row_id][col_id], errwidth=3, join=False,)
                #saving source data
                plot_data = data.query('target == "shuffled"')[['update_type', 'score']]
                save_path = self.results_io.get_source_data_path()
                filename = f'pointplot_prediction_data_{tags}.xlsx'
                plot_data.to_excel(save_path / filename, index=False)
                
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
                ax[row_id][col_id].set(xlabel=f'update type', ylabel=f'prediction accuracy', ylim=(0.475, 0.625))

                p_values = []
                accuracy_scores = []
                for u in ['switch']:
                    actual_score = data.query(f'update_type == "{u}" & target == "actual"')['score'].to_numpy()
                    shuffled_scores = data.query(f'update_type == "{u}" & target == "shuffled"')['score'].to_numpy()
                    p_values.append(sum(shuffled_scores >= actual_score) / len(shuffled_scores))
                    accuracy_scores.append(np.round(actual_score, 4))
                rainbow_text(0.05, 0.85, p_values, colors, ax=ax[row_id][col_id], size=8)
                rainbow_text(0.05, 0.85, accuracy_scores, colors, ax=ax[row_id][col_id], size=8)

                ax[row_id][col_id].set(title=f'{comp[1]} trials - {comp[0]} cue - {pred} - {groupers[2]}: {comp[2]}')
                # ax[row_id][col_id].set(xlim=(0.475, 0.625), ylim=(0, 0.36))
                # ax[row_id][col_id].get_legend().remove()

            # rainbow_text(0.05, 0.85, ['delay only', 'stay', 'switch'], colors, ax=ax[0][0], size=8)
            sfig.suptitle(f'prediction of {comparison} trial types with goal codes', fontsize=12)

        return sfig

    def plot_theta_phase_modulation(self, sfig, comparison='update_type', prob_value='prob_sum', tags='', time=None,
                                    update_type=None):
        update_type = update_type or ['switch', 'stay', 'non_update']  # set to defaults
        time = time or ['post']  # set to defaults
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        theta_data = self.aggregator.calc_theta_phase_data(self.aggregator.group_aligned_df, plot_groups,
                                                           ret_by_trial=True)
        
        #exporting source data
        raw_data_export = theta_data.query(f'bin_name == "full" & times in {time} & update_type in {update_type}')
        data_export = raw_data_export[['phase_bins', 'initial', 'new', 'central', 'probability','times']]
        save_dir = self.results_io.get_source_data_path()
        filename = f'phase_plot_data_{tags}.xlsx'
        data_export.to_excel(save_dir / filename, index=False)
            
        averages = (theta_data
                    .query(f'bin_name == "full" & times in {time} & update_type in {update_type}')
                    .groupby(['update_type', 'times', 'phase_mid'])[['initial', 'new', 'central', 'theta_amplitude']]
                    .agg(['mean', sem])
                    .reset_index())

        # calculate difference
        diff_pre_post = (theta_data
                         .query('bin_name == "full"')
                         .pivot(index=['animal', 'session_id', 'trial_id', 'update_type', 'phase_mid', 'correct'],
                                columns='times', values=['initial', 'new', 'central', 'theta_amplitude'])
                         .reset_index())
        for var in ['initial', 'new', 'central', 'theta_amplitude']:
            diff_pre_post[f'{var}_post_vs_pre'] = diff_pre_post[(var, 'post')] - diff_pre_post[(var, 'pre')]
            diff_pre_post.drop([(var, 'post'), (var, 'pre')], axis=1, inplace=True)
        new_col_names = [f'{v}_post_vs_pre' for v in ['initial', 'new', 'central', 'theta_amplitude']]
        pre_post_averages = (diff_pre_post
                             .droplevel(1, axis=1)
                             .groupby(['update_type', 'phase_mid'])[new_col_names]
                             .agg(['mean', self.nan_sem])
                             .reset_index())
        # ax[row_ind][col_ind].plot(data[('phase_mid', '')].to_numpy() / np.pi, data[(f'{var}_post_vs_pre', 'mean')],
        #                           color=self.colors[f'{var}'])
        # ax[row_ind][col_ind].fill_between(data[('phase_mid', '')].to_numpy() / np.pi,
        #                                   data[(f'{var}_post_vs_pre', 'mean')] - data[(f'{var}_post_vs_pre', 'nan_sem')],
        #                                   data[(f'{var}_post_vs_pre', 'mean')] + data[(f'{var}_post_vs_pre', 'nan_sem')],
        #                                   color=self.colors[f'{var}'], alpha=0.25)

        ax = sfig.subplots(nrows=1, ncols=len(update_type), squeeze=False, sharey='row', sharex=True)
        row_ind = 0
        for name, data in averages.groupby(['update_type', 'times']):
            col_ind = np.argwhere(np.array(update_type) == name[0])[0][0]
            color_adjust = '_light' if name[1] == 'pre' else ''
            # ax[0][col_ind].plot(data[('phase_mid', '')].to_numpy() / np.pi, data[('theta_amplitude', 'mean')],
            #                     color=self.colors[f'home'])
            for var in ['initial', 'new', 'central']:
                ax[row_ind][col_ind].plot(data[('phase_mid', '')].to_numpy() / np.pi, data[(var, 'mean')],
                                          color=self.colors[f'{var}{color_adjust}'])
                ax[row_ind][col_ind].fill_between(data[('phase_mid', '')].to_numpy() / np.pi,
                                                  data[(var, 'mean')] - data[(var, 'sem')],
                                                  data[(var, 'mean')] + data[(var, 'sem')],
                                                  color=self.colors[f'{var}'], alpha=0.25)

            ax[row_ind][col_ind].set(title=label_map[name[0]])
            ax[row_ind][0].set(ylabel='prob. density')

        ax_list = ax.flat
        for a in ax_list:
            a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
            a.xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
            ax[row_ind][col_ind].set(xlabel='theta phase')

        return sfig

    def plot_theta_phase_stats(self, sfig, comparison='update_type', prob_value='prob_sum', tags='', type='modulation',
                               divider='half', update_type=None, ylim=None):
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        update_type = update_type or ['switch', 'stay', 'non_update']  # set to defaults
        theta_data = self.aggregator.calc_theta_phase_data(self.aggregator.group_aligned_df, plot_groups,
                                                           ret_by_trial=True)
        cats = ['-2.3565', '-0.7855', '0.7855', '2.3565'] if divider == 'quarter' else ['-1.571', '1.571']
        arm_cats = ['initial', 'new', 'central']
        ylim = ylim or (0.1, 0.18)  # set to default

        modulation = (theta_data
                      .query(f'bin_name == "{divider}" & update_type in {update_type}')
                      .melt(value_name='prob_value', value_vars=arm_cats, var_name='choice',
                            id_vars=['animal', 'session_id', 'trial_id', 'update_type', 'phase_mid', 'times'])
                      .assign(phase_mid=lambda x: pd.Categorical(x['phase_mid'].astype(str), ordered=True,
                                                                 categories=cats)))
        post_modulation = modulation.query('times == "post"')
        stats = Stats(levels=['animal', 'session_id', 'trial_id'], results_io=self.results_io,
                      approaches=['mixed_effects'], tests=['anova', 'emmeans'], results_type='manuscript')
        combos = [('initial', 'central'), ('new', 'central')]
        group_list = post_modulation['phase_mid'].unique()
        pairs = [((g, c[0]), (g, c[1],)) for c in combos for g in group_list]
        
        #exporting source data
        df_export = post_modulation[['phase_mid','prob_value','choice']]
        save_dir = self.results_io.get_source_data_path()
        filename = f'phase_stats_data_{tags}.xlsx'
        df_export.to_excel(save_dir / filename, index=False)

        # plot phase modulation comparisons between goal arm locations
        colors = [self.colors[t] for t in arm_cats]
        col_dict = dict(modulation=1, pre_vs_post=2, phase_modulation=3, )
        # col_multiplier = col_multiplier * 2 if divider == 'quarter' else col_multiplier
        ax = sfig.subplots(nrows=1, ncols=len(update_type) * col_dict[type], sharey='row', squeeze=False)
        row_id = 0
        if type == 'modulation':
            col_id = 0
            for comp, data in post_modulation.groupby(comparison):
                common_kwargs = dict(data=data, x='phase_mid', y='prob_value', hue='choice', ax=ax[row_id][col_id],
                                     hue_order=arm_cats, errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), scale=0.75, errorbar=None)
                ax[row_id][col_id].set(title=comp)
                ax[row_id][col_id].get_legend().remove()

                # get stats
                stats.run(data, dependent_vars=['prob_value'], group_vars=['phase_mid', 'choice',],
                          pairs=pairs, filename=f'theta_mod_by_phase_{comparison}_{tags}_{comp}')
                stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                                  f'& variable == "prob_value"')
                stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
                pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]

                # annotate the plot
                annot = Annotator(ax[row_id][col_id], pairs=pairs, data=data, x='phase_mid', y='prob_value', hue='choice',
                                  hue_order=arm_cats)
                annot.new_plot(ax[row_id][col_id], pairs=pairs, data=data,  x='phase_mid', y='prob_value', hue='choice',
                               hue_order=arm_cats)
                (annot
                 .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
                 .set_pvalues(pvalues=pvalues)
                 .annotate(line_offset=0.1, line_offset_to_group=0.025))
                col_id += 1
            rainbow_text(0.01, 0.9, arm_cats, colors, ax=ax[row_id][0], size=8)
        elif type == 'pre_vs_post':
            # plot pre vs. post comparisons and between trial types
            group_list = modulation['times'].unique()
            pairs = [((group_list[0], c), (group_list[1], c)) for c in arm_cats]
            for comp, data in modulation.groupby([comparison, 'phase_mid']):  # TODO - query only some phases first
                col_multiplier = np.argwhere(modulation['phase_mid'].unique() == comp[1])[0][0]
                col_id = np.argwhere(np.array(update_type) == comp[0])[0][0] * 2 + col_multiplier

                common_kwargs = dict(data=data, x='times', y='prob_value', hue='choice', ax=ax[row_id][col_id],
                                     hue_order=arm_cats, errwidth=3, dodge=(0.8 - 0.8 / 3), )
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=colors, scale=1.5)
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=['w'] * len(colors), join=False,
                                                   scale=0.75, errorbar=None)
                ax[row_id][col_id].get_legend().remove()

                # get stats
                stats.run(data, dependent_vars=['prob_value'], group_vars=['times', 'choice'],
                          pairs=pairs, filename=f'theta_mod_pre_vs_post_{comparison}_{tags}_{comp}')
                stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                                  f'& variable == "prob_value"')
                stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])
                pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]

                # annotate the plot
                annot = Annotator(ax[row_id][col_id], pairs=pairs, data=data, x='times', y='prob_value', hue='choice',
                                  hue_order=arm_cats)
                annot.new_plot(ax[row_id][col_id], pairs=pairs, data=data, x='times', y='prob_value', hue='choice',
                               hue_order=arm_cats)
                (annot
                 .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
                 .set_pvalues(pvalues=pvalues)
                 .annotate(line_offset=0.1, line_offset_to_group=0.025))
            rainbow_text(0.01, 0.9, arm_cats, colors, ax=ax[row_id][0], size=8)
            ax[row_id][0].set(ylim=ylim)
        elif type == 'phase_modulation':
            combos = list(itertools.combinations(cats, r=2))
            pairs = [((c[0],), (c[1],)) for c in combos]
            for comp, data in post_modulation.groupby('choice'):
                col_id = np.argwhere(np.array(['initial', 'new', 'central']) == comp)[0][0]
                common_kwargs = dict(data=data, x='phase_mid', y='prob_value', ax=ax[row_id][col_id],
                                     errwidth=3, join=False, dodge=(0.8 - 0.8 / 3), )
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=[self.colors[comp]] * len(cats), scale=1.5)
                ax[row_id][col_id] = sns.pointplot(**common_kwargs, palette=['w'] * len(cats), scale=0.75,
                                                   errorbar=None)
                ax[row_id][col_id].set(title=comp)

                # get stats
                stats.run(data, dependent_vars=['prob_value'], group_vars=['phase_mid', ],
                          pairs=pairs, filename=f'theta_mod_by_phase_{comparison}_{tags}_{comp}')
                stats_data = stats.stats_df.query(f'approach == "mixed_effects" & test == "emmeans"'
                                                  f'& variable == "prob_value"')
                stats_data['pair'] = stats_data['pair'].apply(lambda x: x[0])  # TODO - add to stats function
                pvalues = [stats_data[stats_data['pair'] == p]['p_val'].to_numpy()[0] for p in pairs]

                # annotate the plot
                annot = Annotator(ax[row_id][col_id], pairs=combos, data=data, x='phase_mid', y='prob_value',)
                annot.new_plot(ax[row_id][col_id], pairs=combos, data=data, x='phase_mid', y='prob_value')
                (annot
                 .configure(test=None, test_short_name='mann-whitney', text_format='star', text_offset=0.05)
                 .set_pvalues(pvalues=pvalues)
                 .annotate(line_offset=0.1, line_offset_to_group=0.025))
            rainbow_text(0.01, 0.9, arm_cats, colors, ax=ax[row_id][0], size=8)

        # run stats for comparisons between trial types
        modulation = (theta_data
                      .query(f'bin_name == "{divider}" & times == "post"')
                      .melt(value_name='prob_value', value_vars=arm_cats, var_name='choice',
                            id_vars=['animal', 'session_id', 'trial_id', 'update_type', 'phase_mid', 'times'])
                      .assign(phase_mid=lambda x: pd.Categorical(x['phase_mid'].astype(str), ordered=True,
                                                                 categories=cats)))

        combo_list = ['switch', 'stay', 'non_update']
        combos = list(itertools.combinations(combo_list, r=2))
        group_list = post_modulation['phase_mid'].unique()
        pairs = [((g, c[0]), (g, c[1],)) for c in combos for g in group_list]
        for name, data in modulation.groupby('choice'):
            stats.run(data, dependent_vars=['prob_value'], group_vars=['phase_mid', 'update_type'],
                      pairs=pairs, filename=f'theta_mod_switch_vs_stay_vs_delay_{name}_{comparison}_{tags}')

        return sfig

    @staticmethod
    def plot_phase_reference(fig):
        x = np.arange(0, 2 * np.pi, 0.1)
        y = np.cos(x)

        ax = fig.subplots(nrows=1, ncols=1)
        ax.plot(x, y)
        ax.set(xlabel='phase', xlim=(0, 2 * np.pi), ylim=(-1, 1))

        return fig

    def plot_motor_controls(self, param_data, plot_groups=None, comparison='correct', tags=''):
        filter_dict = {k: [val[0] for val in v] for k, v in plot_groups.items()}
        filter_dict.update(turn_type=[1, 2])
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(param_data, filter_dict)
        trial_types = ['correct', 'incorrect'] if comparison == 'correct' else ['switch', 'stay', 'non_update']
        resid_data = self.aggregator.get_residuals(trial_data)

        for trial_type, t_data in resid_data.groupby('update_type'):
            sfig = plt.figure()
            ax = sfig.subplots(nrows=4, ncols=len(resid_data[comparison].unique()), sharey='row', squeeze=False)
            for comp, data in t_data.groupby(comparison):
                for row_id, val in enumerate(['prob_sum', 'resid', 'rotational_velocity', 'translational_velocity']):
                    col_id = np.argwhere(np.array(t_data[comparison].unique()) == comp)[0][0]
                    trial_mat = data.pivot(index=['choice', 'trial_index', 'session_id', 'animal'], columns='times',
                                           values=val)
                    new_mat = trial_mat.query('choice == "switch"').to_numpy()
                    initial_mat = trial_mat.query('choice == "initial_stay"').to_numpy()
                    time_bins = trial_mat.columns.to_numpy()

                    ax[row_id][col_id].plot(time_bins, np.nanmean(initial_mat, axis=0), color=self.colors['initial'])
                    ax[row_id][col_id].plot(time_bins, np.nanmean(new_mat, axis=0), color=self.colors['new'])
                    ax[row_id][col_id].fill_between(time_bins,
                                                    np.nanmean(new_mat, axis=0) + sem(new_mat, nan_policy='omit'),
                                                    np.nanmean(new_mat, axis=0) - sem(new_mat, nan_policy='omit'),
                                                    color=self.colors['new'], alpha=0.2)
                    ax[row_id][col_id].fill_between(time_bins, np.nanmean(initial_mat, axis=0) + sem(initial_mat,
                                                                                                     nan_policy='omit'),
                                                    np.nanmean(initial_mat, axis=0) - sem(initial_mat,
                                                                                          nan_policy='omit'),
                                                    color=self.colors['initial'],
                                                    alpha=0.2)
                    ax[row_id][col_id].axvline(0, color='k', linestyle='dashed', alpha=0.5)
                    ax[row_id][col_id].set_title(f'{comp} {val}')
            sfig.supylabel('prob. density', fontsize=10)
            sfig.supxlabel('time around update (s)', ha='center', fontsize=10)

            colors = [self.colors[t] for t in ['initial', 'new']]
            rainbow_text(0.05, 0.05, ['initial', 'new'], colors, ax=ax[0][col_id], size=8)

            self.results_io.save_fig(fig=sfig, axes=ax, filename=f'motor_controls_{comparison}_{trial_type}',
                                     additional_tags=tags)

    def plot_residual_r_squared(self, sfig, comparison='update_type', groups=None, prob_value='prob_sum',
                         xlim=None, update_type=['switch', 'non_update'], correct_type=[1, 0]):
        # load up data
        xlim = xlim or (0, 0.3)
        plot_groups = self.plot_group_comparisons_full[comparison]
        label_map = self.label_maps[comparison]
        plot_groups.update(update_type=update_type)
        trial_data, _ = self.aggregator.calc_trial_by_trial_quant_data(self.aggregator.group_aligned_df, plot_groups,
                                                                       prob_value=prob_value)
        trial_data = trial_data.query('update_type == "switch"') if comparison == 'correct' else trial_data
        resid_data = self.aggregator.get_residuals(trial_data, by_session=True)

        # plot figure
        ax = sfig.subplots(nrows=1, ncols=1, sharex=True, sharey='row')
        hist_data = resid_data.groupby('session_id').mean('r_squared')
        ax = sns.histplot(hist_data, x='r_squared', ax=ax, binwidth=0.01)
        ax.set(title=f"{resid_data['region'].unique()[0][0]} - model goodness of fit", xlabel='pseudo r-squared',
               ylabel='# sessions', xlim=xlim)

        return sfig