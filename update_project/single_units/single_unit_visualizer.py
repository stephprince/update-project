import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import itertools

from pathlib import Path
from scipy.stats import sem

from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_color_theme
from update_project.single_units.single_unit_aggregator import SingleUnitAggregator

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')
rcparams = mpl.rcParams


class SingleUnitVisualizer:

    def __init__(self, data, session_id=None, grid_search=False, target_var='choice'):
        self.data = data
        self.colors = get_color_theme()
        self.virtual_track = data[0]['analyzer'].virtual_track
        self.align_times = data[0]['analyzer'].align_times
        self.plot_groups = dict(update_type=[['non_update'], ['switch'], ['stay']],
                                turn_type=[[1], [2], [1, 2]],
                                outcomes=[[0], [1], [0, 1]])

        self.aggregator = SingleUnitAggregator()
        self.aggregator.run_aggregation(data)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def plot(self):

        for plot_types in list(itertools.product(*self.plot_groups.values())):
            plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
            tags = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
            self.plot_units_aligned(plot_group_dict, tags)

        self.plot_place_fields()

    def plot_place_fields(self):
        # 1 plot of all cells (sorted by peak), 1 of left selective/right selective only, 1 of selectivity dist in both
        # 1 figure per brain region
        for g_name, g_data in self.aggregator.group_tuning_curves.groupby('region'):
            fig, axes = plt.subplots(2, 3, figsize=(8.5, 8.5), layout='constrained')

            # plot heatmaps of cell spatial maps
            for row_ind, metric in enumerate(['mean_selectivity_index', 'max_selectivity_index']):
                for col_ind, condition in enumerate(['< 0', '>= 0']):  # right selective, left selective, any cells
                    data = g_data.query(f'{metric} {condition}')
                    cols_to_skip = ['session_id', 'animal', 'feature_name', 'unit_id', 'region', 'cell_type',
                                    'mean_selectivity_index', 'max_selectivity_index', 'place_field_threshold']
                    tuning_curve_mat = np.stack(data[data.columns.difference(cols_to_skip)].to_numpy())
                    tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
                    sort_index = np.argsort(np.argmax(tuning_curve_scaled, axis=1))
                    tuning_curve_bins = self.aggregator.group_df['tuning_bins'].to_numpy()[0]

                    y_limits = [0, np.shape(tuning_curve_scaled)[0]]
                    x_limits = [np.round(np.min(tuning_curve_bins), 2), np.round(np.max(tuning_curve_bins), 2)]
                    im = axes[row_ind][col_ind].imshow(tuning_curve_scaled[sort_index, :], cmap=self.colors['cmap_r'],
                                                       origin='lower', vmin=0.1, vmax=0.9, aspect='auto',
                                                       extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]])

                    # plot annotation lines
                    locations = self.virtual_track.cue_end_locations.get(g_data['feature_name'].values[0], dict())  #TODO - generalize
                    for key, value in locations.items():
                        axes[row_ind][col_ind].axvline(value, linestyle='dashed', color='k', alpha=0.5)
                    axes[row_ind][col_ind].set(xlim=x_limits, ylim=y_limits, xlabel='units', ylabel='position',
                                               title=f'{metric} {condition}')

                plt.colorbar(im, ax=axes[row_ind][col_ind], pad=0.04, location='right', fraction=0.046,
                             label='Normalized firing rate')

                # plot distribution of selectivity
                axes[row_ind][col_ind + 1].hist(g_data[metric].dropna().to_numpy(),
                                                bins=np.linspace(-1, 1, 20))
                axes[row_ind][col_ind + 1].set(xlabel=metric, ylabel='count',
                                               title='distribution of goal selectivity')
                axes[row_ind][col_ind + 1].axvline(0, linestyle='dashed', color='k', alpha=0.5)

            # save figure
            fig.suptitle(f'Goal selectivity - {g_name}')
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves', additional_tags=g_name,
                                     tight_layout=False)

    def plot_unit_selectivity(self):
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'goal_selectivity', additional_tags=tags)


    def plot_units_aligned(self, plot_groups, tags):
        # 2 heatmaps initial/new selective cells
        for g_name, g_data in self.aggregator.group_aligned_data.groupby('region', sort='False'):
            aligned_data = self.aggregator.select_group_aligned_data(g_data, plot_groups)
            psth_data = self.aggregator.get_aligned_psth(aligned_data)

            for s_type in ['mean_selectivity_type', 'max_selectivity_type']:
                fig, axes = plt.subplots(7, len(g_data['time_label'].unique()), figsize=(17, 22), layout='constrained',
                                         sharex=True, sharey='row')
                for t_name, t_data in psth_data.groupby('time_label'):
                    col = np.argwhere(g_data['time_label'].unique() == t_name)[0][0]
                    psth_times = t_data['psth_times'].to_numpy()[0]
                    selectivity_dict = dict(all=dict(filter=[np.nan, 'switch', 'stay'], rows=[0, 1], color='all'),
                                            non_goal=dict(filter=[np.nan], rows=[2, 3], color='home'),
                                            switch=dict(filter=['switch'], rows=[4, 5], color='switch'),
                                            stay=dict(filter=['stay'], rows=[4, 6], color='stay'))
                    for key, value in selectivity_dict.items():
                        plot_data = t_data[t_data[s_type].isin(value['filter'])]
                        psth_mat_all = np.vstack(plot_data['psth_mean'])
                        psth_mean = np.nanmean(psth_mat_all, axis=0)
                        psth_sem = sem(psth_mat_all, axis=0, nan_policy='omit')

                        # average + heatmap of all cells
                        axes[value['rows'][0]][col].plot(psth_times, np.nanmean(psth_mat_all, axis=0),
                                                         color=self.colors[value['color']])
                        axes[value['rows'][0]][col].fill_between(psth_times, psth_mean + psth_sem, psth_mean - psth_sem,
                                                                 color=self.colors[value['color']], alpha=0.3)
                        axes[value['rows'][0]][col].set(ylabel='mean fr', ylim=[0, 5])

                        im = axes[value['rows'][1]][col].imshow(psth_mat_all, cmap=self.colors[f'{value["color"]}_cmap'],
                                                                origin='lower', aspect='auto', vmin=0, vmax=20,
                                                                extent=[psth_times[0], psth_times[-1],
                                                                        0, np.shape(psth_mat_all)[0]])
                        axes[value['rows'][1]][col].set(ylabel='units', ylim=(0, np.shape(psth_mat_all)[0]))

                        if value['rows'][0] == 0:
                            axes[value['rows'][0]][col].set(title='t_name')
                        if col == len(g_data['time_label'].unique()) - 1:
                            plt.colorbar(im, ax=axes[value['rows'][1]], pad=0.04, location='right', fraction=0.046,
                                         label=f'{key} mean fr')

                add_lines = [a.axvline(0, color='k', linestyle='dashed', alpha=0.5) for a in axes.flatten()]

                # save figure
                fig.supxlabel('time around cue (s)')
                fig.suptitle(f'Aligned goal selective activity - {g_name} - {plot_groups}')
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'spiking_aligned_{g_name}_{s_type}',
                                         additional_tags=tags, tight_layout=False)