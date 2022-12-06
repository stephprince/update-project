import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn.objects as so

from pathlib import Path
from scipy.stats import sem

from update_project.general.results_io import ResultsIO
from update_project.general.plots import get_color_theme
from update_project.single_units.psth_visualizer import show_psth_raster
from update_project.single_units.single_unit_aggregator import SingleUnitAggregator
from update_project.base_visualization_class import BaseVisualizationClass


class SingleUnitVisualizer(BaseVisualizationClass):

    def __init__(self, data, session_id=None, grid_search=False, target_var='choice'):
        super().__init__(data)
        self.align_times = data[0]['analyzer'].align_times
        self.aggregator = SingleUnitAggregator()
        self.aggregator.run_aggregation(data)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

        self.cutoffs = [0.25, 0.5]
        self.update_plot_dict = dict(
            switch_vs_non_update_index=dict(psths=['mean_switch', 'mean_non_update', 'diff_switch_non_update'],
                                            conditions=['>', '<'], ),
            switch_vs_stay_index=dict(psths=['mean_switch', 'mean_stay', 'diff_switch_stay'],
                                      conditions=['>', '<'], ))

    def plot(self):
        # for g_name, g_data in self.aggregator.group_tuning_curves.groupby(['region', 'feature_name']):
        #     for cutoff in self.cutoffs:
        #         self.plot_update_selectivity(g_data, g_name, cutoff)
        #         self.plot_update_selective_cell_types(g_data, g_name, cutoff)
        #     self.plot_place_fields(g_data, g_name)

        for g_name, g_data in self.aggregator.group_aligned_data.groupby(['region', 'feature_name'], sort='False'):
            title = '_'.join(g_name)
            for plot_types in list(itertools.product(*self.plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
                tags = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
                # self.plot_theta_phase_modulation(g_data,  plot_group_dict, f'{title}_{tags}')
                # self.plot_units_aligned(g_data, plot_group_dict, f'{title}_{tags}')
                self.plot_movement_reaction_times(g_data, plot_group_dict, f'{title}_{tags}')
                # self.plot_goal_selectivity(g_data, plot_group_dict, f'{title}_{tags}')

    def plot_place_fields(self, g_data, g_name):
        # 1 plot of all cells (sorted by peak), 1 of left selective/right selective only, 1 of selectivity dist in both
        # 1 figure per brain region
        fig, axes = plt.subplots(4, 3, figsize=(8.5, 8.5), layout='constrained')

        # plot heatmaps of cell spatial maps
        # note these are generated by the non-update trials so only have that data even when sort by switch/stay index
        plot_dict = dict(mean_selectivity_index=dict(row=0, cutoff=0, sort_by='place_field_peak_ind'),
                         max_selectivity_index=dict(row=1, cutoff=0, sort_by='place_field_peak_ind'),
                         switch_vs_non_update_index=dict(row=2, cutoff=self.cutoff, sort_by='place_field_peak_ind'),
                         switch_vs_stay_index=dict(row=3, cutoff=self.cutoff, sort_by='place_field_peak_ind'))
        for metric, value in plot_dict.items():
            for col_ind, condition in enumerate([f'< -{value["cutoff"]}', f'>= {value["cutoff"]}']):  # right selective, left selective, any cells
                row_ind = value['row']
                data = g_data.query(f'{metric} {condition}')
                data.sort_values(value['sort_by'], inplace=True)
                tuning_curve_mat = np.stack(data['tuning_curves'].to_numpy())
                tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
                tuning_curve_bins = self.aggregator.group_df['tuning_bins'].to_numpy()[0]

                y_limits = [0, np.shape(tuning_curve_scaled)[0]]
                x_limits = [np.round(np.min(tuning_curve_bins), 2), np.round(np.max(tuning_curve_bins), 2)]
                im = axes[row_ind][col_ind].imshow(tuning_curve_scaled, cmap=self.colors['cmap_r'],
                                                   origin='lower', vmin=0.1, vmax=0.9, aspect='auto',
                                                   extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]])

                # plot annotation lines
                locations = self.virtual_track.cue_end_locations.get(g_data['feature_name'].values[0], dict())  #TODO - generalize
                for key, val in locations.items():
                    axes[row_ind][col_ind].axvline(val, linestyle='dashed', color='k', alpha=0.5)
                axes[row_ind][col_ind].set(xlim=x_limits, ylim=y_limits, ylabel='units', xlabel='position',
                                           title=f'{metric} {condition}')

                # plot bounds
                spaces = self.virtual_track.edge_spacing
                for s in spaces:
                    axes[row_ind][col_ind].axvspan(*s, color='#DDDDDD', edgecolor=None)

            plt.colorbar(im, ax=axes[row_ind][col_ind], pad=0.04, location='right', fraction=0.046,
                         label='Normalized firing rate')

            # plot distribution of selectivity
            axes[row_ind][col_ind + 1].hist(g_data[metric].dropna().to_numpy(),
                                            bins=np.linspace(-1, 1, 20))
            axes[row_ind][col_ind + 1].set(xlabel=metric, ylabel='count',
                                           title=f'distribution of {metric}')
            axes[row_ind][col_ind + 1].axvline(-value['cutoff'], linestyle='dashed', color='k', alpha=0.5)
            axes[row_ind][col_ind + 1].axvline(value['cutoff'], linestyle='dashed', color='k', alpha=0.5)

        # save figure
        fig.suptitle(f'Goal selectivity - {g_name}')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves', additional_tags=g_name,
                                 tight_layout=False)

    def plot_update_selective_cell_types(self, g_data, g_name, cutoff=0.25):
        cutoffs = [cutoff, -cutoff]
        g_data['goal_selectivity_type'] = ['switch' if x < 0 else 'stay' if x > 0 else 'home'
                                           for x in g_data['max_selectivity_index']]

        for metric, value in self.update_plot_dict.items():
            g_data['update_selectivity_type'] = [
                'positive' if x > cutoffs[0] else 'negative' if x < cutoffs[1] else 'none'
                for x in g_data[metric]]

            # separate +/- update selectivity of cell types and goal selective cells
            fig = plt.figure(figsize=(10, 10))
            sfigs = fig.subfigures(2, 2, )
            selectivity_dict = dict(cell_selectivity_type=dict(ind=0, group='cell_type',),
                                    goal_selectivity_type=dict(ind=1, group='goal_selectivity_type',))
            for s_key, s_value in selectivity_dict.items():
                # plot distribution data
                (
                    so.Plot(g_data, x=metric, color=s_value['group'])
                        # .facet(row='update_selectivity_type')
                        .add(so.Area(), so.Hist(stat='proportion', binwidth=0.1, common_norm=False))
                        .theme(rcparams)
                        .scale(color=[self.colors[c] for c in g_data[s_value['group']].unique()])
                        .on(sfigs[0][s_value['ind']])
                        .label(y='proportion', title=s_key)
                        .plot()
                )
                medians = g_data.groupby(s_value['group'])[metric].median().to_dict()
                for m_key, m_value in medians.items():
                    sfigs[0][s_value['ind']].axes[0].axvline(m_value, color=self.colors[m_key], linestyle='dashed')

                # look at population and proportion that are interneurons/pyramidal/goal-selective
                proportion_data = (g_data[['update_selectivity_type', 'session_id', s_value['group']]]
                                   .groupby(['update_selectivity_type', 'session_id'])
                                   .value_counts(normalize=True)
                                   .reset_index()
                                   .rename(columns={0: 'proportion'}))
                (
                    so.Plot(proportion_data, x='update_selectivity_type', y='proportion', color=s_value['group'])
                        .add(so.Dot(alpha=0.5), so.Dodge(), so.Jitter(0.5))
                        .add(so.Dash(), so.Agg(), so.Dodge())
                        .add(so.Range(), so.Est(errorbar='sd'), so.Dodge())
                        .theme(rcparams)
                        .scale(color=[self.colors[c] for c in proportion_data[s_value['group']].unique()])
                        .on(sfigs[1][s_value['ind']])
                        .label(title=s_key)
                        .plot()
                )

                info = g_data.groupby(['update_selectivity_type'])[s_value['group']].value_counts(normalize=True).to_dict()
                text = ''.join([f'{k}: {v:.2f}, {new_line}' for k, v in info.items()])
                sfigs[1][s_value['ind']].axes[0].text(0.75, 0.75, text, fontsize=6,
                                                      transform=sfigs[1][s_value['ind']].axes[0].transAxes,)
                leg = fig.legends.pop(0)
                sfigs[1][s_value['ind']].legend(leg.legendHandles, [t.get_text() for t in leg.texts], loc='upper right',
                                                fontsize='large')

            # save figure
            fig.suptitle(f'Goal selectivity - {g_name} - {metric}')
            self.results_io.save_fig(fig=fig, filename=f'update_selectivity_cell_types_{metric}', additional_tags=g_name,
                                     tight_layout=False)

    def plot_update_selectivity(self, g_data, g_name, cutoff=0.25, plot_dict=None):
        # plot heatmaps of cell spatial maps
        # note these are generated by the non-update trials so only have that data even when sort by switch/stay index
        plot_dict = plot_dict or self.update_plot_dict
        cutoffs = [cutoff, -cutoff]
        cmaps = ['cmap_r', 'cmap_r', 'div_cmap']
        linestyles = ['solid', 'dashed']
        g_data['goal_selectivity_type'] = ['switch' if x < 0 else 'stay' if x > 0 else 'home'
                                           for x in g_data['max_selectivity_index']]

        for metric, value in plot_dict.items():
            fig, axes = plt.subplots(3, 7, figsize=(20, 10), layout='constrained', sharex='col',
                                     gridspec_kw={'height_ratios': [1, 4, 4]})
            for row, cond in enumerate(value['conditions']):
                # plot psths for all cells sorted by selectivity
                data = (g_data
                        .query(f'{metric} {cond} {cutoffs[row]}')
                        .sort_values(metric, na_position='last')
                        .dropna(subset=[f'psth_{value["psths"][0]}', f'psth_{value["psths"][1]}'])
                        .reset_index(drop=True))

               # get zscore information from both psths
                end_ind = int(np.shape(data['psth_times'].values[0])[0] / 2) - 1
                data['zscore_mean'] = (data[[f'psth_{p}' for p in value['psths'][:2]]]  # get baseline from both periods pre-update
                               .apply(lambda x: np.nanmean(np.vstack(x)[:, :end_ind]), axis=1))
                data['zscore_std'] = (data[[f'psth_{p}' for p in value['psths'][:2]]]
                              .apply(lambda x: np.nanstd(np.vstack(x)[:, :end_ind]), axis=1))
                for p in value['psths'][:2]:
                    data[f'psth_{p}_zscore'] = (data.apply(lambda x: (x[f'psth_{p}'] - x['zscore_mean']) / x['zscore_std'], axis=1))
                data[f'psth_{value["psths"][-1]}_zscore'] = data.apply(lambda x: np.subtract(x[f'psth_{value["psths"][0]}_zscore'],
                                                                                             x[f'psth_{value["psths"][1]}_zscore']),
                                                                       axis=1)

                # remove any rows where cell has 0 max firing rate or 0 zscore std
                zero_max = (data[[f'psth_{p}' for p in value['psths'][:2]]]
                            .apply(lambda x: np.max(np.vstack(x), axis=1) == 0))
                data = data[~zero_max.any(axis=1)]
                zero_max = (data[[f'psth_{p}_zscore' for p in value['psths'][:2]]]
                            .apply(lambda x: np.isnan(np.vstack(x)[:, 0]), axis=1))
                data = data[~zero_max.apply(lambda x: x.any())]

                for col, p in enumerate(value['psths']):
                    psth_mat = np.stack(data[f'psth_{p}_zscore'].to_numpy())
                    vmin, vmax = -1.75, 1.75
                    y_limits = [0, np.shape(psth_mat)[0]]
                    x_limits = [data['psth_times'].values[0].min(), data['psth_times'].values[0].max()]
                    im = axes[row + 1][col].imshow(psth_mat, cmap=self.colors[cmaps[col]],
                                                        origin='lower', aspect='auto', vmin=vmin, vmax=vmax,
                                                        extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
                                                       )
                    axes[row + 1][col].set(xlabel='time around update', ylabel='unit')
                    axes[row + 1][col].axvline(0, linestyle='dashed', color='k')
                    plt.colorbar(im, ax=axes[row + 1][col], pad=0.04, location='right', fraction=0.046,
                                 label='Firing rate')

                    psth_mean = np.nanmean(psth_mat, axis=0)
                    psth_err = sem(psth_mat, axis=0, nan_policy='omit')
                    axes[0][col].plot(data['psth_times'].values[0], psth_mean, linestyle=linestyles[row])
                    axes[0][col].fill_between(data['psth_times'].values[0], psth_mean + psth_err,
                                                             psth_mean - psth_err, color='k', alpha=0.2)
                    axes[0][col].axvline(0, linestyle='dashed', color='k')
                    axes[0][col].set(title=f'{p} {cond} {cutoffs[row]}', ylabel='mean fr')

                # plot place fields for all cells
                conditions = dict(all=dict(col=3, sort_by=metric, group_by='region'),  # dummy groupby variable
                                  sorted=dict(col=4, sort_by='place_field_peak_ind', group_by='region'),
                                  cell_types=dict(col=5, sort_by=['cell_type', 'place_field_peak_ind'],
                                                  group_by='cell_type'),
                                  goal_types=dict(col=6, sort_by=['goal_selectivity_type', 'place_field_peak_ind'],
                                                  group_by='goal_selectivity_type'))
                for key, val in conditions.items():
                    t_data = data.sort_values(val['sort_by'], na_position='first')
                    tuning_curve_mat = np.stack(t_data['tuning_curves'].to_numpy())
                    tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
                    tuning_curve_bins = self.aggregator.group_df['tuning_bins'].to_numpy()[0]
                    y_limits = [0, np.shape(tuning_curve_scaled)[0]]
                    x_limits = [np.round(np.min(tuning_curve_bins), 2), np.round(np.max(tuning_curve_bins), 2)]
                    im_tuning = axes[row + 1][val['col']].imshow(tuning_curve_scaled, cmap=self.colors['cmap_r'],
                                                             origin='lower', vmin=0.1, vmax=0.9, aspect='auto',
                                                             extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]])
                    axes[row + 1][val['col']].set(title=f'{key} {new_line} sorted by {val["sort_by"]}', xlabel='position')

                    counter = 0
                    for s_name, s_data in t_data.groupby(val['group_by'], sort=False):
                        tuning_curve_mat = np.stack(s_data['tuning_curves'].to_numpy())
                        tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
                        tuning_mean = np.nanmean(tuning_curve_scaled, axis=0)
                        tuning_err = sem(tuning_curve_scaled, axis=0, nan_policy='omit')
                        tuning_bins = (tuning_curve_bins[1:] + tuning_curve_bins[:-1]) / 2
                        axes[0][val['col']].plot(tuning_bins, tuning_mean, color=self.colors[s_name],
                                                 linestyle=linestyles[row])
                        axes[0][val['col']].fill_between(tuning_bins, tuning_mean + tuning_err,
                                                         tuning_mean - tuning_err, color=self.colors[s_name], alpha=0.2,)

                        axes[row + 1][val['col']].axvspan(tuning_bins.min(), tuning_bins.min() + 0.05*(tuning_bins.max() - tuning_bins.min()), ymin=counter/len(t_data),
                                                              ymax=(counter + np.shape(tuning_curve_mat)[0]) / len(t_data),
                                                              color=self.colors[s_name])
                        counter = counter + np.shape(tuning_curve_mat)[0]

                    # plot annotation lines
                    locations = self.virtual_track.cue_end_locations.get(g_data['feature_name'].values[0],)
                    for k, v in locations.items():
                        axes[row + 1][val['col']].axvline(v, linestyle='dashed', color='k', alpha=0.5)

                    for s in self.virtual_track.edge_spacing:
                        axes[0][val['col']].axvspan(*s, color='#DDDDDD', edgecolor=None, zorder=10)
                        axes[row + 1][val['col']].axvspan(*s, color='#DDDDDD', edgecolor=None, zorder=10)

                plt.colorbar(im_tuning, ax=axes[row + 1][val['col']], pad=0.04, location='right', fraction=0.046,
                             label='Normalized firing rate')

            # save figure
            fig.suptitle(f'Goal selectivity - {g_name} - {metric}')
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'update_selectivity_{metric}_cutoff{cutoffs[0]}',
                                     additional_tags=g_name, tight_layout=False)

    def plot_goal_selectivity(self, g_data, plot_groups, tags):
        aligned_data = self.aggregator.select_group_aligned_data(g_data, {**plot_groups, 'time_label': ['t_update']})
        psth_data = self.aggregator.get_aligned_psth(aligned_data)
        psth_data.dropna(subset=['mean_selectivity_type', 'max_selectivity_type'], inplace=True)  # only selective
        psth_data.sort_values(by='max_selectivity_type', inplace=True)
        psth_data = psth_data[(psth_data['max_selectivity'] >= 0.25) | (psth_data['max_selectivity'] <= -0.25)]
        # heatmap + rasters for all trials for all units
        if np.size(psth_data):
            nrows, ncols = 10, 5
            fig = plt.figure(figsize=(17, 22), layout='constrained',)
            sfigs = fig.subfigures(nrows, ncols).flatten()
            i, total = 0, 0
            for g_name, g_data in psth_data.groupby(['session_id', 'unit_id'], sort=False):
                # average + heatmap of all cells
                times = g_data['psth_times'].to_numpy()[0]
                psth_mean = g_data['psth_mean'].to_numpy()[0]
                psth_err = g_data['psth_err'].to_numpy()[0]
                ax = sfigs[i].subplots(2, 1, sharex=True)
                ax[0].plot(times, psth_mean, color=self.colors[g_data['max_selectivity_type'].values[0]])
                ax[0].fill_between(times, psth_mean + psth_err, psth_mean - psth_err,
                                   color=self.colors[g_data['max_selectivity_type'].values[0]], alpha=0.3)

                show_psth_raster(g_data['spikes'].values[0], ax=ax[1], start=times[0], end=times[-1],
                                 group_inds=np.array([0] * np.shape(g_data['spikes'].values[0])[0]),
                                 labels=[g_data['max_selectivity_type'].values[0]],
                                 colors=[self.colors[g_data['max_selectivity_type'].values[0]]])

                [a.axvline(0, color='k', linestyle='dashed', alpha=0.5) for a in ax]
                ax[0].set(title=g_name)

                total += 1
                i += 1

                if i == nrows*ncols or total == len(psth_data.groupby(['session_id', 'unit_id'])):
                    fig.supxlabel('time_around_update (s)')
                    fig.supylabel('psth + rasters')
                    fig.suptitle(f'Goal selective rasters - {g_name} - {plot_groups}')
                    self.results_io.save_fig(fig=fig, filename=f'spiking_rasters_plot{round(total/(nrows*ncols))}',
                                             additional_tags=tags, tight_layout=False)

                    fig = plt.figure(figsize=(17, 22), layout='constrained', )
                    sfigs = fig.subfigures(nrows, ncols).flatten()
                    i = 0

    def plot_theta_phase_modulation(self, data, plot_groups, tags):
        # NOTE - Kay et al doesn't look at single unit spiking on different theta phases I think,
        # they just look at decoded probability densities
        # plot theta modulation for this trial
        theta_phase_data = self.aggregator.calc_theta_phase_data(data)
        time_labels = ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made']

        fig, axes = plt.subplots(2, len(theta_phase_data['time_label'].unique()), figsize=(10, 5),
                                 layout='constrained', gridspec_kw={'height_ratios': [1, 2]})
        for g_name, g_data in theta_phase_data.groupby(['time_label', 'new_times', 'max_selectivity_type'], dropna=False):
            col_ind = np.where(np.array(time_labels) == g_name[0])[0][0]
            axes[0][col_ind].plot(g_data['phase_mid'] / np.pi, g_data['theta_amplitude'], color='k', label='theta')
            axes[0][col_ind].fill_between((g_data['phase_mid'] / np.pi),
                                 g_data['theta_amplitude'] + g_data['theta_amplitude_err'],
                                 g_data['theta_amplitude'] - g_data['theta_amplitude_err'],
                                 color='k', alpha=0.2)
            axes[0][col_ind].set(ylabel='theta amplitude', title=g_name[0])

            lstyles = ['dashed', 'solid']
            lstyle = lstyles[np.where(np.array(['pre-update', 'post-update']) == g_name[1])[0][0]]
            axes[1][col_ind].plot((g_data['phase_mid'] / np.pi), g_data['spike_counts'],
                                                color=self.colors[str(g_name[2])], label=g_name[1:], linestyle=lstyle)
            axes[1][col_ind].fill_between((g_data['phase_mid'] / np.pi),
                                                        g_data['spike_counts'] + g_data['spike_counts_err'],
                                                        g_data['spike_counts'] - g_data['spike_counts_err'],
                                                        color=self.colors[str(g_name[2])], alpha=0.2)
            axes[1][col_ind].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g $\pi$'))
            axes[1][col_ind].xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
            axes[1][col_ind].relim()
            axes[1][col_ind].set(ylabel='mean spikes', xlabel='theta phase', )
            if col_ind == (len(theta_phase_data['time_label'].unique()) - 1):
                axes[1][col_ind].legend()

        fig.suptitle(f'Theta modulation of goal selective cells - {plot_groups}')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'spiking_theta_modulation',
                                 additional_tags=tags, tight_layout=False)

    def plot_units_aligned(self, g_data, plot_groups, tags):
        aligned_data = self.aggregator.select_group_aligned_data(g_data, plot_groups)
        sorting_data = self.aggregator.get_peak_sorting_index()
        psth_data = self.aggregator.get_aligned_psth(aligned_data)
        combined_data = psth_data.merge(sorting_data, on=['session_id', 'unit_id', 'feature_name'], how='left')
        combined_data.sort_values(by=['peak_sort_index'], inplace=True)

        if np.size(combined_data):
            for s_type in ['mean_selectivity_type', 'max_selectivity_type']:
                fig, axes = plt.subplots(9, len(g_data['time_label'].unique()), figsize=(17, 22), layout='constrained',
                                         sharex=True, sharey='row')
                for t_name, t_data in combined_data.groupby('time_label'):
                    col = np.argwhere(g_data['time_label'].unique() == t_name)[0][0]
                    psth_times = t_data['psth_times'].to_numpy()[0]
                    selectivity_dict = dict(all=dict(filter=[np.nan, 'switch', 'stay'], rows=[0, 1], color='all'),
                                            non_goal=dict(filter=[np.nan], rows=[2, 3], color='home'),
                                            switch=dict(filter=['switch'], rows=[2, 4], color='switch'),
                                            stay=dict(filter=['stay'], rows=[2, 5], color='stay'),
                                            switch_high=dict(filter=['switch'], rows=[6, 7], color='switch'),
                                            stay_high=dict(filter=['stay'], rows=[6, 8], color='stay'))
                    for key, value in selectivity_dict.items():
                        plot_data = t_data[t_data[s_type].isin(value['filter'])]
                        if key in ['switch_high', 'stay_high']:
                            plot_data = plot_data[(plot_data[s_type.replace('_type', '')] >= 0.25)
                                                  | (plot_data[s_type.replace('_type', '')] <= -0.25)]
                        psth_mat_all = np.vstack(plot_data['psth_mean'])
                        psth_mat_all = psth_mat_all[~np.isnan(psth_mat_all).any(axis=1), :]
                        psth_mean = np.nanmean(psth_mat_all, axis=0)
                        psth_sem = sem(psth_mat_all, axis=0, nan_policy='omit')

                        # average + heatmap of all cells
                        axes[value['rows'][0]][col].plot(psth_times, np.nanmean(psth_mat_all, axis=0),
                                                         color=self.colors[value['color']])
                        axes[value['rows'][0]][col].fill_between(psth_times, psth_mean + psth_sem, psth_mean - psth_sem,
                                                                 color=self.colors[value['color']], alpha=0.3)
                        axes[value['rows'][0]][col].set(ylabel='mean fr', ylim=[-0.1, 0.5])

                        im = axes[value['rows'][1]][col].imshow(psth_mat_all, cmap=self.colors[f'{value["color"]}_cmap'],
                                                                origin='lower', aspect='auto', vmin=-0.15, vmax=1,
                                                                extent=[psth_times[0], psth_times[-1],
                                                                        0, np.shape(psth_mat_all)[0]])
                        axes[value['rows'][1]][col].set(ylabel='units', ylim=(0, np.shape(psth_mat_all)[0]))

                        if value['rows'][0] == 0:
                            axes[value['rows'][0]][col].set(title=t_name)
                        if col == len(g_data['time_label'].unique()) - 1:
                            plt.colorbar(im, ax=axes[value['rows'][1]], pad=0.04, location='right', fraction=0.046,
                                         label=f'{key} mean fr')

                add_lines = [a.axvline(0, color='k', linestyle='dashed', alpha=0.5) for a in axes.flatten()]

                # save figure
                fig.supxlabel('time around cue (s)')
                fig.suptitle(f'Aligned goal selective activity - {plot_groups}')
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'spiking_aligned_{s_type}',
                                         additional_tags=tags, tight_layout=False)

    def plot_movement_reaction_times(self, param_data, plot_groups=None, tags=''):
        reaction_data = self.aggregator.calc_movement_reaction_times(param_data, {'time_label': ['t_update'], **plot_groups})
        if np.size(reaction_data):
            # reaction_data_by_time = reaction_data.explode(['new_times', 'rotational_velocity', 'veloc_diff'])
            lims = dict()
            for col in ['rotational_velocity', 'veloc_diff']:
                mag = np.min(np.abs((reaction_data[col].apply(np.min).min(), reaction_data[col].apply(np.max).max())))
                lims[col] = (-mag, mag)

            ncols, nrows = (1, 4)
            fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20), constrained_layout=True, squeeze=False,
                                     sharex='col')

            # plot heatmaps of rotational velocity with derivative overlay
            for name, data in reaction_data.groupby(['time_label'], sort=False):
                col = np.argwhere(reaction_data['time_label'].unique() == name)[0][0]
                data.sort_values('reaction_time', inplace=True, ascending=False)

                rot = np.stack(data['rotational_velocity'])
                times = data['new_times'].to_numpy()[0]
                im_rot = axes[0][col].imshow(rot, cmap=self.colors['div_cmap'], aspect='auto',
                                             vmin=lims['rotational_velocity'][0] * 0.5,
                                             vmax=lims['rotational_velocity'][-1] * 0.5,
                                             origin='lower', extent=[times[0], times[-1], 0, np.shape(rot)[0]], )
                axes[0][col].set(xlim=(times[0], times[-1]), ylim=(0, np.shape(rot)[0]), title=name)

                rot_diff = np.stack(data['veloc_diff'])
                im_diff = axes[1][col].imshow(rot_diff, cmap=self.colors['div_cmap'], aspect='auto',
                                              vmin=lims['veloc_diff'][0] * 0.5, vmax=lims['veloc_diff'][-1] * 0.5,
                                              origin='lower', extent=[times[0], times[-1], 0, np.shape(rot_diff)[0]], )
                axes[1][col].set(xlim=(times[0], times[-1]), ylim=(0, np.shape(rot_diff)[0]), title=name)
                if col == 0:
                    axes[0][col].set_ylabel('trials')
                    axes[1][col].set_ylabel('trials')

                means = np.nanmean(np.stack(data['rotational_velocity']), axis=0)
                errs = sem(np.stack(data['rotational_velocity']), axis=0, nan_policy='omit')
                hist = np.histogram(data['reaction_time'], range=(0, 2.5), bins=20)
                axes[2][col].plot(data['new_times'].to_numpy()[0], means, color='k')
                axes[2][col].fill_between(data['new_times'].to_numpy()[0],
                                   means + errs, means - errs, color='k', alpha=0.2)
                axes[2][col].set(xlim=(times[0], times[-1]), title=name)

                axes[3][col].fill_between((hist[1][1:] + hist[1][:-1]) / 2, hist[0], step='mid', color='k', alpha=0.2)
                axes[3][col].set(xlim=(times[0], times[-1]), title=name)

            plt.colorbar(im_rot, ax=axes[0][col], label='rotational velocity')
            plt.colorbar(im_diff, ax=axes[1][col], label='veloc_diff')

            # (  # plot average rotational velocity over time
            #     so.Plot(reaction_data_by_time, x='new_times', y='rotational_velocity')
            #         .facet(col='time_label')
            #         .add(so.Band(color='k'), so.Est(errorbar='se'))
            #         .add(so.Line(color='k', linewidth=2), so.Agg(), )
            #         .theme(rcparams)
            #         .layout(engine='constrained')
            #         .on(sfigs[1])
            #         .plot()
            # )

            # (  # plot distribution of movement
            #     so.Plot(reaction_data, x='reaction_time')
            #         .facet(col='time_label')
            #         .add(so.Bars(color='k'), so.Hist(stat='proportion', binrange=(0, 2.5), binwidth=0.25))
            #         .limit(x=(reaction_data['new_times'].apply(min).min(), reaction_data['new_times'].apply(max).max()))
            #         .label(y='proportion')
            #         .theme(rcparams)
            #         .scale(color=self.colors['control'])
            #         .layout(engine='constrained')
            #         .on(sfigs[2])
            #         .plot()
            # )

            medians = reaction_data.groupby('time_label', sort=False)['reaction_time'].median()
            for ind, ax in enumerate(axes):
                for a in ax:
                    med = medians.get(a.title.get_text(), np.nan)
                    a.axvline(0, color='k', linestyle='dashed', alpha=0.5)
                    a.axvline(med, color='purple', linestyle='dashed')
                    if ind == len(axes) - 1:  # add median number to last plot only
                        a.annotate(f'median: {med:.3f}', (0.65, 0.7), xycoords='axes fraction', xytext=(0.65, 0.7),
                                   textcoords='axes fraction', )

            # save figures
            fig.suptitle(tags)
            self.results_io.save_fig(fig=fig, filename=f'reaction_times', additional_tags=tags, tight_layout=False)