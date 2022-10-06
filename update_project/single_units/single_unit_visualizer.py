import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so

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

        self.aggregator = SingleUnitAggregator()
        self.aggregator.run_aggregation(data)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def plot(self):
        self.plot_place_fields()
        self.plot_unit_selectivity()
        self.plot_units_aligned()

    def plot_place_fields(self):
        # 1 plot of all cells (sorted by peak), 1 of left selective/right selective only, 1 of selectivity dist in both
        # 1 figure per brain region
        for g_name, g_data in self.aggregator.group_tuning_curves.groupby('region'):
            fig, axes = plt.subplots(2, 2, figsize=(8.5, 11), layout='constrained')
            ax = axes.flatten()

            # plot heatmaps of cell spatial maps
            for ind, condition in enumerate(['< 0', '>= 0', '!= 100']):  # right selective, left selective, any cells
                data = g_data.query(f'selectivity_index {condition}')
                cols_to_skip = ['session_id', 'animal', 'feature_name', 'unit_id', 'region', 'cell_type',
                                'selectivity_index', 'place_field_threshold']
                tuning_curve_mat = np.stack(data[data.columns.difference(cols_to_skip)].to_numpy())
                tuning_curve_scaled = tuning_curve_mat / np.nanmax(tuning_curve_mat, axis=1)[:, None]
                sort_index = np.argsort(np.argmax(tuning_curve_scaled, axis=1))
                tuning_curve_bins = self.aggregator.group_df['tuning_bins'].to_numpy()[0]

                y_limits = [0, np.shape(tuning_curve_scaled)[0]]
                x_limits = [np.round(np.min(tuning_curve_bins), 2), np.round(np.max(tuning_curve_bins), 2)]
                im = ax[ind].imshow(tuning_curve_scaled[sort_index, :], cmap=self.colors['cmap_r'],
                                                 origin='lower', vmin=0.1, vmax=0.9, aspect='auto',
                                                 extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]])

                # plot annotation lines
                locations = self.virtual_track.cue_end_locations.get(g_data['feature_name'].values[0], dict())  #TODO - generalize
                for key, value in locations.items():
                    ax[ind].axvline(value, linestyle='dashed', color='k', alpha=0.5)
                ax[ind].set(xlim=x_limits, ylim=y_limits, xlabel='units', ylabel='position',
                            title=f'selectivity index {condition}')

            plt.colorbar(im, ax=ax[ind], pad=0.04, location='right', fraction=0.046,
                         label='Normalized firing rate')

            # plot distribution of selectivity
            ax[ind + 1].hist(g_data['selectivity_index'].dropna().to_numpy(), bins=np.linspace(-1, 1, 20))
            ax[ind + 1].set(xlabel='selectivity_index', ylabel='count', title='distribution of goal selectivity')
            ax[ind + 1].axvline(0, linestyle='dashed', color='k', alpha=0.5)

            fig.suptitle(f'Goal selectivity - {g_name}')
            plt.show()

            # save figure
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves', additional_tags=g_name,
                                     tight_layout=False)

    def plot_unit_selectivity(self):
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'goal_selectivity', additional_tags=tags)


    def plot_units_aligned(self):
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'spiking_aligned', additional_tags=tags)
