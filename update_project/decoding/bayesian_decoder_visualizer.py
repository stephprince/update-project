import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

from update_project.camera_sync.cam_plot_utils import write_camera_video
from update_project.general.utils import create_track_boundaries
from update_project.decoding.interpolate import griddata_time_intervals


class BayesianDecoderVisualizer:
    def __init__(self, data, type='session'):
        self.data = data
        self.type = type

        # get example times
        if self.type == 'session':
            times, locs = self._get_example_period()
            self.start_time = np.min(times)
            self.end_time = np.max(times)
            self.start_loc = np.min(locs)
            self.end_loc = np.max(locs)

        # calculate some of the basic input structures to plotting functions
        self.confusion_matrix = self._get_confusion_matrix()
        self.prob_density_grid = self._get_prob_density_grid()

    def plot(self):
        if self.type == 'session':
            print(f'Plotting data for session {self.data.results_io.session_id}...')
            self.plot_session_summary()
            self.plot_aligned_data()
        elif self.type == 'group':
            print(f'Plotting data for group')
            self.plot_group_summary()

    def _get_confusion_matrix(self):
        # get decoding matrices
        if self.data.convert_to_binary:
            bins = [-1, 0, 1]
        else:
            bins = self.data.bins
        bins = pd.cut(self.data.summary_df['actual_feature'], bins, include_lowest=True)
        decoding_matrix = self.data.summary_df['prob_dist'].groupby(bins).apply(
            lambda x: np.nanmean(np.vstack(x.values), axis=0)).values

        confusion_matrix = np.vstack(decoding_matrix).T  # transpose so that true position is on the x-axis

        return confusion_matrix

    def _get_example_period(self, window=500):
        time_window = self.data.decoder_bin_size*window
        times = [self.data.summary_df['decoding_error_rolling'].idxmin()]  # get times/locs with minimum error
        locs = [self.data.summary_df.index.searchsorted(times[0])]
        if (times[0] + time_window) < self.data.summary_df.index.max():
            locs.append(self.data.summary_df.index.searchsorted(times[0] + time_window))
        else:
            locs.append(self.data.summary_df.index.searchsorted(times[0] - time_window))
        times.append(self.data.summary_df.iloc[locs[1]].name)

        return times, locs

    def _get_prob_density_grid(self):
        nbins = int(self.end_loc - self.start_loc)
        trials_to_flip = self.data.test_data['turn_type'] == 100  # set all to false
        grid_prob = griddata_time_intervals(self.data.prob_densities, [self.start_loc], [self.end_loc], nbins,
                                            trials_to_flip, method='linear')

        return np.squeeze(grid_prob)

    def plot_session_summary(self):
        # plot the decoding data
        mosaic = """
           AAAAAA
           BBBBBB
           CCCCCC
           DDDEEE
           DDDEEE
           """
        axes = plt.figure(figsize=(15, 15)).subplot_mosaic(mosaic)
        label = self.data.feature_names[0]
        range = self.data.bins[-1] - self.data.bins[0]

        im = axes['A'].imshow(self.prob_density_grid, aspect='auto', origin='lower', cmap='YlGnBu', vmin=0, vmax=0.75,
                              extent=[self.start_time, self.end_time, self.data.bins[0], self.data.bins[-1]], )
        axes['A'].plot(self.data.features_test.loc[self.start_time:self.end_time], label='True', color=[0, 0, 0, 0.5],
                       linestyle='dashed')
        axes['A'].set(xlim=[self.start_time, self.end_time], ylim=[self.data.bins[0], self.data.bins[-1]], xlabel='Time (s)', ylabel=label)
        axes['A'].set_title(f'Bayesian decoding - {self.data.results_io.session_id} - example period', fontsize=14)
        axes['A'].legend(loc='upper right')

        axes['B'].plot(self.data.features_test.loc[self.start_time:self.end_time], color=[0, 0, 0, 0.5], label='True')
        axes['B'].plot(self.data.decoded_values.loc[self.start_time:self.end_time], color='b', label='Decoded')
        axes['B'].set(xlim=[self.start_time, self.end_time], ylim=[self.data.bins[0], self.data.bins[-1]], xlabel='Time (s)', ylabel=label)
        axes['B'].legend(loc='upper right')

        axes['C'].plot(self.data.summary_df['decoding_error'].loc[self.start_time:self.end_time], color=[0, 0, 0],
                       label='True')
        axes['C'].set(xlim=[self.start_time, self.end_time], ylim=[0, range], xlabel='Time (s)', ylabel='Decoding error')

        tick_values = self.data.model.index.values.astype(int)
        tick_labels = np.array([0, int(len(tick_values) / 2), len(tick_values) - 1])
        axes['D'] = sns.heatmap(self.confusion_matrix, cmap='YlGnBu', ax=axes['D'], square=True,
                                vmin=0, vmax=0.5 * np.nanmax(self.confusion_matrix),
                                cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
        axes['D'].plot([0, 285], [0, 285], linestyle='dashed', color=[0, 0, 0, 0.5])
        axes['D'].invert_yaxis()
        axes['D'].set_title('Decoding accuracy ', fontsize=14)
        axes['D'].set(xticks=tick_labels, yticks=tick_labels,
                      xticklabels=tick_values[tick_labels], yticklabels=tick_values[tick_labels],
                      xlabel=f'True {label}', ylabel=f'Decoded {label}')

        axes['E'] = sns.ecdfplot(self.data.summary_df['decoding_error'], ax=axes['E'], color='k')
        axes['E'].set_title('Decoding accuracy - error')
        axes['E'].set(xlabel='Decoding error', ylabel='Proportion')
        axes['E'].set_aspect(1. / axes['E'].get_data_ratio(), adjustable='box')

        plt.colorbar(im, ax=axes['A'], label='Probability density', pad=0.06, location='bottom', shrink=0.25,
                     anchor=(0.9, 1))
        plt.tight_layout()

        # save figure
        kwargs = self.data.results_io.get_figure_args(f'decoding_summary_{label}', results_type='session')
        plt.savefig(**kwargs)

        plt.close('all')

    def plot_aligned_data(self):
        window = self.data.aligned_data_window
        nbins = self.data.aligned_data_nbins
        if self.data.dim_num == 1:
            mosaic = """
            AACC
            EEGG
            IIKK
            MMOO
            QQSS
            """
            axes = plt.figure(figsize=(20, 15)).subplot_mosaic(mosaic)
            feature_name = self.data.feature_names[0]
            self.plot_1d_around_update(self.data.aligned_data['switch'][feature_name], nbins, window, 'switch',
                                       feature_name, 'b', axes, ['A', 'E', 'I', 'M', 'Q'])
            self.plot_1d_around_update(self.data.aligned_data['stay'][feature_name], nbins, window, 'stay',
                                       feature_name, 'm', axes, ['C', 'G', 'K', 'O', 'S'])
            plt.suptitle(f'{self.data.results_io.session_id} decoding around update trials', fontsize=20)
            plt.tight_layout()

            kwargs = self.data.results_io.get_figure_args(f'decoding_around_update_{feature_name}', results_type='session')
            plt.savefig(**kwargs)
        elif self.data.dim_num == 2:
            # plot 2d decoding around update
            times = np.linspace(-window, window, num=nbins)
            plot_filenames = []
            for time_bin in range(nbins):
                mosaic = """
                    AB
                    CD
                    """
                axes = plt.figure(figsize=(16, 8)).subplot_mosaic(mosaic)
                self.plot_2d_around_update(self.data.aligned_data['switch'], time_bin, times, 'switch', 'b', axes,
                                           ['A', 'B'])
                self.plot_2d_around_update(self.data.aligned_data['stay'], time_bin, times, 'stay', 'm', axes,
                                           ['C', 'D'])
                plt.suptitle(f'{self.data.results_io.session_id} decoding around update trials', fontsize=20)
                plt.tight_layout()

                filename = f'decoding_around_update_frame_no{time_bin}'
                kwargs = self.data.results_io.get_figure_args(fname=filename, format='pdf',
                                                              results_name='timelapse_figures')
                plot_filenames.append(kwargs['fname'])

                plt.savefig(**kwargs)
                plt.close()

            # make videos for each plot type
            vid_filename = f'decoding_around_update_video'
            results_path = self.data.results_io.get_results_path(results_type='session')
            write_camera_video(results_path, plot_filenames, vid_filename)

        plt.close('all')

    def plot_1d_around_update(self, data_around_update, nbins, window, title, label, color, axes,
                              ax_dict):

        limits = [np.min(np.min(data_around_update['feature'])), np.max(np.max(data_around_update['feature']))]
        stats = data_around_update['stats']
        times = np.linspace(-window, window, num=nbins)
        time_tick_values = times.astype(int)
        time_tick_labels = np.array([0, int(len(time_tick_values) / 2), len(time_tick_values) - 1])
        if self.data.feature_names[0] in ['x_position', 'view_angle', 'choice', 'turn_type']:  # divergent color maps for div data
            cmap_pos = 'RdGy'
            cmap_decoding = 'PRGn'
            scaling_value = 0.5
        elif self.data.feature_names[0] in ['y_position']:  # sequential color map for seq data
            cmap_pos = 'Greys'
            if title == 'switch':
                cmap_decoding = 'Blues'
            elif title == 'stay':
                cmap_decoding = 'RdPu'
            scaling_value = 1

        prob_map = np.nanmean(data_around_update['probability'], axis=0)
        if data_around_update['probability']:  # skip if there is no data
            n_position_bins = np.shape(prob_map)[0]
            data_values = np.linspace(np.min(data_around_update['decoding']), np.max(data_around_update['decoding']),
                                      n_position_bins)
            ytick_values = data_values.astype(int)
            ytick_labels = np.array([0, int(len(ytick_values) / 2), len(ytick_values) - 1])
            update_time_values = [[len(time_tick_values) / 2, len(time_tick_values) / 2],
                                  [0, np.shape(data_around_update['feature'])[1]]]
            pos_values_after_update = np.sum(
                data_around_update['feature'][time_tick_labels[1]:time_tick_labels[1] + 10],
                axis=0)
            sort_index = np.argsort(pos_values_after_update)

            axes[ax_dict[0]] = sns.heatmap(prob_map, cmap='YlGnBu', ax=axes[ax_dict[0]],
                                           vmin=0, vmax=0.75 * np.nanmax(prob_map),
                                           cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
            axes[ax_dict[0]].plot(update_time_values[0], update_time_values[1], linestyle='dashed',
                                  color=[0, 0, 0, 0.5])
            axes[ax_dict[0]].invert_yaxis()
            axes[ax_dict[0]].set(xticks=time_tick_labels, yticks=ytick_labels,
                                 xticklabels=time_tick_values[time_tick_labels], yticklabels=ytick_values[ytick_labels],
                                 xlabel='Time around update (s)', ylabel=f'{label}')
            axes[ax_dict[0]].set_title(f'{title} trials - probability density - {label}', fontsize=14)

            axes[ax_dict[1]].plot(times, stats['feature']['mean'], color='k', label='True position')
            axes[ax_dict[1]].fill_between(times, stats['feature']['lower'], stats['feature']['upper'], alpha=0.2,
                                          color='k', label='95% CI')
            axes[ax_dict[1]].plot(times, stats['decoding']['mean'], color=color, label='Decoded position')
            axes[ax_dict[1]].fill_between(times, stats['decoding']['lower'], stats['decoding']['upper'], alpha=0.2,
                                          color=color, label='95% CI')
            axes[ax_dict[1]].plot([0, 0], limits, linestyle='dashed', color='k', alpha=0.25)
            axes[ax_dict[1]].set(xlim=[-window, window], ylim=limits, xlabel='Time around update(s)',
                                 ylabel=f'{label}')
            axes[ax_dict[1]].legend(loc='upper left')
            axes[ax_dict[1]].set_title(f'{title} trials - decoded {label} position', fontsize=14)

            axes[ax_dict[2]] = sns.heatmap(data_around_update['feature'][:, sort_index].T, cmap=cmap_pos,
                                           ax=axes[ax_dict[2]],
                                           vmin=scaling_value * limits[0], vmax=scaling_value * limits[1],
                                           cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
            axes[ax_dict[2]].plot(update_time_values[0], update_time_values[1], linestyle='dashed',
                                  color=[0, 0, 0, 0.5])
            axes[ax_dict[2]].set(xticks=time_tick_labels, xticklabels=time_tick_values[time_tick_labels],
                                 xlabel='Time around update (s)', ylabel='Trials')
            axes[ax_dict[2]].set_title(f'{title} trials - true {label}', fontsize=14)

            axes[ax_dict[3]] = sns.heatmap(data_around_update['decoding'][:, sort_index].T, cmap=cmap_decoding,
                                           ax=axes[ax_dict[3]],
                                           vmin=scaling_value * limits[0], vmax=scaling_value * limits[1],
                                           cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
            axes[ax_dict[3]].plot(update_time_values[0], update_time_values[1], linestyle='dashed',
                                  color=[0, 0, 0, 0.5])
            axes[ax_dict[3]].set(xticks=time_tick_labels, xticklabels=time_tick_values[time_tick_labels],
                                 xlabel='Time around update (s)', ylabel='Trials')
            axes[ax_dict[3]].set_title(f'{title} trials - decoded {label}', fontsize=14)

            axes[ax_dict[4]].plot(times, stats['error']['mean'], color='r', label='|True - decoded|')
            axes[ax_dict[4]].fill_between(times, stats['error']['lower'], stats['error']['upper'], alpha=0.2, color='r',
                                          label='95% CI')
            axes[ax_dict[4]].plot([0, 0], [0, np.max(stats['error']['upper'])], linestyle='dashed', color='k',
                                  alpha=0.25)
            axes[ax_dict[4]].set(xlim=[-window, window], ylim=[0, np.max(stats['error']['upper'])],
                                 xlabel='Time around update(s)', ylabel=label)
            axes[ax_dict[4]].set_title(f'{title} trials - decoding error {label}', fontsize=14)
            axes[ax_dict[4]].legend(loc='upper left')

    def plot_2d_around_update(self, data_around_update, time_bin, times, title, color, axes, ax_dict):
        stats = data_around_update['stats']
        prob_map = np.nanmean(data_around_update['probability'], axis=0)
        if title == 'switch':
            correct_multiplier = -1
        elif title == 'stay':
            correct_multiplier = 1
        xlims = [-30, 30]
        ylims = [5, 285]
        track_bounds_xs, track_bounds_ys = create_track_boundaries()

        if data_around_update['probability']:  # skip if there is no data
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

            im = axes[ax_dict[1]].imshow(prob_map[:, :, time_bin], cmap='YlGnBu', origin='lower', aspect='auto',
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

    def plot_group_summary(self):
        # plot data for all sessions
        mosaic = """
        AABCC
        AADEE
        FFGHH
        FFIJJ
        """
        axes = plt.figure(figsize=(20, 18)).subplot_mosaic(mosaic)

        locations = {'x': {'left arm': -2, 'home arm': 2, 'right arm': 33},  # actually -1,+1 but add for bins
                     'y': {'initial cue': 120.35, 'delay cue': 145.35, 'update cue': 215.35, 'delay2 cue': 250.35,
                           'choice cue': 285}}

        pos_values = position_bins.astype(int)
        limits = [np.min(pos_values), np.max(pos_values)]
        matrix = np.vstack(decoding_matrix_prob) * nb_bins
        im = axes[ax_dict[0]].imshow(matrix.T, cmap='YlGnBu', origin='lower',  # aspect='auto',
                                     vmin=0, vmax=0.3 * np.nanmax(matrix),
                                     extent=[limits[0], limits[1], limits[0], limits[1]])
        text_offset = (limits[1] - limits[0]) / 15
        previous = np.min(pos_values)
        for key, value in locations.items():
            axes[ax_dict[0]].plot([value, value], [limits[0], limits[1]], linestyle='dashed', color=[0, 0, 0, 0.5])
            axes[ax_dict[0]].plot([limits[0], limits[1]], [value, value], linestyle='dashed', color=[0, 0, 0, 0.5])
            annotation_loc = np.mean([previous, value])
            axes[ax_dict[0]].annotate(f'{key}', (annotation_loc, limits[0]), xycoords='data',
                                      xytext=(annotation_loc, limits[0] - text_offset),
                                      textcoords='data', ha='center')
            axes[ax_dict[0]].annotate(f'{key}', (limits[0], annotation_loc), xycoords='data',
                                      xytext=(limits[0] - text_offset, annotation_loc),
                                      textcoords='data', ha='center')
            previous = value
        axes[ax_dict[0]].set_title(f'{title} decoding accuracy - avg prob', fontsize=14)
        axes[ax_dict[0]].set_xlabel(f'Actual {title} position', labelpad=20)
        axes[ax_dict[0]].set_ylabel(f'Decoded {title} position', labelpad=20)

        plt.colorbar(im, ax=axes[ax_dict[0]], label='probability / chance', pad=0.04, location='right',
                     fraction=0.046)

        # plot box/violin plot of decoding errors across sessions
        h_offset = 0.15
        axes[ax_dict[1]] = sns.violinplot(data=all_decoding_data, y='decoding_error', color='k',
                                          ax=axes[ax_dict[1]])
        plt.setp(axes[ax_dict[1]].collections, alpha=.25)
        axes[ax_dict[1]].set_title(f'Group error - {title} position')
        axes[ax_dict[1]].set_ylim([0, limits[1] - limits[0]])
        axes[ax_dict[1]].annotate(f"median: {all_decoding_data['decoding_error'].median():.2f}",
                                  (h_offset, limits[1] - limits[0] - text_offset * 2),
                                  textcoords='data', ha='left')
        axes[ax_dict[1]].annotate(f"mean: {all_decoding_data['decoding_error'].mean():.2f}",
                                  (h_offset, limits[1] - limits[0] - text_offset),
                                  textcoords='data', ha='left')

        axes[ax_dict[2]] = sns.violinplot(data=all_decoding_data, y='decoding_error', x='animal', palette='husl',
                                          ax=axes[ax_dict[2]])
        plt.setp(axes[ax_dict[2]].collections, alpha=.25)
        axes[ax_dict[2]].set_title(f'Individual error - {title} position')
        axes[ax_dict[2]].set_ylim([0, limits[1] - limits[0]])
        medians = all_decoding_data.groupby(['animal'])['decoding_error'].median()
        means = all_decoding_data.groupby(['animal'])['decoding_error'].mean()
        for xtick in axes[ax_dict[2]].get_xticks():
            axes[ax_dict[2]].annotate(f"median: {medians.iloc[xtick]:.2f}",
                                      (xtick + h_offset, limits[1] - limits[0] - text_offset * 2),
                                      textcoords='data', ha='left')
            axes[ax_dict[2]].annotate(f"mean: {means.iloc[xtick]:.2f}",
                                      (xtick + h_offset, limits[1] - limits[0] - text_offset), textcoords='data',
                                      ha='left')

        axes[ax_dict[3]] = sns.violinplot(data=session_rmse, y='session_rmse', color='k', ax=axes[ax_dict[3]])
        plt.setp(axes[ax_dict[3]].collections, alpha=.25)
        sns.stripplot(data=session_rmse, y='session_rmse', color="k", size=3, jitter=True, ax=axes[ax_dict[3]])
        axes[ax_dict[3]].set_title(f'Group session RMSE - {title} position')
        axes[ax_dict[3]].set_ylim([0, limits[1] - limits[0]])
        axes[ax_dict[3]].annotate(f"median: {session_rmse['session_rmse'].median():.2f}",
                                  (h_offset, limits[1] - limits[0] - text_offset * 2),
                                  textcoords='data', ha='left')
        axes[ax_dict[3]].annotate(f"mean: {session_rmse['session_rmse'].mean():.2f}",
                                  (h_offset, limits[1] - limits[0] - text_offset),
                                  textcoords='data', ha='left')

        axes[ax_dict[4]] = sns.violinplot(data=session_rmse, y='session_rmse', x='animal', palette='husl',
                                          ax=axes[ax_dict[4]])
        plt.setp(axes[ax_dict[4]].collections, alpha=.25)
        sns.stripplot(data=session_rmse, y='session_rmse', x='animal', color="k", size=3, jitter=True,
                      ax=axes[ax_dict[4]])
        axes[ax_dict[4]].set_title(f'Individual session RMSE - {title} position')
        axes[ax_dict[4]].set_ylim([0, limits[1] - limits[0]])
        medians = session_rmse.groupby(['animal'])['session_rmse'].median()
        means = session_rmse.groupby(['animal'])['session_rmse'].mean()
        for xtick in axes[ax_dict[4]].get_xticks():
            axes[ax_dict[4]].annotate(f"median: {medians.iloc[xtick]:.2f}",
                                      (xtick + h_offset, limits[1] - limits[0] - text_offset * 2),
                                      textcoords='data', ha='left')
            axes[ax_dict[4]].annotate(f"mean: {means.iloc[xtick]:.2f}",
                                      (xtick + h_offset, limits[1] - limits[0] - text_offset), textcoords='data',
                                      ha='left')

        plt.suptitle(f'Group decoding accuracy - non update trials only', fontsize=20)
        plt.tight_layout()

        kwargs = self.data.results_io.get_figure_args(fname='decoding_error_summary_position',
                                                      session_id=session_id,
                                                      format='pdf')
        plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})
