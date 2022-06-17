import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from pathlib import Path

from update_project.camera_sync.cam_plot_utils import write_camera_video
from update_project.general.utils import get_track_boundaries, get_cue_locations
from update_project.decoding.interpolate import griddata_time_intervals
from update_project.results_io import ResultsIO

class BayesianDecoderVisualizer:
    def __init__(self, data, type='session'):
        self.data = data
        self.type = type
        self.data_exists = True

        # get session visualization info
        if self.type == 'session':
            if not self.data.summary_df.empty:
                times, locs = self._get_example_period()
                self.start_time, self.end_time = times
                self.start_loc, self.end_loc = locs
                self.prob_density_grid = self._get_prob_density_grid()  # prob density plot for example period

                # calculate confusion matrix for session
                self.confusion_matrix = self._get_confusion_matrix(data)
            else:
                self.data_exists = False
                warnings.warn(f'No summary dataframe found for {self.data.results_io.session_id}, skipping...')

        # get group visualization
        if self.type == 'group':
            for sess_dict in self.data:  # this could probably be done with dataframe group not with loop
                sess_dict.update(confusion_matrix=self._get_confusion_matrix(sess_dict['decoder']))

                session_error = self._get_session_error(sess_dict['decoder'])
                sess_dict.update(confusion_matrix_sum=session_error['confusion_matrix_sum'])
                sess_dict.update(rmse=session_error['rmse'])
                sess_dict.update(raw_error=session_error['raw_error_median'])

                sess_dict.update(excluded_session=sess_dict['decoder'].excluded_session)

            group_df = pd.DataFrame(data)
            self.group_df = group_df[group_df['excluded_session'] == False]  # only keep non-excluded sessions
            self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def plot(self, group_by=None):
        if self.data_exists:
            if self.type == 'session':
                self._plot_session_data()
            elif self.type == 'group':
                self._plot_group_data(group_by)
        else:
            print(f'No data found to plot')

    def _plot_session_data(self):
        print(f'Plotting data for session {self.data.results_io.session_id}...')
        self.plot_session_summary()
        self.plot_aligned_data()

    def _plot_group_data(self, group_by):
        params = ['units_threshold', 'trials_threshold']

        # make plots inspecting errors across all groups
        self.plot_all_groups_error(main_group='feature', sub_group='units_threshold')
        self.plot_all_groups_error(main_group='feature', sub_group='trials_threshold')
        self.plot_all_groups_error(main_group='feature', sub_group='region', thresholds=params)  # thresholds to use
        self.plot_all_groups_error(main_group='region', sub_group='feature', thresholds=params)

        # make plots for each individual subgroup
        self.groups = group_by
        group_data = self.group_df.groupby(list(group_by.keys()))
        for name, data in group_data:
            print(f'Plotting data for group {name}...')
            self.plot_group_confusion_matrices(data)  # plot all the heatmaps for all individual animals
            self.plot_parameter_comparison(data, name, params=params)

    @staticmethod
    def _get_mean_prob_dist(probabilities, bins):
        if probabilities.empty:
            mean_prob_dist = np.empty(np.shape(bins[1:]))  # remove one bin to match other probabilities shapes
            mean_prob_dist[:] = np.nan
        else:
            mean_prob_dist = np.nanmean(np.vstack(probabilities.values), axis=0)

        return mean_prob_dist

    @staticmethod
    def _get_group_summary_df(group_data):
        summary_df_list = []
        for _, row in group_data.iterrows():
            summary_df_list.append(row['decoder'].summary_df)

        group_summary_df = pd.concat(summary_df_list, axis=0, ignore_index=True)

        return group_summary_df

    def _get_confusion_matrix(self, data):
        # get decoding matrices
        if data.convert_to_binary:
            bins = [-1, 0, 1]
        else:
            bins = data.bins

        if len(bins):
            df_bins = pd.cut(data.summary_df['actual_feature'], bins, include_lowest=True)
            decoding_matrix = data.summary_df['prob_dist'].groupby(df_bins).apply(
                lambda x: self._get_mean_prob_dist(x, bins)).values

            confusion_matrix = np.vstack(decoding_matrix).T  # transpose so that true position is on the x-axis
        else:
            confusion_matrix = []

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
        times.sort()
        locs.sort()

        if np.max(locs) - np.min(locs) <= 1:
            times, locs = self._get_example_period(window=1000)

        return times, locs

    def _get_prob_density_grid(self):
        # something still off - the big jumps are taking up too any bins, everything is a little slow
        # check that end time is getting us all the way
        nbins = int((self.end_time - self.start_time) / self.data.decoder_bin_size)
        trials_to_flip = self.data.test_df['turn_type'] == 100  # set all to false
        time_bins = np.linspace(self.start_time, self.end_time, nbins)  # time bins
        grid_prob = griddata_time_intervals(self.data.decoded_probs, [self.start_loc], [self.end_loc], nbins,
                                            trials_to_flip, method='linear', time_bins=time_bins)

        return np.squeeze(grid_prob)

    def _get_session_error(self, data):
        # get error from heatmap
        confusion_matrix = self._get_confusion_matrix(data)
        values_to_sum = []
        for i in range(len(confusion_matrix)):
            values_to_sum.append(confusion_matrix[i, i])  # identity line values
            if i < len(confusion_matrix) - 1:
                values_to_sum.append(confusion_matrix[i+1, i])  # one above
                values_to_sum.append(confusion_matrix[i, i+1])  # one to right
        confusion_matrix_sum = np.nansum(values_to_sum)

        # get error from
        rmse = data.summary_df['session_rmse'].mean()
        raw_error_median = data.summary_df['decoding_error'].median()
        session_error = dict(confusion_matrix_sum=confusion_matrix_sum,
                             rmse=rmse,
                             raw_error_median=raw_error_median)

        return session_error

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
        kwargs = self.data.results_io.get_figure_args(f'decoding_summary', results_type='session')
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

            kwargs = self.data.results_io.get_figure_args(f'decoding_around_update', results_type='session')
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
        if data_around_update['feature'].any():
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
        track_bounds_xs, track_bounds_ys = get_track_boundaries()

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

    def plot_group_summary(self, group_data):
        # plot data for all sessions
        mosaic = """
        AABCC
        AADEE
        FFGHH
        FFIJJ
        """

        for feat in self.features:
            matrix = np.vstack(decoding_matrix_prob) * self.data.encoder_bin_num

            axes = plt.figure(figsize=(20, 18)).subplot_mosaic(mosaic)

            locations = get_cue_locations()
            pos_values = self.data.bins.astype(int)
            limits = [np.min(pos_values), np.max(pos_values)]

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

            kwargs = self.data.results_io.get_figure_args(fname='decoding_error_summary_position', format='pdf')
            plt.savefig(**kwargs)
            plt.close()

    def plot_group_error(self, data):
        all_summary_df = self._get_group_summary_df(data)
        group_confusion_matrix = self._get_confusion_matrix(all_summary_df)

        # plot confusion matrix
        matrix = group_confusion_matrix * data.iloc[0].encoder_bin_num  # scale to be probability/chance
        locations = get_cue_locations().get(data['feature'], dict())  # don't annotate graph if no locations indicated
        limits = [np.min(sess_data.bins.astype(int)), np.max(sess_data.bins.astype(int))]
        im = axes[col_id][row_id].imshow(matrix, cmap='YlGnBu', origin='lower',  # aspect='auto',
                                         vmin=0, vmax=5,
                                         extent=[limits[0], limits[1], limits[0], limits[1]])

        # plot annotation lines
        for key, value in locations.items():
            axes[col_id][row_id].plot([value, value], [limits[0], limits[1]], linestyle='dashed',
                                      color=[0, 0, 0, 0.5])
            axes[col_id][row_id].plot([limits[0], limits[1]], [value, value], linestyle='dashed',
                                      color=[0, 0, 0, 0.5])

        # add labels
        axes[col_id][row_id].set_title(f'{sess_key}')
        axes[col_id][row_id].set_xlim(limits)
        axes[col_id][row_id].set_ylim(limits)
        if col_id == (nrows - 1):
            axes[col_id][row_id].set_xlabel(f'Actual')
        if row_id == 0:
            axes[col_id][row_id].set_ylabel(f'Decoded')
        if row_id == (ncols - 1):
            plt.colorbar(im, ax=axes[col_id][row_id], pad=0.04, location='right', fraction=0.046,
                         label='probability / chance')

        for _, row in data.iterrows():
            test = 1

    def plot_all_groups_error(self, main_group, sub_group, thresholds=None):
        # select no threshold data to plot if thresholds indicated, otherwise combine
        if thresholds:
            thresh_mask = pd.concat([group_df[t] == 0 for t in thresholds], axis=1).all(axis=1)
            df = group_df[thresh_mask]
        else:
            df = group_df

        group_data = df.groupby(main_group)  # main group is what gets the different plots
        for name, data in group_data:
            nrows = 3  # 1 row for each plot type (cum fract, hist, violin)
            ncols = 3  # 1 column for RMSE dist, 1 for error dist, 1 for confusion_matrix dist
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

            # raw decoding errors
            title = 'Median raw error - all sessions'
            xlabel = 'Decoding error (|true - decoded|)'
            self.plot_distributions(data, axes=axes, column_name='raw_error', group=sub_group, row_ids=[0, 1, 2],
                                    col_ids=[0, 0, 0], xlabel=xlabel, title=title)

            # rmse
            title = 'Root mean square error - all sessions'
            xlabel = 'RMSE'
            self.plot_distributions(data, axes=axes, column_name='rmse', group=sub_group, row_ids=[0, 1, 2],
                                    col_ids=[1, 1, 1], xlabel=xlabel, title=title)

            # confusion matrix sums
            title = 'Confusion matrix sum - all sessions'
            xlabel = 'Probability'
            self.plot_distributions(data, axes=axes, column_name='confusion_matrix_sum', group=sub_group, row_ids=[0, 1, 2],
                                    col_ids=[2, 2, 2], xlabel=xlabel, title=title)

            # wrap up and save plot
            fig.suptitle(f'Decoding error - all sessions - {name}', fontsize=14)
            plt.tight_layout()
            kwargs = self.results_io.get_figure_args(filename=f'group_error', additional_tags=f'{name}_{sub_group}', format='pdf')
            plt.savefig(**kwargs)
            plt.close()

    def plot_parameter_comparison(self, data, name, params):
        nrows = len(list(itertools.combinations(params, r=2)))  # 1 row for each parameter combo
        ncols = 3  # 1 column for RMSE dist, 1 for error dist, 1 for confusion_matrix dist
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 5), squeeze=False)
        error_metrics = dict(raw_error=0, rmse=1, confusion_matrix_sum=2)
        params_dict = {p:ind for ind, p in enumerate(params)}

        # plot heatmaps comparing parameters (1 heatmap/2 parameters)
        medians = data.groupby(params).median().reset_index()
        row = 0
        for thresh1, thresh2 in itertools.combinations(params, r=2):
            for err, col in error_metrics.items():
                df = medians.pivot(thresh1, thresh2, err)
                axes[row][col] = sns.heatmap(df, annot=True, fmt='.2f', ax=axes[row][col], cmap='mako_r', square=True,
                                             cbar_kws={'pad': 0.01, 'label': err, 'fraction': 0.046},
                                             annot_kws={'size': 10}, )
                axes[row][col].set_title(f'Parameter median {err}')
                axes[row][col].invert_yaxis()
            row += 1


        # wrap up and save plot
        fig.suptitle(f'Parameter comparison - median all sessions - {name}', fontsize=14)
        plt.tight_layout()
        tags = '_'.join([''.join(n) for n in name])
        kwargs = self.results_io.get_figure_args(filename=f'group_param_comparison', additional_tags=tags, format='pdf')
        plt.savefig(**kwargs)
        plt.close()

    @staticmethod
    def plot_distributions(data, axes, column_name, group, row_ids, col_ids, xlabel, title):
        # cum fraction plots
        axes[row_ids[0]][col_ids[0]] = sns.ecdfplot(data=data, x=column_name, hue=group, ax=axes[row_ids[0]][col_ids[0]],
                                                    palette='husl')
        axes[row_ids[0]][col_ids[0]].set_title(title)
        axes[row_ids[0]][col_ids[0]].set(xlabel=xlabel, ylabel='Proportion')
        axes[row_ids[0]][col_ids[0]].set_aspect(1. / axes[row_ids[0]][col_ids[0]].get_data_ratio(), adjustable='box')

        # add median annotations to the first plot
        medians = data.groupby([group])[column_name].median()
        new_line = '\n'
        median_text = [f"{g} median: {m:.2f} {new_line}" for g, m in medians.items()]
        axes[row_ids[0]][col_ids[0]].text(0.55, 0.2, ''.join(median_text),
                                          transform=axes[row_ids[0]][col_ids[0]].transAxes, verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # histograms
        axes[row_ids[1]][col_ids[1]] = sns.histplot(data=data, x=column_name, hue=group, ax=axes[row_ids[1]][col_ids[1]],
                                                    palette='husl', element='step')
        axes[row_ids[1]][col_ids[1]].set(xlabel=xlabel, ylabel='Proportion')

        # violin plots
        axes[row_ids[2]][col_ids[2]] = sns.violinplot(data=data, x=group, y=column_name, ax=axes[row_ids[2]][col_ids[2]],
                                                      palette='husl')
        plt.setp(axes[row_ids[2]][col_ids[2]].collections, alpha=.25)
        sns.stripplot(data=data, y=column_name, x=group, size=3, jitter=True, ax=axes[row_ids[2]][col_ids[2]],
                      palette='husl')
        axes[row_ids[2]][col_ids[2]].set_title(title)

    @staticmethod
    def plot_group_confusion_matrices(data):
        plot_num = 0
        counter = 0
        ncols, nrows = [6, 3]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))
        for _, row in data.iterrows():
            sess_matrix = row['confusion_matrix']
            sess_data = row['decoder']
            sess_key = row['session_id']
            vmax = 5  # default to 5 vmax probability/chance
            if row['feature'] in ['choice', 'turn_type']:
                vmax = 2

            if (counter % (ncols * nrows) == 0) and (counter != 0):
                fig.suptitle(f'Confusion matrices - all sessions - {sess_data.results_tags}', fontsize=14)
                plt.tight_layout()
                kwargs = sess_data.results_io.get_figure_args(f'group_confusion_matrices',
                                                              additional_tags=f'plot{plot_num}', format='pdf')
                plt.savefig(**kwargs)
                plt.close()

                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))
                counter = 0
                plot_num += 1
            else:
                row_id = int(np.floor(counter / ncols))
                col_id = counter - row_id * ncols

                # plot confusion matrix
                if hasattr(sess_data.bins, 'astype'):  # if the matrix exists
                    matrix = np.vstack(sess_matrix) * sess_data.encoder_bin_num  # scale to be probability/chance
                    locations = get_cue_locations().get(row['feature'], dict())  # don't annotate graph if no locations indicated
                    limits = [np.min(sess_data.bins.astype(int)), np.max(sess_data.bins.astype(int))]
                    im = axes[row_id][col_id].imshow(matrix, cmap='YlGnBu', origin='lower',  # aspect='auto',
                                                     vmin=0, vmax=vmax,
                                                     extent=[limits[0], limits[1], limits[0], limits[1]])

                    # plot annotation lines
                    for key, value in locations.items():
                        axes[row_id][col_id].plot([value, value], [limits[0], limits[1]], linestyle='dashed',
                                                  color=[0, 0, 0, 0.5])
                        axes[row_id][col_id].plot([limits[0], limits[1]], [value, value], linestyle='dashed',
                                                  color=[0, 0, 0, 0.5])

                    # add labels
                    new_line = '\n'
                    axes[row_id][col_id].text(0.6, 0.2, f'{len(sess_data.trials)} trials {new_line}'
                                                        f'{len(sess_data.spikes)} units',
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
        fig.suptitle(f'Confusion matrices - all sessions - {sess_data.results_tags}', fontsize=14)
        plt.tight_layout()
        kwargs = sess_data.results_io.get_figure_args(f'group_confusion_matrices', additional_tags=f'plot{plot_num}',
                                                      format='pdf')
        plt.savefig(**kwargs)
        plt.close()


