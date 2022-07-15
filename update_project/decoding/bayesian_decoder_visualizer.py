import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from collections import defaultdict
from pathlib import Path

from update_project.camera_sync.cam_plot_utils import write_camera_video
from update_project.decoding.interpolate import griddata_time_intervals
from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_limits_from_data, get_color_theme
from update_project.statistics import get_fig_stats

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')


class BayesianDecoderVisualizer:

    def __init__(self, data, exclusion_criteria=None, params=None, threshold_params=None):

        self.data = data
        self.data_exists = True
        self.exclusion_criteria = exclusion_criteria
        self.params = params
        self.threshold_params = threshold_params or dict(num_units=[self.exclusion_criteria['units']],
                                                         num_trials=[self.exclusion_criteria['trials']])
        self.colors = get_color_theme()

    def _get_confusion_matrix(self, data, bins):
        if len(bins):
            df_bins = pd.cut(data['actual_feature'], bins, include_lowest=True)
            decoding_matrix = data['prob_dist'].groupby(df_bins).apply(
                lambda x: self._get_mean_prob_dist(x, bins)).values

            confusion_matrix = np.vstack(decoding_matrix).T  # transpose so that true position is on the x-axis
        else:
            confusion_matrix = []

        return confusion_matrix

    @staticmethod
    def _get_confusion_matrix_sum(confusion_matrix):
        num_bins_to_sum = int(
            (len(confusion_matrix) / 10 - 1) / 2)  # one tenth of the track minus the identity line bin
        values_to_sum = []
        for i in range(len(confusion_matrix)):
            values_to_sum.append(confusion_matrix[i, i])  # identity line values
            if i < len(confusion_matrix) - num_bins_to_sum:
                values_to_sum.append(confusion_matrix[i + num_bins_to_sum, i])  # bins above
                values_to_sum.append(confusion_matrix[i, i + num_bins_to_sum])  # bins to right
        confusion_matrix_sum = np.nansum(values_to_sum)

        return confusion_matrix_sum

    @staticmethod
    def _get_mean_prob_dist(probabilities, bins):
        if probabilities.empty:
            mean_prob_dist = np.empty(np.shape(bins[1:]))  # remove one bin to match other probabilities shapes
            mean_prob_dist[:] = np.nan
        else:
            mean_prob_dist = np.nanmean(np.vstack(probabilities.values), axis=0)

        return mean_prob_dist

    @staticmethod
    def _check_data_exists(data):
        if isinstance(data, np.ndarray):
            data_exists = data.any()
        else:
            data_exists = bool(data)

        return data_exists

    def plot_1d_around_update(self, data_around_update, quantification_data, title, label, axes,
                              row_id, col_id, feature_name=None, prob_map_axis=0, bounds=[]):
        # get color mappings
        data_split = dict(initial_stay=dict(data=quantification_data['left'],
                                       color=self.colors['stay'],
                                       cmap=self.colors['stay_cmap']),
                          switch=dict(data=quantification_data['right'],
                                        color=self.colors['switch'],
                                        cmap=self.colors['switch_cmap']))

        feature_name = feature_name or self.data.feature_names[0]
        if feature_name in ['x_position', 'view_angle', 'choice', 'turn_type']:  # divergent color maps for div data
            cmap_pos = self.colors['left_right_cmap_div']
            scaling_value = 0.5
            balanced = True
        elif feature_name in ['y_position']:  # sequential color map for seq data
            cmap_pos = 'Greys'
            scaling_value = 1
            balanced = False

        if self._check_data_exists(data_around_update['probability']):  # skip if there is no data
            prob_map = np.nanmean(data_around_update['probability'], axis=prob_map_axis)
            if prob_map_axis == 1:
                prob_map = prob_map.T

            stats = data_around_update['stats']
            limits = get_limits_from_data([data_around_update['feature']], balanced=balanced)
            err_limits = get_limits_from_data(
                [(v['data']['stats']['prob_sum']['err_lower'], v['data']['stats']['prob_sum']['err_upper'])
                 for k, v in data_split.items()], balanced=False)
            err_limits[0] = 0
            all_limits_balanced = get_limits_from_data([v['data']['prob_sum'] for v in data_split.values()])
            times = data_around_update['times']
            time_tick_values = times.astype(int)
            n_position_bins = np.shape(prob_map)[0]
            data_values = np.linspace(np.nanmin(data_around_update['decoding']), np.nanmax(data_around_update['decoding']),
                                      n_position_bins)

            pos_values_after_update = np.sum(
                data_around_update['feature'][int(len(time_tick_values) / 2):int(len(time_tick_values) / 2) + 10],
                axis=0)
            sort_index = np.argsort(pos_values_after_update)

            im = axes[row_id[0]][col_id[0]].imshow(prob_map, cmap=self.colors['cmap'], origin='lower', aspect='auto',
                                         vmin=0.25 * np.nanmin(prob_map), vmax=0.75 * np.nanmax(prob_map),
                                         extent=[times[0], times[-1], data_values[0], data_values[-1]])
            axes[row_id[0]][col_id[0]].plot([0, 0], [data_values[0], data_values[-1]], linestyle='dashed', color=[0, 0, 0, 0.5])
            axes[row_id[0]][col_id[0]].invert_yaxis()
            axes[row_id[0]][col_id[0]].set(xlabel='Time around update (s)', ylabel=f'{label}', xlim=[times[0], times[-1]],
                                 ylim=[data_values[0], data_values[-1]])
            axes[row_id[0]][col_id[0]].set_title(f'{title} trials - probability density - {label}', fontsize=14)
            plt.colorbar(im, ax=axes[row_id[0]][col_id[0]], label='probability density', pad=0.01, location='right',
                         fraction=0.046)
            for b in bounds:
                axes[row_id[0]][col_id[0]].plot([times[0], times[-1]], [b[0], b[0]], linestyle='dashed', color=[0, 0, 0, 0.5],
                                                linewidth=0.5)
                axes[row_id[0]][col_id[0]].plot([times[0], times[-1]], [b[1], b[1]], linestyle='dashed', color=[0, 0, 0, 0.5],
                                                linewidth=0.5)

            axes[row_id[1]][col_id[1]].plot(times, stats['error']['mean'], color=self.colors['error'], label='|True - decoded|')
            axes[row_id[1]][col_id[1]].fill_between(times, stats['error']['lower'], stats['error']['upper'], alpha=0.2, color=self.colors['error'],
                                          label='95% CI')
            axes[row_id[1]][col_id[1]].plot([0, 0], [0, np.max(stats['error']['upper'])], linestyle='dashed', color='k',
                                  alpha=0.25)
            axes[row_id[1]][col_id[1]].set(xlim=[times[0], times[-1]], ylim=[0, np.nanmax(stats['error']['upper'])],
                                 xlabel='Time around update(s)', ylabel=label)
            axes[row_id[1]][col_id[1]].set_title(f'{title} trials - decoding error {label}', fontsize=14)
            axes[row_id[1]][col_id[1]].legend(loc='upper left')

            im = axes[row_id[2]][col_id[2]].imshow(data_around_update['feature'][:, sort_index], cmap=cmap_pos, origin='lower',
                                         vmin=scaling_value * limits[0], vmax=scaling_value * limits[1],
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
                axes[row_id[3]][col_id[3]].plot(times, stats['prob_sum']['mean'], color=value['color'], label=key)
                axes[row_id[3]][col_id[3]].fill_between(times, stats['prob_sum']['err_lower'], stats['prob_sum']['err_upper'],
                                             alpha=0.2, color=value['color'], label='95% CI')
                axes[row_id[3]][col_id[3]].plot([0, 0], err_limits, linestyle='dashed', color='k', alpha=0.25)
                axes[row_id[3]][col_id[3]].set(xlim=[times[0], times[-1]], ylim=err_limits, ylabel=key)
                axes[row_id[3]][col_id[3]].legend(loc='upper left')
                axes[row_id[3]][col_id[3]].set_title(f'{title} trials - {label} - prob_sum', fontsize=14)

                # heat maps by trial
                update_time_values = [[len(time_tick_values) / 2, len(time_tick_values) / 2],
                                      [0, np.shape(value['data']['prob_sum'])[1]]]
                im = axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].imshow(value['data']['prob_sum'], cmap=value['cmap'], origin='lower',
                                             aspect='auto', extent=[times[0], times[-1], 0, len(value['data']['prob_sum'].T)],
                                             vmin=0 * all_limits_balanced[0], vmax=0.4 * all_limits_balanced[1])
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].invert_yaxis()
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].plot(update_time_values[0], update_time_values[1], linestyle='dashed', color=[0, 0, 0, 0.5])
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].set(xlim=[times[0], times[-1]], ylabel=f'trials')
                plt.colorbar(im, ax=axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]], label='probability', pad=0.01, location='right',
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


class SessionVisualizer(BayesianDecoderVisualizer):
    def __init__(self, data, exclusion_criteria=None, params=None):
        super().__init__(data, exclusion_criteria=exclusion_criteria, params=params)

        if not self.data.summary_df.empty:
            times, locs = self._get_example_period()
            self.start_time, self.end_time = times
            self.start_loc, self.end_loc = locs
            self.prob_density_grid = self._get_prob_density_grid()  # prob density plot for example period

            # calculate confusion matrix for session
            self.confusion_matrix = self._get_confusion_matrix(data.summary_df, data)
        else:
            self.data_exists = False
            warnings.warn(f'No summary dataframe found for {self.data.results_io.session_id}, skipping...')

    def plot(self):
        if self.data_exists:
            print(f'Plotting data for session {self.data.results_io.session_id}...')
            self.plot_session_summary()
            self.plot_aligned_data()
        else:
            print(f'No data found to plot')

    def _get_example_period(self, window=500):
        time_window = self.data.decoder_bin_size * window
        times = [self.data.summary_df['decoding_error_rolling'].idxmin()]  # get times/locs with minimum error
        locs = [self.data.summary_df.index.searchsorted(times[0])]
        if (times[0] + time_window) < self.data.summary_df.index.max():
            locs.append(self.data.summary_df.index.searchsorted(times[0] + time_window))
        else:
            locs.append(self.data.summary_df.index.searchsorted(times[0] - time_window))
        times.append(self.data.summary_df.iloc[locs[1]].name)
        times.sort()
        locs.sort()

        if np.max(locs) - np.min(locs) <= 1:  # try with a larger window if the bins fall in a weird place
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

    def plot_session_summary(self):
        # plot the decoding data
        mosaic = """
           AAAAAA
           BBBBBB
           CCCCCC
           DDDEEE
           DDDEEE
           """
        axes = plt.figure(figsize=(10, 10)).subplot_mosaic(mosaic)
        label = self.data.feature_names[0]
        range = self.data.bins[-1] - self.data.bins[0]

        im = axes['A'].imshow(self.prob_density_grid, aspect='auto', origin='lower', cmap='YlGnBu', vmin=0, vmax=0.75,
                              extent=[self.start_time, self.end_time, self.data.bins[0], self.data.bins[-1]], )
        axes['A'].plot(self.data.features_test.loc[self.start_time:self.end_time], label='True', color=[0, 0, 0, 0.5],
                       linestyle='dashed')
        axes['A'].set(xlim=[self.start_time, self.end_time], ylim=[self.data.bins[0], self.data.bins[-1]],
                      xlabel='Time (s)', ylabel=label)
        axes['A'].set_title(f'Bayesian decoding - {self.data.results_io.session_id} - example period', fontsize=14)
        axes['A'].legend(loc='upper right')

        axes['B'].plot(self.data.features_test.loc[self.start_time:self.end_time], color=[0, 0, 0, 0.5], label='True')
        axes['B'].plot(self.data.decoded_values.loc[self.start_time:self.end_time], color='b', label='Decoded')
        axes['B'].set(xlim=[self.start_time, self.end_time], ylim=[self.data.bins[0], self.data.bins[-1]],
                      xlabel='Time (s)', ylabel=label)
        axes['B'].legend(loc='upper right')

        axes['C'].plot(self.data.summary_df['decoding_error'].loc[self.start_time:self.end_time], color=[0, 0, 0],
                       label='True')
        axes['C'].set(xlim=[self.start_time, self.end_time], ylim=[0, range], xlabel='Time (s)',
                      ylabel='Decoding error')

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
            axes = plt.figure().subplot_mosaic(mosaic)
            feature_name = self.data.feature_names[0]
            self.plot_1d_around_update(self.data.aligned_data['switch'][feature_name], nbins, window, 'switch',
                                       feature_name, 'b', axes, ['A', 'E', 'I', 'M', 'Q'])
            self.plot_1d_around_update(self.data.aligned_data['stay'][feature_name], nbins, window, 'stay',
                                       feature_name, 'm', axes, ['C', 'G', 'K', 'O', 'S'])
            plt.suptitle(f'{self.data.results_io.session_id} decoding around update trials')
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
                axes = plt.figure().subplot_mosaic(mosaic)
                self.plot_2d_around_update(self.data.aligned_data['switch'], time_bin, times, 'switch', 'b', axes,
                                           ['A', 'B'])
                self.plot_2d_around_update(self.data.aligned_data['stay'], time_bin, times, 'stay', 'm', axes,
                                           ['C', 'D'])
                plt.suptitle(f'{self.data.results_io.session_id} decoding around update trials')
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


class GroupVisualizer(BayesianDecoderVisualizer):
    def __init__(self, data, exclusion_criteria=None, params=None, threshold_params=None):
        super().__init__(data, exclusion_criteria=exclusion_criteria, params=params, threshold_params=threshold_params)

        # get session visualization info
        for sess_dict in self.data:  # this could probably be done with dataframe group not with loop
            # get decoding matrices
            if sess_dict['decoder'].convert_to_binary:
                bins = [-1, 0, 1]
            else:
                bins = sess_dict['decoder'].bins
            sess_dict.update(confusion_matrix=self._get_confusion_matrix(sess_dict['decoder'].summary_df, bins))

            # get session error data
            session_error = self._get_session_error(sess_dict['decoder'])
            sess_dict.update(confusion_matrix_sum=session_error['confusion_matrix_sum'])
            sess_dict.update(rmse=session_error['rmse'])
            sess_dict.update(raw_error=session_error['raw_error_median'])

            # loop through different exclusion criteria
            sess_dict.update(num_units=len(sess_dict['decoder'].spikes))
            sess_dict.update(num_trials=len(sess_dict['decoder'].train_df))
            sess_dict.update(excluded_session=self._meets_exclusion_criteria(sess_dict['decoder']))

        group_df = pd.DataFrame(self.data)
        self.group_df = group_df[~group_df['excluded_session']]  # only keep non-excluded sessions
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def plot(self, group_by=None):
        if self.data_exists:

            # make plots inspecting errors across all groups (have to change units/trials loops location if I want this)
            self.groups = group_by
            group_names = list(group_by.keys())
            self.plot_all_groups_error(main_group=group_names[0], sub_group=group_names[1])
            self.plot_all_groups_error(main_group=group_names, sub_group='animal')
            self.plot_all_groups_error(main_group='feature', sub_group='num_units', thresh_params=True)
            self.plot_all_groups_error(main_group='feature', sub_group='num_trials', thresh_params=True)

            # make plots for each individual subgroup
            group_data = self.group_df.groupby(group_names)
            for name, data in group_data:
                print(f'Plotting data for group {name}...')
                # plot model metrics
                self.plot_tuning_curves(data, name)
                self.plot_group_confusion_matrices(data, name)
                self.plot_parameter_comparison(data, name, thresh_params=True)

                # plot correct vs. incorrect trial types
                self.plot_group_aligned_data(data, name)

                for iter_list in itertools.product(*self.threshold_params.values()):
                    thresh_mask = pd.concat([data[k] >= v for k, v in zip(self.threshold_params.keys(), iter_list)],
                                            axis=1).all(axis=1)
                    subset_data = data[thresh_mask]
                    tags = '_'.join([f'{k}_{v}' for k, v in zip(self.threshold_params.keys(), iter_list)])
                    self.plot_all_confusion_matrices(subset_data, name, tags=tags)
        else:
            print(f'No data found to plot')

    @staticmethod
    def _get_group_summary_df(data):
        # get giant dataframe of all decoding data
        summary_df_list = []
        for _, sess_data in data.iterrows():
            summary_df_list.append(sess_data['decoder'].summary_df)
        group_summary_df = pd.concat(summary_df_list, axis=0, ignore_index=True)

        return group_summary_df

    @staticmethod
    def _get_group_aligned_data(param_data):
        # compile aligned data
        num_trials = 0
        aligned_data_list = []
        for _, sess_data in param_data.iterrows():
            for item in sess_data['decoder'].aligned_data:
                aligned_data_list.append(dict(session_id=sess_data['session_id'],
                                              animal=sess_data['animal'],
                                              region=sess_data['region'],
                                              **item))
                num_trials = num_trials + len(item['turn_type'])
        aligned_data_df = pd.DataFrame(aligned_data_list)
        aligned_data_df.drop(['stats'], axis='columns', inplace=True)  # get rid of stats dict bc for session
        data_to_explode = ['feature', 'decoding', 'error', 'probability', 'turn_type', 'correct']
        other_cols = [col for col in aligned_data_df.columns.values if col not in data_to_explode]
        exploded_df = pd.concat([aligned_data_df.explode(d).reset_index()[d] if d != 'feature'
                                 else aligned_data_df[[*other_cols, 'feature']].explode(d).reset_index(drop=True)
                                 for d in data_to_explode], axis=1)
        exploded_df.dropna(axis='rows', inplace=True)
        assert len(exploded_df) == num_trials, 'Number of expected trials does not match dataframe'

        return exploded_df

    @staticmethod
    def _get_group_tuning_curves(param_data):
        tuning_curve_list = []
        for _, sess_data in param_data.iterrows():
            for key, unit_tuning in sess_data['decoder'].model.items():
                tuning_curve_list.append(dict(animal=sess_data['decoder'].results_io.animal,
                                              session=sess_data['decoder'].results_io.session_id,
                                              unit=key,
                                              tuning_curve=unit_tuning,
                                              bins=sess_data['decoder'].bins))

        tuning_curve_df = pd.DataFrame(tuning_curve_list)

        return tuning_curve_df

    def _meets_exclusion_criteria(self, data):
        exclude_session = False  # default to include session

        # apply exclusion criteria
        units_threshold = self.exclusion_criteria.get('units', 0)
        if len(data.spikes) < units_threshold:
            exclude_session = True

        trials_threshold = self.exclusion_criteria.get('trials', 0)
        if len(data.train_df) < trials_threshold:
            exclude_session = True

        return exclude_session

    def _get_session_error(self, data):
        # get error from heatmap
        if data.convert_to_binary:
            bins = [-1, 0, 1]
        else:
            bins = data.bins
        confusion_matrix = self._get_confusion_matrix(data.summary_df, bins)
        confusion_matrix_sum = self._get_confusion_matrix_sum(confusion_matrix)

        # get error from
        rmse = data.summary_df['session_rmse'].mean()
        raw_error_median = data.summary_df['decoding_error'].median()
        session_error = dict(confusion_matrix_sum=confusion_matrix_sum,
                             rmse=rmse,
                             raw_error_median=raw_error_median)

        return session_error

    def _sort_group_confusion_matrices(self, data):
        param_group_data_sorted = []
        for iter_list in itertools.product(*self.threshold_params.values()):
            thresh_mask = pd.concat([data[t] >= i for t, i in zip(self.threshold_params.keys(), iter_list)],
                                    axis=1).all(axis=1)
            subset_data = data[thresh_mask]

            param_group_data = subset_data.groupby(self.params)  # main group is what gets the different plots
            for param_name, param_data in param_group_data:

                # get parameter specific data (confusion matrix, track, bins)
                group_summary_df = self._get_group_summary_df(param_data)
                if param_data['decoder'].values[0].convert_to_binary:
                    bins = [-1, 0, 1]
                else:
                    bins = param_data['decoder'].values[0].bins
                vmax = 2  # default to 5 vmax probability/chance
                if param_data['feature'].values[0] in ['choice', 'turn_type']:
                    vmax = 2
                virtual_track = param_data['decoder'].values[0].virtual_track
                confusion_matrix = self._get_confusion_matrix(group_summary_df, bins)
                confusion_matrix_sum = self._get_confusion_matrix_sum(confusion_matrix)
                locations = virtual_track.get_cue_locations().get(param_data['feature'].values[0], dict())

                # save to output list
                param_group_data_sorted.append(dict(confusion_matrix=confusion_matrix,
                                                    confusion_matrix_sum=confusion_matrix_sum,
                                                    locations=locations,
                                                    bins=bins,
                                                    vmax=vmax,
                                                    param_values=param_name,
                                                    thresh_values=iter_list))

        # sort the list output
        sorted_data = pd.DataFrame(param_group_data_sorted)
        sorted_data.sort_values('confusion_matrix_sum', ascending=False, inplace=True)

        return sorted_data

    @staticmethod
    def _flip_y_position(data, bounds, bins=None):
        flipped_data = []
        for ind, row in data.iteritems():
            if bins is not None:
                areas = dict()
                for bound_name, bound_values in bounds.items():  # loop through left/right bounds
                    prob_map_bins = (bins[1:] + bins[:-1]) / 2
                    areas[f'{bound_name}_start'] = np.searchsorted(prob_map_bins, bound_values[0])
                    areas[f'{bound_name}_stop'] = np.searchsorted(prob_map_bins, bound_values[1])

                left_data = row[areas['left_start']:areas['left_stop'], :].copy()
                right_data = row[areas['right_start']:areas['right_stop'], :].copy()

                row[areas['left_start']:areas['left_stop'], :] = right_data
                row[areas['right_start']:areas['right_stop'], :] = left_data
            else:
                offset = bounds['right'][0] - bounds['left'][0]
                left_data = np.logical_and(row > bounds['left'][0], row < bounds['left'][1])
                right_data = np.logical_and(row > bounds['right'][0], row < bounds['right'][1])
                row[left_data] = row[left_data] + offset
                row[right_data] = row[right_data] - offset
            flipped_data.append(row)

        return flipped_data

    def _select_group_aligned_data(self, param_data, filter_dict, window=None, flip_trials=True):
        group_aligned_df = self._get_group_aligned_data(param_data)

        # filter for specific features
        mask = pd.concat([group_aligned_df[k].isin(v) for k, v in filter_dict.items()], axis=1).all(axis=1)
        data_subset = group_aligned_df[mask]

        # flip trials if indicated
        turn_to_flip = 2
        trials_to_flip = data_subset[data_subset['turn_type'] == turn_to_flip]
        if np.size(trials_to_flip):
            if flip_trials and data_subset['feature_name'].values[0] == 'y_position':
                feat_bins = param_data['decoder'].values[0].bins
                virtual_track = param_data['decoder'].values[0].virtual_track
                bounds = virtual_track.choice_boundaries.get(param_data['feature'].values[0], dict())

                trials_to_flip = trials_to_flip.apply(lambda x: self._flip_y_position(x, bounds) if x.name in ['feature', 'decoding'] else x)
                trials_to_flip = trials_to_flip.apply(lambda x: self._flip_y_position(x, bounds, feat_bins) if x.name in ['probability'] else x)
                data_subset.loc[data_subset['turn_type'] == turn_to_flip, :] = trials_to_flip
            elif flip_trials:
                feat_before_flip = trials_to_flip['feature'].values[0][0]
                prob_before_flip = trials_to_flip['probability'].values[0][0][0]
                trials_to_flip = trials_to_flip.apply(lambda x: x*-1 if x.name in ['feature', 'decoding'] else x)
                trials_to_flip['probability'] = trials_to_flip['probability'].apply(lambda x: np.flipud(x))
                data_subset.loc[data_subset['turn_type'] == turn_to_flip, :] = trials_to_flip
                if ~np.isnan(feat_before_flip):
                    assert feat_before_flip == -trials_to_flip['feature'].values[0][0], 'Data not correctly flipped'
                if ~np.isnan(prob_before_flip):
                    assert prob_before_flip == trials_to_flip['probability'].values[0][-1][0], 'Data not correctly flipped'

        if np.size(data_subset):
            if window:
                nbins = len(data_subset['times'].values[0])
                orig_window_start = data_subset['window_start'].values[0]
                orig_window_stop = data_subset['window_stop'].values[0]
                times = np.linspace(np.max([-window, orig_window_start]), np.min([window, orig_window_stop]), num=nbins)
            else:
                times = data_subset['times'].values[0]

            field_names = ['feature', 'decoding', 'error']
            group_data = {n: np.vstack(data_subset[n]) for n in field_names}
            group_data.update(probability=np.stack(data_subset['probability']))
            group_data.update(times=times)
            group_data.update(stats={k: get_fig_stats(v, axis=0) for k, v in group_data.items()})
        else:
            group_data = None

        return group_data

    @staticmethod
    def _quantify_aligned_data(param_data, aligned_data):
        # get bounds to use to quantify choices
        bins = param_data['decoder'].values[0].bins
        virtual_track = param_data['decoder'].values[0].virtual_track
        bounds = virtual_track.choice_boundaries.get(param_data['feature'].values[0], dict())

        prob_map = aligned_data['probability']
        prob_choice = dict()
        num_bins = dict()
        for bound_name, bound_values in bounds.items():  # loop through left/right bounds
            prob_map_bins = (bins[1:] + bins[:-1]) / 2
            start_bin = np.searchsorted(prob_map_bins, bound_values[0])
            stop_bin = np.searchsorted(prob_map_bins, bound_values[1])

            threshold = 0.1  # total probability density to call a left/right choice
            integrated_prob = np.nansum(prob_map[:, start_bin:stop_bin, :], axis=1)  # (trials, feature bins, window)
            num_bins[bound_name] = len(prob_map[0, start_bin:stop_bin, 0])
            assert len(bins) - 1 == np.shape(prob_map)[1], 'Bound bins not being sorted on right dimension'

            bound_quantification = dict(prob_sum=integrated_prob,  # (trials x window_bins)
                                        thresh_crossing=integrated_prob > threshold)  # (trials x window_bins)
            bound_quantification.update(stats={k: get_fig_stats(v, axis=0) for k, v in bound_quantification.items()})
            bound_quantification.update(bound_values=bound_values,
                                        threshold=threshold)
            prob_choice[bound_name] = bound_quantification  # choice calculating probabilities for

        assert len(np.unique(list(num_bins.values()))) == 1, 'Number of bins for different bounds are not equal'

        return prob_choice

    def plot_all_groups_error(self, main_group, sub_group, thresh_params=False):
        # select no threshold data to plot if thresholds indicated, otherwise combine
        if thresh_params:
            df_list = []
            for thresh_val in self.threshold_params[sub_group]:
                new_df = self.group_df[self.group_df[sub_group] >= thresh_val]
                new_df[f'{sub_group}_thresh'] = thresh_val
                df_list.append(new_df)  # duplicate dfs for each thresh
            df = pd.concat(df_list, axis=0, ignore_index=True)
            sub_group = f'{sub_group}_thresh'  # rename threshold
        else:
            df = self.group_df

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

    def plot_all_confusion_matrices(self, data, name, tags=None):
        param_group_data = data.groupby(self.params)  # main group is what gets the different plots
        for param_name, param_data in param_group_data:
            plot_num = 0
            counter = 0
            ncols, nrows = [6, 3]
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
            for _, row in param_data.iterrows():
                sess_matrix = row['confusion_matrix']
                sess_data = row['decoder']
                sess_key = row['session_id']
                vmax = 4  # default to 5 vmax probability/chance
                if row['feature'] in ['choice', 'turn_type']:
                    vmax = 2

                if (counter % (ncols * nrows) == 0) and (counter != 0):
                    fig.suptitle(f'Confusion matrices - all sessions - {sess_data.results_tags}')
                    param_tags = '_'.join([f'{p}_{n}' for p, n in zip(self.params, param_name)])
                    add_tags = f'{tags}_{"_".join(["".join(n) for n in name])}_{param_tags}_plot{plot_num}'
                    self.results_io.save_fig(fig=fig, axes=axes, filename=f'all_confusion_matrices', additional_tags=add_tags)

                    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
                    counter = 0
                    plot_num += 1
                else:
                    row_id = int(np.floor(counter / ncols))
                    col_id = counter - row_id * ncols

                    # plot confusion matrix
                    if hasattr(sess_data.bins, 'astype'):  # if the matrix exists
                        matrix = np.vstack(sess_matrix) * sess_data.encoder_bin_num  # scale to be probability/chance
                        locations = sess_data.virtual_track.get_cue_locations().get(row['feature'],
                                                                                    dict())  # don't annotate graph if no locations indicated
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
            fig.suptitle(f'Confusion matrices - all sessions - {sess_data.results_tags}')
            param_tags = '_'.join([f'{p}_{n}' for p, n in zip(self.params, param_name)])
            add_tags = f'{tags}_{"_".join(["".join(n) for n in name])}_{param_tags}_plot{plot_num}'
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'all_confusion_matrices', additional_tags=add_tags)

    def plot_group_confusion_matrices(self, data, name):
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
                        limits = [np.min(sorted_data['bins'].astype(int)), np.max(sorted_data['bins'].astype(int))]
                    im = axes[row_id][col_id].imshow(matrix, cmap='YlGnBu', origin='lower',  # aspect='auto',
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
                if col_id == (ncols - 1):
                    plt.colorbar(im, ax=axes[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                                 label='probability / chance')

                counter += 1

        # wrap up last plot after loop finished
        fig.suptitle(f'Confusion matrices - all parameters - {name}')
        tags = f'{"_".join(["".join(n) for n in name])}_plot{plot_num}'
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_confusion_matrices', additional_tags=tags)

    def plot_group_aligned_data(self, data, name, plot_groups=dict(update_type=[['non_update'], ['switch'], ['stay']],
                                                                   turn_type=[[1], [2], [1, 2]],
                                                                   correct=[[0], [1], [0, 1]])):
        feat = data['feature'].values[0]
        param_group_data = data.groupby(self.params)  # main group is what gets the different plots
        for param_name, param_data in param_group_data:
            windows = [2]  #, param_data['decoder'].values[0].aligned_data_window]  # time in seconds
            align_times = param_data['decoder'].values[0].align_times

            for plot_types in list(itertools.product(*plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
                title = '_'.join([''.join([k, str(v)]) for k, v in zip(plot_groups.keys(), plot_types)])

                for window in windows:  # TODO - fix for multiple windows bc right now flips back and forth
                    # make plots (1 row for each plot, 1 col for each align time)
                    ncols, nrows = (len(align_times[:-1]), 6)
                    fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row')
                    for ind, time_label in enumerate(align_times[:-1]):
                        filter_dict = dict(time_label=[time_label], **plot_group_dict)
                        group_aligned_data = self._select_group_aligned_data(param_data, filter_dict, window)
                        if np.size(group_aligned_data) and group_aligned_data is not None:
                            quant_aligned_data = self._quantify_aligned_data(param_data, group_aligned_data)
                            bounds = [v['bound_values'] for v in quant_aligned_data.values()]
                            self.plot_1d_around_update(group_aligned_data, quant_aligned_data, time_label, feat, axes,
                                                   row_id=np.arange(nrows), col_id=[ind] * nrows,
                                                   feature_name=feat, prob_map_axis=0, bounds=bounds)
                    # save figure
                    fig.suptitle(title)
                    tags = f'{"_".join(["".join(n) for n in name])}_{title}_win{window}_' \
                           f'{"_".join([f"{p}_{n}" for p, n in zip(self.params, param_name)])}'
                    self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_aligned_data', additional_tags=tags)

    def plot_tuning_curves(self, data, name):
        feat = data['feature'].values[0]
        locations = data['decoder'].values[0].virtual_track.get_cue_locations().get(feat, dict())
        tuning_curve_params = [p for p in self.params if p not in ['decoder_bins']]
        data_const_decoding = data[data['decoder_bins'] == data['decoder_bins'].values[0]]
        param_group_data = data_const_decoding.groupby(tuning_curve_params)
        plot_num, counter = (0, 0)
        nrows, ncols = (1, 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for param_name, param_data in param_group_data:
            group_tuning_curve_df = self._get_group_tuning_curves(param_data)
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
                x_limits = [np.min(tuning_curve_bins.astype(int)), np.max(tuning_curve_bins.astype(int))]
                im = axes[row_id][col_id].imshow(tuning_curve_scaled[sort_index, :], cmap=self.colors['cmap'], origin='lower',
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
                if row_id == (nrows - 1):
                    axes[row_id][col_id].set_xlabel(f'{feat}')
                if col_id == 0:
                    axes[row_id][col_id].set_ylabel(f'Units')
                if col_id == (ncols - 1) or counter == len(param_name):
                    plt.colorbar(im, ax=axes[row_id][col_id], pad=0.04, location='right', fraction=0.046,
                                 label='Normalized firing rate')
                counter += 1

        # save figure
        fig.suptitle(f'Feature tuning curves - {name}')
        self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves', additional_tags=tags)
