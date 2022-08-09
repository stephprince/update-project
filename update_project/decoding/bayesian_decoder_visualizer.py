import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from pathlib import Path
from matplotlib import ticker

from update_project.camera_sync.cam_plot_utils import write_camera_video
from update_project.decoding.interpolate import griddata_time_intervals
from update_project.decoding.bayesian_decoder_aggregator import BayesianDecoderAggregator
from update_project.results_io import ResultsIO
from update_project.general.plots import plot_distributions, get_limits_from_data, get_color_theme

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
        error_bars = ''
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
                [(v['data']['stats']['prob_sum'][f'{error_bars}lower'], v['data']['stats']['prob_sum'][f'{error_bars}upper'])
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
                                         #vmin=0.25 * np.nanmin(prob_map), vmax=0.75 * np.nanmax(prob_map),
                                                   vmin=0.015, vmax=0.07,  # other options are (0.01, 0.25 to 0.1()
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
                axes[row_id[3]][col_id[3]].fill_between(times, stats['prob_sum'][f'{error_bars}lower'], stats['prob_sum'][f'{error_bars}upper'],
                                             alpha=0.2, color=value['color'], label='95% CI')
                axes[row_id[3]][col_id[3]].plot([0, 0], err_limits, linestyle='dashed', color='k', alpha=0.25)
                axes[row_id[3]][col_id[3]].set(xlim=[times[0], times[-1]], ylim=err_limits, ylabel=key)
                axes[row_id[3]][col_id[3]].legend(loc='upper left')
                axes[row_id[3]][col_id[3]].set_title(f'{title} trials - {label} - prob_sum', fontsize=14)

                # heat maps by trial
                im = axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].imshow(value['data']['prob_sum'], cmap=value['cmap'], origin='lower',
                                             aspect='auto', extent=[times[0], times[-1], 0, len(value['data']['prob_sum'])],
                                             vmin=0 * all_limits_balanced[0], vmax=0.4 * all_limits_balanced[1])
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].invert_yaxis()
                axes[row_id[4 + bound_ind]][col_id[4 + bound_ind]].plot([0, 0], [0, len(value['data']['prob_sum'])], linestyle='dashed', color=[0, 0, 0, 0.5])
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

            im = axes[ax_dict[1]].imshow(prob_map[:, :, time_bin], cmap=self.colors['cmap'], origin='lower', aspect='auto',
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

        im = axes['A'].imshow(self.prob_density_grid, aspect='auto', origin='lower', cmap=self.colors['cmap'], vmin=0, vmax=0.75,
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
        axes['D'] = sns.heatmap(self.confusion_matrix, cmap=self.colors['cmap'], ax=axes['D'], square=True,
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
    def __init__(self, data, exclusion_criteria=None, params=None, threshold_params=None, overwrite=False):
        super().__init__(data, exclusion_criteria=exclusion_criteria, params=params, threshold_params=threshold_params)

        self.aggregator = BayesianDecoderAggregator(exclusion_criteria=exclusion_criteria)
        self.aggregator.run_df_aggregation(data, overwrite=True)
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
            group_data = self.aggregator.group_df.groupby(group_names)
            for name, data in group_data:
                print(f'Plotting data for group {name}...')
                # plot correct vs. incorrect trial types
                self.plot_theta_phase_histogram(data, name)
                self.plot_group_aligned_data(data, name)
                #self.plot_group_aligned_comparisons(data, name)

                # plot model metrics
                self.plot_tuning_curves(data, name)
                self.plot_group_confusion_matrices(data, name)
                self.plot_parameter_comparison(data, name, thresh_params=True)

                for iter_list in itertools.product(*self.threshold_params.values()):
                    thresh_mask = pd.concat([data[k] >= v for k, v in zip(self.threshold_params.keys(), iter_list)],
                                            axis=1).all(axis=1)
                    subset_data = data[thresh_mask]
                    tags = '_'.join([f'{k}_{v}' for k, v in zip(self.threshold_params.keys(), iter_list)])
                    self.plot_all_confusion_matrices(subset_data, name, tags=tags)
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
                sess_key = row['session_id']
                vmax = 4  # default to 5 vmax probability/chance
                if row['feature'] in ['choice', 'turn_type']:
                    vmax = 2

                if (counter % (ncols * nrows) == 0) and (counter != 0):
                    fig.suptitle(f'Confusion matrices - all sessions - {row["results_tags"]}')
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
                    if hasattr(row['bins'], 'astype'):  # if the matrix exists
                        matrix = np.vstack(sess_matrix) * row['encoder_bin_num']  # scale to be probability/chance
                        locations = row['virtual_track'].get_cue_locations().get(row['feature'],
                                                                                    dict())  # don't annotate graph if no locations indicated
                        limits = [np.min(row['bins'].astype(int)), np.max(row['bins'].astype(int))]
                        im = axes[row_id][col_id].imshow(matrix, cmap=self.colors['cmap'], origin='lower',  # aspect='auto',
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

    def plot_group_aligned_data(self, data, name, plot_groups=None):

        plot_groups = plot_groups or dict(update_type=[['non_update'], ['switch'], ['stay']],
                                          turn_type=[[1], [2], [1, 2]], correct=[[0], [1], [0, 1]])
        feat = data['feature'].values[0]
        param_group_data = data.groupby(self.params)  # main group is what gets the different plots
        for param_name, param_data in param_group_data:
            window = 2
            align_times = self.aggregator.align_times

            for plot_types in list(itertools.product(*plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
                title = '_'.join([''.join([k, str(v)]) for k, v in zip(plot_groups.keys(), plot_types)])

                # make plots for aligned data (1 row for each plot, 1 col for each align time)
                ncols, nrows = (len(align_times[:-1]), 6)
                fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row')
                for ind, time_label in enumerate(align_times[:-1]):
                    filter_dict = dict(time_label=[time_label], **plot_group_dict)
                    p_data = param_data.copy(deep=True)
                    group_aligned_data = self.aggregator.select_group_aligned_data(p_data, filter_dict, window)
                    if np.size(group_aligned_data) and group_aligned_data is not None:
                        quant_aligned_data = self.aggregator.quantify_aligned_data(p_data, group_aligned_data)
                        bounds = [v['bound_values'] for v in quant_aligned_data.values()]
                        self.plot_1d_around_update(group_aligned_data, quant_aligned_data, time_label, feat, axes,
                                               row_id=np.arange(nrows), col_id=[ind] * nrows,
                                               feature_name=feat, prob_map_axis=0, bounds=bounds)

                # save figure
                fig.suptitle(title)
                tags = f'{"_".join(["".join(n) for n in name])}_{title}_win{window}_' \
                       f'{"_".join([f"{p}_{n}" for p, n in zip(self.params, param_name)])}'
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_aligned_data', additional_tags=tags)

    def plot_group_aligned_comparisons(self, data, name, plot_groups=None):
        plot_groups = plot_groups or dict(update_type=[['non_update'], ['switch'], ['stay']],
                                          turn_type=[[1, 2]], correct=[[0], [1]])
        feat = data['feature'].values[0]
        window = 2
        for param_name, param_data in data.groupby(self.params):
            compiled_data = []
            for plot_types in list(itertools.product(*plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}
                for ind, time_label in enumerate(self.aggregator.align_times[:-1]):
                    filter_dict = dict(time_label=[time_label], **plot_group_dict)
                    p_data = param_data.copy(deep=True)
                    group_aligned_data = self.aggregator.select_group_aligned_data(p_data, filter_dict, window)
                    if np.size(group_aligned_data) and group_aligned_data is not None:
                        quant_aligned_data = self.aggregator.quantify_aligned_data(p_data, group_aligned_data)
                        compiled_data.append(dict(data=group_aligned_data, quant=quant_aligned_data,
                                                  **{k: v[0] for k, v in filter_dict.items()}))

            # compile data for comparisons
            compiled_data_df = pd.DataFrame(compiled_data)
            filter_dict = dict(time_label='t_update', correct=1)
            update_mask = pd.concat([compiled_data_df[k] == v for k, v in filter_dict.items()], axis=1).all(axis=1)
            filter_dict = dict(time_label='t_update', update_type='switch')
            correct_mask = pd.concat([compiled_data_df[k] == v for k, v in filter_dict.items()], axis=1).all(axis=1)
            compare_df = dict(update_type=dict(data=compiled_data_df[update_mask],
                                               comparison=['switch', 'stay']),
                              correct=dict(data=compiled_data_df[correct_mask],
                                           comparison=[0, 1]))

            # plot the data and accompanying stats
            stats_data_all = []
            for comp, data_dict in compare_df.items():
                ncols, nrows = (3, 6)  # one col for each comparison, 1 col for difference between two
                fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row')
                tags = f'{"_".join(["".join(n) for n in name])}' \
                       f'{"_".join([f"{p}_{n}" for p, n in zip(self.params, param_name)])}'
                stats_data, stats_plot_data = self.aggregator.get_aligned_stats(comp, data_dict, quant_aligned_data, tags)
                stats_data_all.append(stats_data)
                self.plot_aligned_comparison(comp, data_dict, stats_plot_data, nrows, feat, filter_dict)
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'compare_{comp}_aligned_data',
                                         additional_tags=tags)  # TODO - is feature in this name/tags

            self.plot_group_aligned_stats(stats_data_all, tags=tags)

    def plot_aligned_comparison(self, comp, data_dict, stats_data, axes, nrows, feat, filter_dict):
        data_to_compare = []
        quant_to_compare = []
        for ind, v in enumerate(data_dict['comparison']):
            title = f'{v}_{"".join([f"{k}{v}" for k, v in filter_dict.items()])}'
            data = data_dict['data'][data_dict['data'][comp] == v]['data'].values[0]
            quant = data_dict['data'][data_dict['data'][comp] == v]['quant'].values[0]
            bounds = [v['bound_values'] for v in quant.values()]
            self.plot_1d_around_update(data, quant, title, feat, axes, row_id=np.arange(nrows), col_id=[ind] * nrows,
                                       feature_name=feat, prob_map_axis=0, bounds=bounds)
            data_to_compare.append(data)
            quant_to_compare.append(quant)

        # get color mappings
        data_split = dict(initial_stay=dict(data=quant_to_compare['left'],
                                            color=self.colors['stay'],
                                            cmap=self.colors['stay_cmap']),
                          switch=dict(data=quantification_data['right'],
                                      color=self.colors['switch'],
                                      cmap=self.colors['switch_cmap']))

        prob_map = np.nanmean(data_around_update['probability'], axis=prob_map_axis)
        if prob_map_axis == 1:
            prob_map = prob_map.T

        stats = data_around_update['stats']
        limits = get_limits_from_data([data_around_update['feature']], balanced=balanced)
        err_limits = get_limits_from_data(
            [(v['data']['stats']['prob_sum'][f'{error_bars}lower'],
              v['data']['stats']['prob_sum'][f'{error_bars}upper'])
             for k, v in data_split.items()], balanced=False)
        err_limits[0] = 0
        all_limits_balanced = get_limits_from_data([v['data']['prob_sum'] for v in data_split.values()])
        times = data_around_update['times']
        time_tick_values = times.astype(int)
        n_position_bins = np.shape(prob_map)[0]
        data_values = np.linspace(np.nanmin(data_around_update['decoding']),
                                  np.nanmax(data_around_update['decoding']),
                                  n_position_bins)

        pos_values_after_update = np.sum(
            data_around_update['feature'][int(len(time_tick_values) / 2):int(len(time_tick_values) / 2) + 10],
            axis=0)
        sort_index = np.argsort(pos_values_after_update)

        # add significance stars
        bound_plot_info = dict(initial=dict(height=10, color=self.colors['stay']),
                               new=dict(height=5, color=self.colors['switch']))
        for key, val in bound_plot_info.items():
            stars_height = [axes[3][0].get_ylim()[1] / (val['height'])] * len(stats_data['sig'][key])
            blanks_height = [axes[3][0].get_ylim()[1] / (val['height'])] * len(stats_data['ns'][key])
            axes[3][2].plot(stats_data['ns'][key], blanks_height, marker="o", linestyle='', markerfacecolor='k',
                            markersize=5, label='n.s.')
            axes[3][2].plot(stats_data['sig'][key], stars_height, marker="*", linestyle='',
                            markerfacecolor=val['color'],
                            markersize=10, label=f'{key} sig.')

    def plot_group_aligned_stats(self, data_for_stats, tags=''):
        # grab data from the first second after the update occurs  TODO - make this more generalized
        prob_sum_df = pd.DataFrame(data_for_stats)
        prob_sum_df = prob_sum_df.explode('prob_sum').reset_index(drop=True)
        bins_to_grab = np.floor([len(prob_sum_df['prob_sum'].values[0])/2,
                                 5 * len(prob_sum_df['prob_sum'].values[0])/8]).astype(int)
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
        feat = data['feature'].values[0]
        locations = data.virtual_track.values[0].get_cue_locations().get(feat, dict())
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
                x_limits = [np.min(tuning_curve_bins.astype(int)), np.max(tuning_curve_bins.astype(int))]
                im = axes[row_id][col_id].imshow(tuning_curve_scaled[sort_index, :], cmap=self.colors['plain_cmap'], origin='lower',
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

    def plot_theta_phase_histogram(self, data, name, plot_groups=None):
        plot_groups = plot_groups or dict(update_type=[['non_update'], ['switch'], ['stay']],
                                          turn_type=[[1], [2], [1, 2]], correct=[[0], [1]])
        feat = data['feature'].values[0]
        align_times = self.aggregator.align_times

        param_group_data = data.groupby(self.params)  # main group is what gets the different plots
        for param_name, param_data in param_group_data:
            for plot_types in list(itertools.product(*plot_groups.values())):
                plot_group_dict = {k: v for k, v in zip(plot_groups.keys(), plot_types)}

                # make plots for aligned data (1 row for hists, half-cycle data, 1 col for each align time)
                ncols, nrows = (len(align_times[:-1]), 6)
                fig, axes = plt.subplots(figsize=(22, 17), nrows=nrows, ncols=ncols, squeeze=False, sharey='row', sharex='col')
                for ind, time_label in enumerate(align_times[:-1]):
                    filter_dict = dict(time_label=[time_label], **plot_group_dict)
                    p_data = param_data.copy(deep=True)
                    theta_phase_data = self.aggregator.calc_theta_phase_data(p_data, filter_dict)
                    for t in theta_phase_data:
                        rows = dict(full=0, half=2, theta_amplitude=0, initial_stay=1, switch=1, home=2)
                        row_ind = rows[t['bin_name']]  # full or half
                        for loc in ['switch', 'initial_stay', 'home', 'theta_amplitude']:
                            if t['times'] == 'post' and time_label == 'choice_made':
                                continue  # don't plot this bc should be minimal values here
                            lstyle = ['dashed' if t['times'] == 'pre' else 'solid'][0]
                            color = [self.colors[loc] if loc in ['switch', 'initial_stay'] else 'k'][0]
                            axes[rows[loc]+row_ind][ind].plot(t['df']['phase_mid']/np.pi, t['df'][f'{loc}'], color=color, linestyle=lstyle,
                                                      label=f'{loc}_{t["times"]}')
                            axes[rows[loc]+row_ind][ind].fill_between(t['df']['phase_mid']/np.pi, t['df'][f'{loc}_err_lower'],
                                                              t['df'][f'{loc}_err_upper'], alpha=0.2, color=color,)
                            axes[rows[loc]+row_ind][ind].xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
                            axes[rows[loc]+row_ind][ind].xaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
                            axes[rows[loc] + row_ind][ind].relim()
                            if ind == 0 and rows[loc]+row_ind != 0:
                                axes[rows[loc]+row_ind][ind].legend()
                                axes[rows[loc]+row_ind][ind].set_ylabel('prob_density')
                            elif ind == 0 and loc == 'theta_amplitude':
                                axes[rows[loc]+row_ind][ind].legend()
                                axes[rows[loc] + row_ind][ind].set_ylabel('theta_amplitude')

                        if rows[loc]+row_ind == nrows-1:
                            axes[rows[loc]+row_ind][ind].set(xlabel='theta phase')
                        if rows[loc]+row_ind == 0:
                            axes[rows[loc] + row_ind][ind].set_title(time_label)

                # save figure
                title = '_'.join([''.join([k, str(v)]) for k, v in zip(plot_groups.keys(), plot_types)])
                fig.suptitle(f'{feat}_{title}')
                tags = f'{"_".join(["".join(n) for n in name])}_{title}' \
                       f'{"_".join([f"{p}_{n}" for p, n in zip(self.params, param_name)])}'
                self.results_io.save_fig(fig=fig, axes=axes, filename=f'theta_phase_hist', additional_tags=tags)