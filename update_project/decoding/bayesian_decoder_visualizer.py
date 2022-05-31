import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pathlib import Path
from scipy.interpolate import griddata
from decoding import plot_decoding_around_update, plot_2d_decoding_around_update
from update_project.camera_sync.cam_plot_utils import write_camera_video
from update_project.utils import create_track_boundaries


class BayesianDecoderVisualizer:
    def __init__(self, dim_num):
        self.dim_num = dim_num

    def plot_decoding_around_update(self):

        if self.dim_num == 1:
            mosaic = """
            ABCD
            EFGH
            IJKL
            MNOP
            QRST
            """
            axes = plt.figure(figsize=(20, 15)).subplot_mosaic(mosaic)
            plot_decoding_around_update(y_around_switch, nbins, window, 'switch', 'y', [0, 258], 'b', axes, ['A', 'E', 'I', 'M', 'Q'])
            plot_decoding_around_update(x_around_switch, nbins, window, 'switch', 'x', [-15, 15], 'purple', axes, ['B', 'F', 'J', 'N', 'R'])
            plot_decoding_around_update(y_around_stay, nbins, window, 'stay', 'y', [0, 258], 'm', axes, ['C', 'G', 'K', 'O', 'S'])
            plot_decoding_around_update(x_around_stay, nbins, window, 'stay', 'x', [-15, 15], 'g', axes, ['D', 'H', 'L', 'P', 'T'])
            plt.suptitle(f'{session_id} decoding around update trials', fontsize=20)
            plt.tight_layout()

            kwargs = UpdateTaskFigureGenerator.get_figure_args(fname='decoding_around_update', session_id=session_id,
                                                               format='pdf')
            plt.savefig(**kwargs)

        elif self.dim_num == 2:
            # plot 2d decoding around update
            times = np.linspace(-window, window, num=nbins)
            Path(figure_path / 'timelapse_figures').mkdir(parents=True, exist_ok=True)
            plot_filenames = []
            for time_bin in range(nbins):
                mosaic = """
                    AB
                    CD
                    """
                axes = plt.figure(figsize=(16, 8)).subplot_mosaic(mosaic)
                plot_2d_decoding_around_update(xy_around_switch, time_bin, times, 'switch', 'b', axes, ['A', 'B'])
                plot_2d_decoding_around_update(xy_around_stay, time_bin, times, 'stay', 'm', axes, ['C', 'D'])
                plt.suptitle(f'{session_id} decoding around update trials', fontsize=20)
                plt.tight_layout()

                filename = 'timelapse_figures'/ f'decoding_around_update_frame_no{time_bin}'
                kwargs = UpdateTaskFigureGenerator.get_figure_args(fname=filename,
                                                                   session_id=session_id,
                                                                   format='pdf')
                plt.savefig(**kwargs)
                plt.close()
                plot_filenames.append(filename)

            # make videos for each plot type
            fps = 5.0
            vid_filename = f'decoding_around_update_video'
            write_camera_video(figure_path, fps, plot_filenames, vid_filename)

        plt.close('all')

    def plot_decoding_around_update(self, data_around_update, nbins, window, title, label, limits, color, axes, ax_dict):
        stats = data_around_update['stats']
        times = np.linspace(-window, window, num=nbins)
        time_tick_values = times.astype(int)
        time_tick_labels = np.array([0, int(len(time_tick_values) / 2), len(time_tick_values) - 1])
        if label == 'x':
            cmap_pos = 'RdGy'
            cmap_decoding = 'PRGn'
            scaling_value = 0.25
        elif label == 'y':
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
                                  [0, np.shape(data_around_update['position'])[1]]]
            v_lims_position = [np.nanmin(data_around_update['position']), np.nanmax(data_around_update['position'])]
            v_lims_decoding = [np.nanmin(data_around_update['decoding']), np.nanmax(data_around_update['decoding'])]
            pos_values_after_update = np.sum(
                data_around_update['position'][time_tick_labels[1]:time_tick_labels[1] + 10],
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
                                 xlabel='Time around update (s)', ylabel=f'{label} position')
            axes[ax_dict[0]].set_title(f'{title} trials - probability density - {label} position', fontsize=14)

            axes[ax_dict[1]].plot(times, stats['position']['mean'], color='k', label='True position')
            axes[ax_dict[1]].fill_between(times, stats['position']['lower'], stats['position']['upper'], alpha=0.2,
                                          color='k', label='95% CI')
            axes[ax_dict[1]].plot(times, stats['decoding']['mean'], color=color, label='Decoded position')
            axes[ax_dict[1]].fill_between(times, stats['decoding']['lower'], stats['decoding']['upper'], alpha=0.2,
                                          color=color, label='95% CI')
            axes[ax_dict[1]].plot([0, 0], limits, linestyle='dashed', color='k', alpha=0.25)
            axes[ax_dict[1]].set(xlim=[-window, window], ylim=limits, xlabel='Time around update(s)',
                                 ylabel=f'{label} position')
            axes[ax_dict[1]].legend(loc='upper left')
            axes[ax_dict[1]].set_title(f'{title} trials - decoded {label} position', fontsize=14)

            axes[ax_dict[2]] = sns.heatmap(data_around_update['position'][:, sort_index].T, cmap=cmap_pos,
                                           ax=axes[ax_dict[2]],
                                           vmin=scaling_value * limits[0], vmax=scaling_value * limits[1],
                                           cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
            axes[ax_dict[2]].plot(update_time_values[0], update_time_values[1], linestyle='dashed',
                                  color=[0, 0, 0, 0.5])
            axes[ax_dict[2]].set(xticks=time_tick_labels, xticklabels=time_tick_values[time_tick_labels],
                                 xlabel='Time around update (s)', ylabel='Trials')
            axes[ax_dict[2]].set_title(f'{title} trials - true {label} position', fontsize=14)

            axes[ax_dict[3]] = sns.heatmap(data_around_update['decoding'][:, sort_index].T, cmap=cmap_decoding,
                                           ax=axes[ax_dict[3]],
                                           vmin=scaling_value * limits[0], vmax=scaling_value * limits[1],
                                           cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
            axes[ax_dict[3]].plot(update_time_values[0], update_time_values[1], linestyle='dashed',
                                  color=[0, 0, 0, 0.5])
            axes[ax_dict[3]].set(xticks=time_tick_labels, xticklabels=time_tick_values[time_tick_labels],
                                 xlabel='Time around update (s)', ylabel='Trials')
            axes[ax_dict[3]].set_title(f'{title} trials - decoded {label} position', fontsize=14)

            axes[ax_dict[4]].plot(times, stats['error']['mean'], color='r', label='|True - decoded|')
            axes[ax_dict[4]].fill_between(times, stats['error']['lower'], stats['error']['upper'], alpha=0.2, color='r',
                                          label='95% CI')
            axes[ax_dict[4]].plot([0, 0], [0, np.max(stats['error']['upper'])], linestyle='dashed', color='k',
                                  alpha=0.25)
            axes[ax_dict[4]].set(xlim=[-window, window], ylim=[0, np.max(stats['error']['upper'])],
                                 xlabel='Time around update(s)', ylabel=label)
            axes[ax_dict[4]].set_title(f'{title} trials - decoding error {label} position', fontsize=14)
            axes[ax_dict[4]].legend(loc='upper left')

    def plot_2d_decoding_around_update(data_around_update, time_bin, times, title, color, axes, ax_dict):
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
            positions_y = stats['position_y']['mean'][:time_bin + 1]
            positions_x = stats['position_x']['mean'][:time_bin + 1]

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


    def plot_decoding_accuracy(self):
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

        plot_decoding_error_summary(all_decoding_data['x'], decoding_matrix_prob['x'], position_bins['x'],
                                    session_rmse['x'],
                                    locations['x'], nb_bins, 'x', axes, ['A', 'B', 'C', 'D', 'E'])
        plot_decoding_error_summary(all_decoding_data['y'], decoding_matrix_prob['y'], position_bins['y'],
                                    session_rmse['y'],
                                    locations['y'], nb_bins, 'y', axes, ['F', 'G', 'H', 'I', 'J'])

        plt.suptitle(f'Group decoding accuracy - non update trials only', fontsize=20)
        plt.tight_layout()


        kwargs = UpdateTaskFigureGenerator.get_figure_args(fname='decoding_error_summary_position',
                                                           session_id=session_id,
                                                           format='pdf')
        plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

    def plot_decoding_summary(self):
        # plot the decoding data
        mosaic = """
           AAAAAA
           BBBBBB
           CCCCCC
           DDEEFF
           DDEEFF
           """
        axes = plt.figure(figsize=(15, 15)).subplot_mosaic(mosaic)

        time_window = 250
        min_error_time = df_decode_results['decoding_error_rolling'].idxmin()
        min_error_index = np.searchsorted(df_decode_results.index, min_error_time)
        if (min_error_index + time_window) < len(df_decode_results):
            start_time = min_error_time
            end_time = df_decode_results.index[min_error_index + time_window]
        else:
            start_time = df_decode_results.index[min_error_index - time_window]
            end_time = min_error_time

        prob = proby_feature.loc[start_time:end_time].stack().reset_index().values
        x1 = np.linspace(min(prob[:, 0]), max(prob[:, 0]), int(end_time - start_time))
        y1 = np.linspace(min(prob[:, 1]), max(prob[:, 1]), len(proby_feature.columns))
        grid_x, grid_y = np.meshgrid(x1, y1)
        grid_prob = griddata(prob[:, 0:2], prob[:, 2], (grid_x, grid_y), method='nearest', fill_value=np.nan)

        im = axes['A'].imshow(grid_prob, aspect='auto', origin='lower', cmap='YlGnBu',  # RdPu
                              extent=[start_time, end_time, position_bins[0], position_bins[-1]],
                              vmin=0, vmax=0.75)
        axes['A'].plot((position_tsg['y'].loc[start_time:end_time]), label='True', color=[0, 0, 0, 0.5],
                       linestyle='dashed')
        axes['A'].set(xlim=[start_time, end_time], ylim=[0, 285],
                      xlabel='Time (s)', ylabel='Y position')
        axes['A'].set_title(f'Bayesian decoding - {session_id} - example period', fontsize=14)
        axes['A'].legend(loc='upper right')

        axes['B'].plot(position_tsg['y'].loc[start_time:end_time], color=[0, 0, 0, 0.5], label='True')
        axes['B'].plot(decoded.loc[start_time:end_time], color='b', label='Decoded')
        axes['B'].set(xlim=[start_time, end_time], ylim=[0, 285],
                      xlabel='Time (s)', ylabel='Y position')
        axes['B'].legend(loc='upper right')

        axes['C'].plot(df_decode_results['decoding_error'].loc[start_time:end_time], color=[0, 0, 0], label='True')
        axes['C'].set(xlim=[start_time, end_time], ylim=[0, 285],
                      xlabel='Time (s)', ylabel='Decoding error')

        tick_values = tuning_curves1d.index.values.astype(int)
        tick_labels = np.array([0, int(len(tick_values) / 2), len(tick_values) - 1])
        axes['D'] = sns.heatmap(np.vstack(decoding_matrix), cmap='YlGnBu', ax=axes['D'], square=True,
                                vmin=0, vmax=0.75 * np.nanmax(np.vstack(decoding_matrix)),
                                cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
        axes['D'].plot([0, 285], [0, 285], linestyle='dashed', color=[0, 0, 0, 0.5])
        axes['D'].invert_yaxis()
        axes['D'].set_title('Decoding accuracy - peak', fontsize=14)
        axes['D'].set(xticks=tick_labels, yticks=tick_labels,
                      xticklabels=tick_values[tick_labels], yticklabels=tick_values[tick_labels],
                      xlabel='Decoded Position', ylabel='Actual Position')

        axes['E'] = sns.heatmap(np.vstack(decoding_matrix_prob), cmap='YlGnBu', ax=axes['E'], square=True,
                                vmin=0, vmax=0.75 * np.nanmax(np.vstack(decoding_matrix_prob)),
                                cbar_kws={'pad': 0.01, 'label': 'mean probability', 'fraction': 0.046})
        axes['E'].plot([0, 285], [0, 285], linestyle='dashed', color=[0, 0, 0, 0.5])
        axes['E'].invert_yaxis()
        axes['E'].set_title('Decoding accuracy - avg prob', fontsize=14)
        axes['E'].set(xticks=tick_labels, yticks=tick_labels,
                      xticklabels=tick_values[tick_labels], yticklabels=tick_values[tick_labels],
                      xlabel='Decoded Position', ylabel='Actual Position')

        axes['F'] = sns.ecdfplot(df_decode_results['decoding_error'], ax=axes['F'], color='k')
        axes['F'].set_title('Decoding accuracy - error')
        axes['F'].set(xlabel='Decoding error', ylabel='Proportion')
        axes['F'].set_aspect(1. / axes['F'].get_data_ratio(), adjustable='box')

        plt.colorbar(im, ax=axes['A'], label='Probability density', pad=0.06, location='bottom', shrink=0.25,
                     anchor=(0.9, 1))
        plt.tight_layout()

        # save figure
        filename = figure_path / f'decoding_summary_git{short_hash}.pdf'
        plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

        plt.close('all')
        io.close()

        def plot_decoding_error_summary(all_decoding_data, decoding_matrix_prob, position_bins, session_rmse, locations,
                                        nb_bins, title, axes, ax_dict):
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