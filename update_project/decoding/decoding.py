import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.interpolate import griddata, interp1d
from scipy.stats import sem

def get_decoding_around_update(position, decoded_data, prob_feature, trials, nbins=50, window=5, flip_trials=False):
    update_times = trials['t_update']
    new_times = np.linspace(-window, window, num=nbins)
    if flip_trials:
        trials_to_flip = trials['turn_type'] == 1  # left trials, flip values so all face the same way
    else:
        trials_to_flip = trials['turn_type'] == 100  # set all to false

    pos_start_locs = position.index.searchsorted(update_times - window - 1)  # a little extra just in case
    pos_stop_locs = position.index.searchsorted(update_times + window + 1)
    pos_interp = interp1d_time_intervals(position, pos_start_locs, pos_stop_locs, new_times, update_times, trials_to_flip)
    pos_out = np.array(pos_interp).T

    decoding_start_locs = decoded_data.index.searchsorted(update_times - window - 1)
    decoding_stop_locs = decoded_data.index.searchsorted(update_times + window + 1)
    decoding_interp = interp1d_time_intervals(decoded_data, decoding_start_locs, decoding_stop_locs, new_times, update_times, trials_to_flip)
    decoding_out = np.array(decoding_interp).T

    prob_out = griddata_time_intervals(prob_feature, decoding_start_locs, decoding_stop_locs, update_times, nbins, trials_to_flip)

    decoding_error = [abs(dec_pos-true_pos) for true_pos, dec_pos in zip(pos_interp, decoding_interp)]
    error_out = np.array(decoding_error).T

    # get means and sem
    stats = {'position': get_stats(pos_interp),
             'decoding': get_stats(decoding_interp),
             'probability': get_stats(prob_out),
             'error': get_stats(decoding_error)}

    return {'position': pos_out,
            'decoding': decoding_out,
            'probability': prob_out,
            'decoding_error': error_out,
            'stats': stats}

def get_2d_decoding_around_update(position, decoded_data, prob_feature, binsxy, trials, nbins=50, window=5):
    update_times = trials['t_update']
    new_times = np.linspace(-window, window, num=nbins)
    trials_to_flip = pd.concat([trials['turn_type'] == 1, trials['turn_type'] == 100], axis=1)
    trials_to_flip.columns = ['x', 'y']

    pos_start_locs = position.index.searchsorted(update_times - window - 1)  # a little extra just in case
    pos_stop_locs = position.index.searchsorted(update_times + window + 1)
    posx_interp = interp1d_time_intervals(position['x'], pos_start_locs, pos_stop_locs, new_times, update_times,
                                         trials_to_flip['x'])
    posy_interp = interp1d_time_intervals(position['y'], pos_start_locs, pos_stop_locs, new_times, update_times,
                                         trials_to_flip['y'])

    decoding_start_locs = decoded_data.index.searchsorted(update_times - window - 1)
    decoding_stop_locs = decoded_data.index.searchsorted(update_times + window + 1)
    decodingx_interp = interp1d_time_intervals(decoded_data['x'], decoding_start_locs, decoding_stop_locs, new_times,
                                              update_times, trials_to_flip['x'])
    decodingy_interp = interp1d_time_intervals(decoded_data['y'], decoding_start_locs, decoding_stop_locs, new_times,
                                              update_times, trials_to_flip['y'])

    prob_out = griddata_2d_time_intervals(prob_feature, binsxy, decoded_data.index.values, decoding_start_locs, decoding_stop_locs, update_times, nbins,
                                       trials_to_flip)

    decodingx_error = [abs(dec_pos - true_pos) for true_pos, dec_pos in zip(posx_interp, decodingx_interp)]
    decodingy_error = [abs(dec_pos - true_pos) for true_pos, dec_pos in zip(posy_interp, decodingy_interp)]

    # get means and sem
    stats = {'position_x': get_stats(posx_interp),
             'position_y': get_stats(posy_interp),
             'decoding_x': get_stats(decodingx_interp),
             'decoding_y': get_stats(decodingy_interp),
             'probability': get_stats(prob_out),
             'error_x': get_stats(decodingx_error),
             'error_y': get_stats(decodingy_error)}

    return {'position_x': posx_interp,
             'position_y': posy_interp,
             'decoding_x': decodingx_interp,
             'decoding_y': decodingy_interp,
             'probability': prob_out,
             'decoding_error_x': decodingx_error,
             'decoding_error_y': decodingy_error,
            'stats': stats}

def plot_decoding_around_update(data_around_update, nbins, window, title, label, limits, color, axes, ax_dict):
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
        data_values = np.linspace(np.min(data_around_update['decoding']), np.max(data_around_update['decoding']), n_position_bins)
        ytick_values = data_values.astype(int)
        ytick_labels = np.array([0, int(len(ytick_values) / 2), len(ytick_values) - 1])
        update_time_values = [[len(time_tick_values) / 2, len(time_tick_values) / 2], [0, np.shape(data_around_update['position'])[1]]]
        v_lims_position = [np.nanmin(data_around_update['position']), np.nanmax(data_around_update['position'])]
        v_lims_decoding = [np.nanmin(data_around_update['decoding']), np.nanmax(data_around_update['decoding'])]
        pos_values_after_update = np.sum(data_around_update['position'][time_tick_labels[1]:time_tick_labels[1]+10],axis=0)
        sort_index = np.argsort(pos_values_after_update)

        axes[ax_dict[0]] = sns.heatmap(prob_map, cmap='YlGnBu', ax=axes[ax_dict[0]],
                                       vmin=0, vmax=0.75 * np.nanmax(prob_map),
                                       cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
        axes[ax_dict[0]].plot(update_time_values[0], update_time_values[1], linestyle='dashed', color=[0, 0, 0, 0.5])
        axes[ax_dict[0]].invert_yaxis()
        axes[ax_dict[0]].set(xticks=time_tick_labels, yticks=ytick_labels,
                             xticklabels=time_tick_values[time_tick_labels], yticklabels=ytick_values[ytick_labels],
                             xlabel='Time around update (s)', ylabel=f'{label} position')
        axes[ax_dict[0]].set_title(f'{title} trials - probability density - {label} position', fontsize=14)

        axes[ax_dict[1]].plot(times, stats['position']['mean'], color='k', label='True position')
        axes[ax_dict[1]].fill_between(times, stats['position']['lower'], stats['position']['upper'], alpha=0.2, color='k', label='95% CI')
        axes[ax_dict[1]].plot(times, stats['decoding']['mean'], color=color, label='Decoded position')
        axes[ax_dict[1]].fill_between(times, stats['decoding']['lower'], stats['decoding']['upper'], alpha=0.2, color=color, label='95% CI')
        axes[ax_dict[1]].plot([0, 0], limits, linestyle='dashed', color='k', alpha=0.25)
        axes[ax_dict[1]].set(xlim=[-window, window], ylim=limits, xlabel='Time around update(s)', ylabel=f'{label} position')
        axes[ax_dict[1]].legend(loc='upper left')
        axes[ax_dict[1]].set_title(f'{title} trials - decoded {label} position', fontsize=14)

        axes[ax_dict[2]] = sns.heatmap(data_around_update['position'][:, sort_index].T, cmap=cmap_pos, ax=axes[ax_dict[2]],
                                       vmin=scaling_value * limits[0], vmax=scaling_value * limits[1],
                                       cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
        axes[ax_dict[2]].plot(update_time_values[0], update_time_values[1], linestyle='dashed', color=[0, 0, 0, 0.5])
        axes[ax_dict[2]].set(xticks=time_tick_labels, xticklabels=time_tick_values[time_tick_labels],
                             xlabel='Time around update (s)', ylabel='Trials')
        axes[ax_dict[2]].set_title(f'{title} trials - true {label} position', fontsize=14)

        axes[ax_dict[3]] = sns.heatmap(data_around_update['decoding'][:, sort_index].T, cmap=cmap_decoding, ax=axes[ax_dict[3]],
                                       vmin=scaling_value * limits[0] , vmax=scaling_value * limits[1],
                                       cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
        axes[ax_dict[3]].plot(update_time_values[0], update_time_values[1], linestyle='dashed', color=[0, 0, 0, 0.5])
        axes[ax_dict[3]].set(xticks=time_tick_labels, xticklabels=time_tick_values[time_tick_labels],
                             xlabel='Time around update (s)', ylabel='Trials')
        axes[ax_dict[3]].set_title(f'{title} trials - decoded {label} position', fontsize=14)

        axes[ax_dict[4]].plot(times, stats['error']['mean'], color='r', label='|True - decoded|')
        axes[ax_dict[4]].fill_between(times, stats['error']['lower'], stats['error']['upper'], alpha=0.2, color='r', label='95% CI')
        axes[ax_dict[4]].plot([0, 0], [0, np.max(stats['error']['upper'])], linestyle='dashed', color='k', alpha=0.25)
        axes[ax_dict[4]].set(xlim=[-window, window], ylim=[0, np.max(stats['error']['upper'])], xlabel='Time around update(s)', ylabel=label)
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
        axes[ax_dict[0]].plot(stats['decoding_x']['mean'][:time_bin+1], stats['decoding_y']['mean'][:time_bin+1],
                              color=color, label='Decoded position')
        axes[ax_dict[0]].plot(stats['decoding_x']['mean'][time_bin], stats['decoding_y']['mean'][time_bin], color=color,
                              marker='o', markersize='10', label='Current decoded position')
        axes[ax_dict[0]].plot(track_bounds_xs, track_bounds_ys, color='black')
        axes[ax_dict[0]].set(xlim=[-25, 25], ylim=ylims, xlabel='X position', ylabel='Y position')
        axes[ax_dict[0]].legend(loc='lower left')
        axes[ax_dict[0]].text(0.65, 0.1, f'Time to update: {np.round(times[time_bin],2):.2f} s',
                              transform=axes[ax_dict[0]].transAxes, fontsize=14,
                              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.25))
        axes[ax_dict[0]].annotate('update cue on here', (2, stats['position_y']['mean'][int(len(times)/2)]),
                                  xycoords='data', xytext=(5, stats['position_y']['mean'][int(len(times)/2)]),
                                  textcoords='data', va='center', arrowprops=dict(arrowstyle='->'))
        axes[ax_dict[0]].annotate('correct side', (17*correct_multiplier, 250), textcoords='data', va='center')
        axes[ax_dict[0]].set_title(f'{title} trials - decoded vs. true position', fontsize=14)

        im = axes[ax_dict[1]].imshow(prob_map[:,:,time_bin], cmap='YlGnBu', origin='lower', aspect='auto',
                                     vmin=0, vmax=0.5 * np.nanmax(prob_map),
                                     extent=[xlims[0], xlims[1], ylims[0], ylims[1]])
        axes[ax_dict[1]].plot(positions_x, positions_y,color='k', label='True position')
        axes[ax_dict[1]].plot(positions_x[-1], positions_y[-1], color='k', marker='o', markersize='10', label='Current true position')
        axes[ax_dict[1]].plot(track_bounds_xs, track_bounds_ys, color='black')
        axes[ax_dict[1]].annotate('update cue on here', (2, stats['position_y']['mean'][int(len(times)/2)]),
                                  xycoords='data', xytext=(5, stats['position_y']['mean'][int(len(times)/2)]),
                                  textcoords='data', va='center', arrowprops=dict(arrowstyle='->'))
        axes[ax_dict[1]].set(xlim=[-25, 25], ylim=ylims,  xlabel='X position', ylabel='Y position')  # cutoff some bc lo
        axes[ax_dict[1]].set_title(f'{title} trials - probability density', fontsize=14)
        plt.colorbar(im, ax=axes[ax_dict[1]], label='Probability density', pad=0.04, location='right', fraction=0.046)

def interp1d_time_intervals(data, start_locs, stop_locs, new_times, time_offset, trials_to_flip):
    interpolated_position = []
    for start, stop, offset, flip_bool in zip(start_locs, stop_locs, time_offset, trials_to_flip):
        times = np.array(data.iloc[start:stop].index) - offset
        values = data.iloc[start:stop].values

        if flip_bool:
            values = -values

        fxn = interp1d(times, values, kind='linear')
        interpolated_position.append(fxn(new_times))

    return interpolated_position


def griddata_time_intervals(data, start_locs, stop_locs, time_offset, nbins, trials_to_flip):
    grid_prob = []
    for start, stop, offset, flip_bool in zip(start_locs, stop_locs, time_offset, trials_to_flip):
        proby = data.iloc[start:stop].stack().reset_index().values
        proby[:, 0] = proby[:, 0] - offset

        if flip_bool:
            proby[:, 1] = -proby[:, 1]  # flip position values so that prob density mapping is flipped

        x1 = np.linspace(min(proby[:, 0]), max(proby[:, 0]), nbins)  # time bins
        y1 = np.linspace(min(proby[:, 1]), max(proby[:, 1]), len(data.columns))  # position bins
        grid_x, grid_y = np.meshgrid(x1, y1)
        grid_prob_y = griddata(proby[:, 0:2], proby[:, 2], (grid_x, grid_y), method='linear', fill_value=np.nan)
        grid_prob.append(grid_prob_y)

    return grid_prob

def griddata_2d_time_intervals(data, binsxy, times, start_locs, stop_locs, time_offset, nbins, trials_to_flip):
    grid_prob = []
    flip_x = trials_to_flip['x'].values
    flip_y = trials_to_flip['y'].values
    for start, stop, offset, flip_bool_x, flip_bool_y in zip(start_locs, stop_locs, time_offset, flip_x, flip_y):
        times_around_update = times[start:stop] - offset
        data_subset = data[start:stop, :, :]

        prob_data = []
        for ind, t in enumerate(times_around_update):
            df = pd.DataFrame(data_subset[ind,:,:], index=binsxy[0], columns=binsxy[1])
            data_around_update = df.stack().reset_index().values
            data_from_timepoint = np.hstack((np.tile(t, len(data_around_update))[:,None], data_around_update))

            if flip_bool_x:
                data_from_timepoint[:, 1] = -data_from_timepoint[:, 1]  # flip position values so that prob density mapping is flipped

            if flip_bool_y:
                data_from_timepoint[:, 2] = -data_from_timepoint[:, 2]

            prob_data.append(data_from_timepoint)

        all_data = np.vstack(prob_data)
        t1 = np.linspace(min(times_around_update), max(times_around_update), nbins)  # time bins
        x1 = np.linspace(min(binsxy[0]), max(binsxy[0]), np.shape(data)[1])  # x_position bins
        y1 = np.linspace(min(binsxy[1]), max(binsxy[1]), np.shape(data)[2])  # y_position bins
        grid_x, grid_y, grid_t = np.meshgrid(x1, y1, t1)

        grid_prob_trial = griddata(all_data[:,0:3], all_data[:,3], (grid_t, grid_x, grid_y), method='linear', fill_value=np.nan)
        grid_prob.append(grid_prob_trial)

    return grid_prob


def get_stats(data, axis=0):

    mean = np.nanmean(data, axis=axis)
    err = sem(data, axis=axis)

    return {'mean': mean,
            'err': err,
            'upper': mean + 2*err,
            'lower': mean - 2*err,
            }

def create_track_boundaries():
    # establish track boundaries
    coords = [[2, 1], [2, 245], [25, 275], [25, 285], [14, 285], [0.5, 265], [-0.5, 265], [-14, 285], [-25, 285],
              [-25, 275], [-2, 245], [-2, 1], [2, 1]]
    xs, ys = zip(*coords)

    return xs, ys