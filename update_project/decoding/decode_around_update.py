import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns

from git import Repo
from pynwb import NWBHDF5IO
from pathlib import Path
from sklearn.metrics import confusion_matrix
from scipy.interpolate import griddata, interp1d
from scipy.stats import sem

from update_project.utils import get_session_info
from plots import show_event_aligned_psth, show_start_aligned_psth
from units import align_by_time_intervals

# set inputs
animals = [25]
dates_included = [210913]
dates_excluded = []  # need to run S20_210511, S20_210519, S25_210909, S28_211118 later, weird spike sorting table issue

# load session info
base_path = Path('Y:/singer/NWBData/UpdateTask/')
spreadsheet_filename = 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
all_session_info = get_session_info(filename=spreadsheet_filename, animals=animals,
                                    dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

# loop through sessions and run conversion
for name, session in unique_sessions:

    # load file
    session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
    filename = base_path / f'{session_id}.nwb'
    io = NWBHDF5IO(str(filename), 'r')
    nwbfile = io.read()

    # get info for figure saving and labelling
    repo = Repo(search_parent_directories=True)
    short_hash = repo.head.object.hexsha[:10]
    this_filename = __file__  # to add to metadata to know which script generated the figure
    figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / f'{session_id}'
    Path(figure_path).mkdir(parents=True, exist_ok=True)

    # load position structure
    position_ss = nwbfile.processing['behavior']['position'].get_spatial_series('position')
    position = {'x': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 0]),
                'y': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 1])}
    position = pd.DataFrame.from_dict(position)
    trial_epochs = nwbfile.intervals['trials'].to_dataframe()
    time_support = nap.IntervalSet(start=trial_epochs['start_time'], end=trial_epochs['stop_time'], time_units='s')
    position_tsg = nap.TsdFrame(position, time_units='s', time_support=time_support)

    # load units structure
    units = nwbfile.units.to_dataframe()
    spikes_dict = {n:nap.Ts(t=units.loc[n,'spike_times'], time_units='s') for n in units.index}
    spikes = nap.TsGroup(spikes_dict, time_units='s')

    # decode 1d data
    nb_bins_y = 30 # y position
    nb_bins_x = 30 # x position
    bin_size = 0.5 # seconds
    tuning_curves1d_y = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg['y'], nb_bins=nb_bins_y)
    tuning_curves1d_x = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg['x'], nb_bins=nb_bins_x)
    decoded_y, proby_feature = nap.decode_1d(tuning_curves=tuning_curves1d_y,
                                           group=spikes,
                                           ep=time_support,
                                           bin_size=bin_size,  # second
                                           feature=position_tsg['y'],
                                           )
    decoded_x, probx_feature = nap.decode_1d(tuning_curves=tuning_curves1d_x,
                                             group=spikes,
                                             ep=time_support,
                                             bin_size=bin_size,  # second
                                             feature=position_tsg['x'],
                                             )

    # get decoding before and after update cue occurs
    window = 5  # seconds around update
    nbins = 50

    switch_trials = trial_epochs[trial_epochs['update_type'] == 2]
    y_around_switch = get_decoding_data_around_update(position_tsg['y'], decoded_y, proby_feature, switch_trials, nbins=nbins, window=window)
    x_around_switch = get_decoding_data_around_update(position_tsg['x'], decoded_x, probx_feature, switch_trials, nbins=nbins, window=window)

    stay_trials = trial_epochs[trial_epochs['update_type'] == 3]
    y_around_stay = get_decoding_data_around_update(position_tsg['y'], decoded_y, proby_feature, stay_trials, nbins=nbins, window=window)
    x_around_stay = get_decoding_data_around_update(position_tsg['x'], decoded_x, probx_feature, stay_trials, nbins=nbins, window=window)

    # plot the decoding data
    mosaic = """
    ADGJ
    BEHK
    CFIL
    """
    axes = plt.figure(figsize=(15, 15)).subplot_mosaic(mosaic)

    plot_decoding_around_update(y_around_switch, nbins, window, 'switch', 'y position', limits, 'b', {'A', 'B', 'C'})
    plot_decoding_around_update(x_around_switch, nbins, window, 'switch', 'x position', limits, 'b', {'D', 'E', 'F'})
    plot_decoding_around_update(y_around_stay, nbins, window, 'stay', 'y position', limits, 'm', {'G', 'H', 'I'})
    plot_decoding_around_update(x_around_stay, nbins, window, 'stay', 'x position', limits, 'm', {'J', 'K', 'L'})

    plt.tight_layout()
    plt.show()

    # save figure
    filename = figure_path / f'decoding_around_update_git{short_hash}.pdf'
    plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

    plt.close('all')
    io.close()


def get_decoding_around_update(position, decoded_data, prob_feature, trials, nbins=50, window=5):
    update_times = trials['t_update']

    pos_start_locs = position.index.searchsorted(update_times - window - 1)  # a little extra just in case
    pos_stop_locs = position.index.searchsorted(update_times + window + 1)
    pos_interp = interp1d_time_intervals(position, pos_start_locs, pos_stop_locs, update_times)
    pos_out = np.array(pos_interp).T

    decoding_start_locs = decoded_data.index.searchsorted(update_times - window - 1)
    decoding_stop_locs = decoded_data.index.searchsorted(update_times + window + 1)
    decoding_interp = interp1d_time_intervals(decoded_data, decoding_start_locs, decoding_stop_locs, update_times)
    decoding_out = np.array(decoding_interp.T)

    prob_out = griddata_time_intervals(prob_feature, decoding_start_locs, decoding_stop_locs, update_times, nbins)

    # get means and sem
    stats['position'] = get_stats(pos_out)
    stats['decoding'] = get_stats(decoding_out)
    stats['probability'] = get_stats(prob_out)

    return {'position': pos_out,
            'decoding': decoding_out,
            'probability': prob_out,
            'stats': stats}

def plot_decoding_around_update(data_around_update, nbins, window, title, label, limits, color, ax_dict):

    stats = data_around_update['stats']
    times = np.linspace(-window, window, num=nbins)
    time_tick_values = times.astype(int)
    time_tick_labels = np.array([0, int(len(time_tick_values) / 2), len(time_tick_values) - 1])

    axes[ax_dict[0]].plot(times, data_around_update['position'], color='k', alpha=0.25, label='True')
    axes[ax_dict[0]].plot(times, data_around_update['decoding'], color=color, alpha=0.25, label='Decoded')
    axes[ax_dict[0]].set(xlim=[-window, window], ylim=[0, 285], xlabel='Time around update(s)', ylabel=label)
    axes[ax_dict[0]].plot([0, 0], limits, linestyle='dashed', color='k', alpha=0.25)
    axes[ax_dict[0]].set_title(f'{session_id} decoding around {title} trials - {label}', fontsize=16)

    axes[ax_dict[1]].plot(times, stats['position']['mean'], color='k', label='True position')
    axes[ax_dict[1]].fill_between(times,stats['position']['lower'], stats['position']['upper'], alpha=0.2, color='k')
    axes[ax_dict[1]].plot(times, stats['decoding']['mean'], color=color, label='Decoded position')
    axes[ax_dict[1]].fill_between(times, stats['decoding']['lower'], stats['decoding']['upper'], alpha=0.2, color=color)
    axes[ax_dict[1]].plot([0, 0], limits, linestyle='dashed', color='k', alpha=0.25)
    axes[ax_dict[1]].set(xlim=[-window, window], ylim=limits, xlabel='Time around update(s)', ylabel=label)
    axes[ax_dict[1]].legend(loc='upper right')
    axes[ax_dict[1]].set_title(f'Average decoded {label}', fontsize=14)

    ytick_values = tuning_curves1d_y.index.values.astype(int)
    ytick_labels = np.array([0, int(len(ytick_values) / 2), len(ytick_values) - 1])

    axes[ax_dict[2]] = sns.heatmap(data_around_update['probability'], cmap='YlGnBu', ax=axes['E'],
                                    vmin=0, vmax=0.75 * np.nanmax(data_around_update['probability']),
                                    cbar_kws={'pad': 0.01, 'label': 'proportion decoded', 'fraction': 0.046})
    axes[ax_dict[2]].plot([len(time_tick_values)/2, len(time_tick_values)/2], [0, len(ytick_values)], linestyle='dashed', color=[0, 0, 0, 0.5])
    axes[ax_dict[2]].invert_yaxis()
    axes[ax_dict[2]].set(xticks=time_tick_labels, yticks=ytick_labels,
                         xticklabels=time_tick_values[time_tick_labels], yticklabels=ytick_values[ytick_labels],
                         xlabel='Time around update (s)', ylabel=label)
    axes[ax_dict[2]].set_title(f'Average probability density - {label}', fontsize=14)


def interpolate_time_intervals(data, start_locs, stop_locs, time_offset):
    interpolated_position = []
    for start, stop, offset in zip(start_locs, stop_locs, time_offset):
        times = np.array(data.iloc[start:stop].index) - offset
        values = data.iloc[start:stop].values

        fxn = interp1d(times, values, kind='linear')
        interpolated_position.append(fxn(new_times))

    return interpolated_position

def griddata_time_intervals(data, start_locs, stop_locs, time_offset, nbins):
    grid_prob = []
    for start, stop, offset in zip(start_locs, stop_locs, time_offset):
        proby = data.iloc[start:stop].stack().reset_index().values
        proby[:, 0] = proby[:, 0] - t_update
        x1 = np.linspace(min(proby[:, 0]), max(proby[:, 0]), nbins)
        y1 = np.linspace(min(proby[:, 1]), max(proby[:, 1]), len(proby_feature.columns))
        grid_x, grid_y = np.meshgrid(x1, y1)
        grid_prob_y = griddata(proby[:, 0:2], proby[:, 2], (grid_x, grid_y), method='linear', fill_value=np.nan)
        grid_prob.append(grid_prob_y)

    return grid_prob

def get_stats(data, axis=0):
    mean = np.nanmean(data, axis=axis)
    err = sem(data, axis=axis)

    return {'mean': mean,
            'err': err,
            'upper': mean + 2 * err,
            'lower': mean - 2 * err,
            }