import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns

from git import Repo
from pynwb import NWBHDF5IO
from pathlib import Path
from scipy.interpolate import griddata

from update_project.utils import get_session_info
from plots import show_event_aligned_psth, show_start_aligned_psth
from units import align_by_time_intervals

# set inputs
animals = [17, 20, 25, 28, 29] # 17, 20, 25, 28, 29
dates_included = [210521]
dates_excluded = []
overwrite_data = True

# load session info
base_path = Path('Y:/singer/NWBData/UpdateTask/')
spreadsheet_filename = 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
all_session_info = get_session_info(filename=spreadsheet_filename, animals=animals,
                                    dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

# get output figure info
repo = Repo(search_parent_directories=True)
short_hash = repo.head.object.hexsha[:10]
this_filename = __file__  # to add to metadata to know which script generated the figure

# loop through sessions and run conversion
for name, session in unique_sessions:

    # load file
    session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
    filename = base_path / f'{session_id}.nwb'
    io = NWBHDF5IO(str(filename), 'r')
    nwbfile = io.read()

    # get info for figure saving and labelling
    figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / f'{session_id}'
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    intermediate_data_path = figure_path / 'intermediate_data'

    # load position structure
    position_ss = nwbfile.processing['behavior']['position'].get_spatial_series('position')
    position = {'x': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 0]),
                'y': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 1])}
    position = pd.DataFrame.from_dict(position)
    trial_epochs = nwbfile.intervals['trials'].to_dataframe()
    non_update_epochs = trial_epochs[trial_epochs['update_type'] == 1]
    time_support_all_trials = nap.IntervalSet(start=trial_epochs['start_time'], end=trial_epochs['stop_time'], time_units='s')
    time_support_non_update = nap.IntervalSet(start=non_update_epochs['start_time'], end=non_update_epochs['stop_time'], time_units='s')
    position_tsg = nap.TsdFrame(position, time_units='s', time_support=time_support_all_trials)

    # get binning info
    nb_bins = 30  # position
    bin_size = 0.5  # seconds
    position_bins = np.linspace(np.min(position_tsg['y']), np.max(position_tsg['y']), nb_bins + 1)

    # load units structure
    units = nwbfile.units.to_dataframe()
    spikes_dict = {n:nap.Ts(t=units.loc[n,'spike_times'], time_units='s') for n in units.index}
    spikes = nap.TsGroup(spikes_dict, time_units='s')

    if overwrite_data or not Path(intermediate_data_path / 'decoding_2d.npz').is_file():
        Path(intermediate_data_path).mkdir(parents=True, exist_ok=True)

        # compute 2d tuning curves
        tuning_curves2d, binsxy = nap.compute_2d_tuning_curves(group=spikes, feature=position_tsg, nb_bins=nb_bins, ep=time_support_non_update)

        # decode 1d data
        tuning_curves1d = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg['y'], nb_bins=nb_bins, ep=time_support_non_update)
        decoded, proby_feature = nap.decode_1d(tuning_curves=tuning_curves1d,
                                               group=spikes,
                                               ep=time_support_all_trials,
                                               bin_size=bin_size,  # second
                                               feature=position_tsg['y'],
                                               )

        # get decoding error
        time_index = []
        position_mean = []
        for index, trial in time_support_all_trials.iterrows():
            trial_bins = np.arange(trial['start'],trial['end']+bin_size, bin_size)
            bins = pd.cut(position['y'].index, trial_bins)
            position_mean.append(position['y'].groupby(bins).mean())
            time_index.append(trial_bins[0:-1] + np.diff(trial_bins)/2)
        time_index = np.hstack(time_index)
        position_means = np.hstack(position_mean)

        actual_series = pd.Series(position_means, index=time_index, name='actual_position')
        decoded_series = decoded.as_series()  # TODO - figure out why decoded and actual series are diff lengths
        df_decode_results = pd.merge(decoded_series.rename('decoded_position'), actual_series, how='left', left_index=True, right_index=True)
        df_decode_results['decoding_error'] = abs(df_decode_results['decoded_position'] - df_decode_results['actual_position'])
        df_decode_results['decoding_error_rolling'] = df_decode_results['decoding_error'].rolling(20, min_periods=20).mean()
        df_decode_results['prob_dist'] = [x for x in proby_feature.as_dataframe().to_numpy()]

        # get decoding matrices
        bins = pd.cut(df_decode_results['actual_position'], position_bins)
        decoding_matrix = df_decode_results['decoded_position'].groupby(bins).apply(
            lambda x: np.histogram(x, position_bins, density=True)[0]).values
        decoding_matrix_prob = df_decode_results['prob_dist'].groupby(bins).apply(
            lambda x: np.nanmean(np.vstack(x.values), axis=0)).values

        # save intermediate data
        np.savez(intermediate_data_path / 'decoding_2d.npz', tuning_curves2d=tuning_curves2d, binsxy=binsxy)
        np.savez(intermediate_data_path / 'decoding_1d.npz', tuning_curves1d=tuning_curves1d, decoded=decoded,
                 proby_feature=proby_feature)
        np.savez(intermediate_data_path / 'decoding_matrix.npz', decoding_matrix=decoding_matrix, decoding_matrix_prob=decoding_matrix_prob)
        df_decode_results.to_csv(intermediate_data_path /'decoding_results.csv')
    else:
        decoding_2d = np.load(intermediate_data_path / 'decoding_2d.npz', allow_pickle=True)
        tuning_curves2d = decoding_2d['tuning_curves2d']
        binsxy = decoding_2d['binsxy']
        decoding_1d = np.load(intermediate_data_path / 'decoding_1d.npz', allow_pickle=True)
        tuning_curves1d = decoding_1d['tuning_curves1d']
        decoded = decoding_1d['decoded']
        proby_feature = decoding_1d['proby_feature']
        decoding_summary = np.load(intermediate_data_path / 'decoding_matrix.npz', allow_pickle=True)
        decoding_matrix_prob = decoding_summary['decoding_matrix_prob']
        decoding_matrix = decoding_summary['decoding_matrix']
        df_decode_results = pd.read_csv(intermediate_data_path /'decoding_results.csv', index_col=0)

    # plot the tuning curves
    counter = 0
    ncols = 3
    fig, axes = plt.subplots(nrows=ncols, ncols=ncols, figsize=(15, 15))
    for key, unit_tuning in tuning_curves2d.items():
        if (counter % (ncols ** 2) == 0) and (counter != 0):
            # wrap up previous plot and save
            fig.suptitle(f'Spatial tuning curves - {session_id}', fontsize=14)
            plt.tight_layout()
            filename = figure_path / f'spatialtuning_unit{key}_git{short_hash}.pdf'
            plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

            # setup next plot
            fig, axes = plt.subplots(nrows=ncols, ncols=ncols, figsize=(15, 15))
            counter = 0
        else:
            scaled_tuning = unit_tuning / np.nanmax(unit_tuning)
            col_id = int(np.floor(counter / ncols))
            axes[col_id][counter - col_id * ncols].imshow(scaled_tuning,
                                                          extent=(
                                                          binsxy[1][0], binsxy[1][-1], binsxy[0][0], binsxy[0][-1]),
                                                          aspect='auto',
                                                          cmap='YlOrRd',
                                                          vmin=0.1, vmax=0.9)
            axes[col_id][counter - col_id * ncols].set_title(f'Unit {key}')
            counter += 1

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
    x1 = np.linspace(min(prob[:,0]), max(prob[:,0]), int(end_time - start_time))
    y1 = np.linspace(min(prob[:,1]), max(prob[:,1]), len(proby_feature.columns))
    grid_x, grid_y = np.meshgrid(x1,y1)
    grid_prob = griddata(prob[:,0:2], prob[:,2], (grid_x, grid_y), method='nearest', fill_value=np.nan)

    im = axes['A'].imshow(grid_prob, aspect='auto', origin='lower', cmap='YlGnBu', # RdPu
                          extent=[start_time, end_time, position_bins[0], position_bins[-1]],
                          vmin=0, vmax=0.75)
    axes['A'].plot((position_tsg['y'].loc[start_time:end_time]), label='True', color=[0, 0, 0, 0.5], linestyle='dashed')
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
                            vmin=0, vmax=0.75*np.nanmax(np.vstack(decoding_matrix)),
                            cbar_kws={'pad':0.01, 'label':'proportion decoded', 'fraction':0.046})
    axes['D'].plot([0, 285], [0, 285], linestyle='dashed', color=[0, 0, 0, 0.5])
    axes['D'].invert_yaxis()
    axes['D'].set_title('Decoding accuracy - peak', fontsize=14)
    axes['D'].set(xticks=tick_labels, yticks=tick_labels,
                xticklabels=tick_values[tick_labels], yticklabels=tick_values[tick_labels],
                xlabel='Decoded Position', ylabel='Actual Position')

    axes['E'] = sns.heatmap(np.vstack(decoding_matrix_prob), cmap='YlGnBu', ax=axes['E'], square=True,
                            vmin=0, vmax=0.75 * np.nanmax(np.vstack(decoding_matrix_prob)),
                            cbar_kws={'pad':0.01, 'label':'mean probability', 'fraction':0.046})
    axes['E'].plot([0, 285], [0, 285], linestyle='dashed', color=[0, 0, 0, 0.5])
    axes['E'].invert_yaxis()
    axes['E'].set_title('Decoding accuracy - avg prob', fontsize=14)
    axes['E'].set(xticks=tick_labels, yticks=tick_labels,
                xticklabels=tick_values[tick_labels], yticklabels=tick_values[tick_labels],
                xlabel='Decoded Position', ylabel='Actual Position')

    axes['F'] = sns.ecdfplot(df_decode_results['decoding_error'], ax=axes['F'], color='k')
    axes['F'].set_title('Decoding accuracy - error')
    axes['F'].set(xlabel='Decoding error', ylabel='Proportion')
    axes['F'].set_aspect(1./axes['F'].get_data_ratio(), adjustable='box')

    plt.colorbar(im, ax=axes['A'], label='Probability density', pad=0.06, location='bottom', shrink=0.25, anchor=(0.9, 1))
    plt.tight_layout()

    # save figure
    filename = figure_path / f'decoding_summary_git{short_hash}.pdf'
    plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

    plt.close('all')
    io.close()

