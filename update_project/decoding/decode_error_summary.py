import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns

from git import Repo
from math import sqrt
from pynwb import NWBHDF5IO
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.interpolate import griddata

from update_project.utils import get_session_info
from plots import show_event_aligned_psth, show_start_aligned_psth
from units import align_by_time_intervals

# set inputs
animals = [17, 20, 25, 28, 29] # 17, 20, 25, 28, 29
dates_included = []
dates_excluded = []
overwrite_data = False

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

# get bins data
nb_bins = 30  # position
bin_size = 0.5  # seconds

# loop through sessions to accumulate data
intermediate_data_path = Path().absolute().parent.parent / 'results' / 'decoding' / 'intermediate_data'
Path(intermediate_data_path).mkdir(parents=True, exist_ok=True)

for dimension in ['x', 'y']:
    # load up the data
    filename = f'decoding_error_group_summary_{dimension}position.csv'
    if overwrite_data or not Path(intermediate_data_path / filename).is_file():
        get_decoding_error_summary(filename, nb_bins, bin_size, unique_sessions)
    else:
        all_decoding_data = pd.read_csv(intermediate_data_path / filename)

    position_bins = np.linspace(np.min(all_decoding_data['actual_position']), np.max(all_decoding_data['actual_position']), nb_bins + 1)
    bins = pd.cut(all_decoding_data['actual_position'], position_bins)
    decoding_matrix = all_decoding_data['decoded_position'].groupby(bins).apply(
        lambda x: np.histogram(x, position_bins, density=True)[0]).values
    decoding_matrix_prob = all_decoding_data['prob_dist'].groupby(bins).apply(
        lambda x: np.nanmean(np.vstack(x.values), axis=0)).values

    session_rmse = all_decoding_data[['animal', 'session', 'session_rmse']].groupby(['animal', 'session']).mean().reset_index()

    # plot data for all sessions
    mosaic = """
    AABCC
    AAFGG
    """

    axes = plt.figure(figsize=(20, 10)).subplot_mosaic(mosaic)

    # plot average decoding matrix (test)
    tick_values = position_bins.astype(int)
    tick_labels = np.array([0, int(len(tick_values) / 2), len(tick_values) - 1])
    axes['A'] = sns.heatmap(np.vstack(decoding_matrix_prob)*nb_bins, cmap='YlGnBu', ax=axes['A'], square=True,
                                vmin=0, vmax=0.3 * np.nanmax(np.vstack(decoding_matrix_prob))*nb_bins,
                                cbar_kws={'pad':0.01, 'label':'probability/chance', 'fraction':0.046})
    axes['A'].plot([0, 285], [0, 285], linestyle='dashed', color=[0, 0, 0, 0.5])
    axes['A'].invert_yaxis()
    axes['A'].set_title('Decoding accuracy - avg prob', fontsize=14)
    axes['A'].set(xlabel='Decoding position', ylabel='Actual position')
    axes['A'].set(xticks=tick_labels, yticks=tick_labels,
                xticklabels=tick_values[tick_labels], yticklabels=tick_values[tick_labels])

    # plot box/violin plot of decoding errors across sessions
    axes['B'] = sns.violinplot(data=all_decoding_data, y='decoding_error', color='k', ax=axes['B'])
    plt.setp(axes['B'].collections, alpha=.25)
    axes['B'].set_title('Group error distribution')

    axes['C'] = sns.violinplot(data=all_decoding_data, y='decoding_error', x='animal', palette='husl', ax=axes['C'])
    plt.setp(axes['C'].collections, alpha=.25)
    axes['C'].set_title('Individual error distribution')

    axes['F'] = sns.violinplot(data=session_rmse, y='session_rmse', color='k', ax=axes['F'])
    plt.setp(axes['F'].collections, alpha=.25)
    sns.stripplot(data=session_rmse, y='session_rmse', color="k", size=3, jitter=True, ax=axes['F'])
    axes['F'].set_title('Group session RMSE distribution')

    axes['G'] = sns.violinplot(data=session_rmse, y='session_rmse', x='animal', palette='husl', ax=axes['G'])
    plt.setp(axes['G'].collections, alpha=.25)
    sns.stripplot(data=session_rmse, y='session_rmse', x='animal', color="k", size=3, jitter=True, ax=axes['G'])
    axes['G'].set_title('Individual session RMSE distribution')

    plt.suptitle(f'Group decoding accuracy', fontsize=20)
    plt.tight_layout()

    # save figure
    figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / 'group_summary'
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    filename = figure_path / f'decoding_error_summary_{dimension}position_git{short_hash}.pdf'
    plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})


def get_decoding_error_summary(out_filename, size_bins, n_bins, sessions):
    decode_df_list = []
    for name, session in sessions:

        # load file
        session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
        filename = base_path / f'{session_id}.nwb'
        io = NWBHDF5IO(str(filename), 'r')
        nwbfile = io.read()

        # get info for figure saving and labelling
        print(f"Getting error data for {session_id}")
        figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / f'{session_id}'
        Path(figure_path).mkdir(parents=True, exist_ok=True)
        intermediate_data_path = figure_path / 'intermediate_data'

        # split data into train and test
        trial_epochs = nwbfile.intervals['trials'].to_dataframe()
        non_update_epochs = trial_epochs[trial_epochs['update_type'] == 1]
        train_data, test_data = train_test_split(non_update_epochs, test_size=0.25, random_state=21)
        time_support_all = nap.IntervalSet(start=non_update_epochs['start_time'],
                                           end=non_update_epochs['stop_time'],
                                           time_units='s')
        time_support_train = nap.IntervalSet(start=train_data['start_time'], end=train_data['stop_time'],
                                             time_units='s')
        time_support_test = nap.IntervalSet(start=test_data['start_time'], end=test_data['stop_time'],
                                            time_units='s')

        # load position structure
        position_ss = nwbfile.processing['behavior']['position'].get_spatial_series('position')
        position = {'x': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 0]),
                    'y': pd.Series(index=position_ss.timestamps[:], data=position_ss.data[:, 1])}
        position = pd.DataFrame.from_dict(position)
        position_tsg = nap.TsdFrame(position, time_units='s', time_support=time_support_all)

        # load units structure
        units = nwbfile.units.to_dataframe()
        spikes_dict = {n: nap.Ts(t=units.loc[n, 'spike_times'], time_units='s') for n in units.index}
        spikes = nap.TsGroup(spikes_dict, time_units='s')

        # decode 1d data
        tuning_curves1d = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg['y'], nb_bins=n_bins,
                                                       ep=time_support_train)
        decoded, proby_feature = nap.decode_1d(tuning_curves=tuning_curves1d,
                                               group=spikes,
                                               ep=time_support_test,
                                               bin_size=bin_size,  # second
                                               feature=position_tsg['y'],
                                               )

        # get decoding error
        time_index = []
        position_mean = []
        for index, trial in time_support_test.iterrows():
            trial_bins = np.arange(trial['start'], trial['end'] + size_bins, size_bins)
            bins = pd.cut(position['y'].index, trial_bins)
            position_mean.append(position['y'].groupby(bins).mean())
            time_index.append(trial_bins[0:-1] + np.diff(trial_bins) / 2)
        time_index = np.hstack(time_index)
        position_means = np.hstack(position_mean)

        actual_series = pd.Series(position_means, index=time_index, name='actual_position')
        decoded_series = decoded.as_series()  # TODO - figure out why decoded and actual series are diff lengths
        df_decode_results = pd.merge(decoded_series.rename('decoded_position'), actual_series, how='left',
                                     left_index=True, right_index=True)
        df_decode_results['decoding_error'] = abs(
            df_decode_results['decoded_position'] - df_decode_results['actual_position'])
        df_decode_results['prob_dist'] = [x for x in proby_feature.as_dataframe().to_numpy()]

        # add summary data
        df_positions = df_decode_results[['actual_position', 'decoded_position']].dropna(how='any')
        mse = mean_squared_error(df_positions['actual_position'], df_positions['decoded_position'])
        rmse = sqrt(mse)
        df_decode_results['session_rmse'] = rmse
        df_decode_results['animal'] = name[1]
        df_decode_results['session'] = name[2]

        # append to the list
        decode_df_list.append(df_decode_results)

    # get data from all sessions
    all_decoding_data = pd.concat(decode_df_list, axis=0, ignore_index=True)
    all_decoding_data.to_csv(intermediate_data_path / out_filename)

    return all_decoding_data
