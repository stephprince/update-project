import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

from git import Repo
from pynwb import NWBHDF5IO
from pathlib import Path

from update_project.utils import get_session_info
from decoding import get_decoding_around_update, plot_decoding_around_update, get_2d_decoding_around_update, plot_2d_decoding_around_update
from update_project.camera_sync.cam_plot_utils import write_camera_video

# set inputs
animals = [25]
dates_included = [210913] #210913
dates_excluded = []

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
    non_update_epochs = trial_epochs[trial_epochs['update_type'] == 1]
    time_support_all_trials = nap.IntervalSet(start=trial_epochs['start_time'], end=trial_epochs['stop_time'], time_units='s')
    time_support_non_update_trials = nap.IntervalSet(start=trial_epochs['start_time'], end=trial_epochs['stop_time'],
                                                     time_units='s')
    position_tsg = nap.TsdFrame(position, time_units='s', time_support=time_support_all_trials)

    # load units structure
    units = nwbfile.units.to_dataframe()
    spikes_dict = {n:nap.Ts(t=units.loc[n,'spike_times'], time_units='s') for n in units.index}
    spikes = nap.TsGroup(spikes_dict, time_units='s')

    # decode 1d data
    nb_bins = 30 # position
    bin_size = 0.5 # seconds
    tuning_curves1d_y = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg['y'], nb_bins=nb_bins, ep=time_support_non_update_trials)
    tuning_curves1d_x = nap.compute_1d_tuning_curves(group=spikes, feature=position_tsg['x'], nb_bins=nb_bins, ep=time_support_non_update_trials)
    decoded_y, proby_feature = nap.decode_1d(tuning_curves=tuning_curves1d_y,
                                           group=spikes,
                                           ep=time_support_all_trials,
                                           bin_size=bin_size,  # second
                                           feature=position_tsg['y'],
                                           )
    decoded_x, probx_feature = nap.decode_1d(tuning_curves=tuning_curves1d_x,
                                             group=spikes,
                                             ep=time_support_all_trials,
                                             bin_size=bin_size,  # second
                                             feature=position_tsg['x'],
                                             )

    # decode 2d data
    tuning_curves2d, binsxy = nap.compute_2d_tuning_curves(group=spikes, feature=position_tsg, nb_bins=nb_bins, ep=time_support_non_update_trials)
    decoded_xy, probxy_feature = nap.decode_2d(tuning_curves=tuning_curves2d,
                                               group=spikes,
                                               ep=time_support_all_trials,
                                               bin_size=bin_size,  # second
                                               xy=binsxy,
                                               features=position_tsg
                                               )

    # get 1d decoding before and after update cue occurs
    window = 5  # seconds around update
    nbins = 50

    switch_trials = trial_epochs[trial_epochs['update_type'] == 2]
    y_around_switch = get_decoding_around_update(position_tsg['y'], decoded_y, proby_feature, switch_trials,
                                                 nbins=nbins, window=window, flip_trials=False)
    x_around_switch = get_decoding_around_update(position_tsg['x'], decoded_x, probx_feature, switch_trials,
                                                 nbins=nbins, window=window, flip_trials=True)

    stay_trials = trial_epochs[trial_epochs['update_type'] == 3]
    y_around_stay = get_decoding_around_update(position_tsg['y'], decoded_y, proby_feature, stay_trials,
                                               nbins=nbins, window=window, flip_trials=False)
    x_around_stay = get_decoding_around_update(position_tsg['x'], decoded_x, probx_feature, stay_trials,
                                               nbins=nbins, window=window, flip_trials=True)

    # get 2d decoding before and after update
    switch_trials = trial_epochs[trial_epochs['update_type'] == 2]
    xy_around_switch = get_2d_decoding_around_update(position_tsg, decoded_xy, probxy_feature, binsxy, switch_trials,
                                                 nbins=nbins, window=window)

    stay_trials = trial_epochs[trial_epochs['update_type'] == 3]
    xy_around_stay = get_2d_decoding_around_update(position_tsg, decoded_xy, probxy_feature, binsxy, stay_trials,
                                               nbins=nbins, window=window)

    # plot the decoding data
    mosaic = """
    AEIM
    BFJN
    CGKO
    DHLP
    """
    axes = plt.figure(figsize=(20, 15)).subplot_mosaic(mosaic)
    plot_decoding_around_update(y_around_switch, nbins, window, 'switch', 'y position', [0, 258], 'b', axes, ['A', 'B', 'C', 'D'])
    plot_decoding_around_update(x_around_switch, nbins, window, 'switch', 'x position', [-20, 20], 'b', axes, ['E', 'F', 'G', 'H'])
    plot_decoding_around_update(y_around_stay, nbins, window, 'stay', 'y position', [0, 258], 'm', axes, ['I', 'J', 'K', 'L'])
    plot_decoding_around_update(x_around_stay, nbins, window, 'stay', 'x position', [-20, 20], 'm', axes, ['M', 'N', 'O', 'P'])
    plt.suptitle(f'{session_id} decoding around update trials', fontsize=20)
    plt.tight_layout()

    filename = figure_path / f'decoding_around_update_git{short_hash}.pdf'
    plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

    # plot 2d decoding around update
    times = np.linspace(-window, window, num=nbins)
    Path(figure_path / 'timelapse_figures').mkdir(parents=True, exist_ok=True)
    plot_filenames = []
    for time_bin in range(nbins):
        mosaic = """
            AB
            CD
            """
        axes = plt.figure(figsize=(20, 10)).subplot_mosaic(mosaic)
        plot_2d_decoding_around_update(xy_around_switch, time_bin, times, 'switch', 'b', axes, ['A', 'B'])
        plot_2d_decoding_around_update(xy_around_stay, time_bin, times, 'stay', 'm', axes, ['C', 'D'])
        plt.suptitle(f'{session_id} decoding around update trials', fontsize=20)
        plt.tight_layout()

        filename =  figure_path / 'timelapse_figures'/ f'decoding_around_update_frame_no{time_bin}_git{short_hash}.png'
        plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})
        plt.close()
        plot_filenames.append(filename)

    # make videos for each plot type
    fps = 5.0
    vid_filename = f'decoding_around_update_video_git{short_hash}'
    write_camera_video(figure_path, fps, plot_filenames, vid_filename)

    plt.close('all')
    io.close()



