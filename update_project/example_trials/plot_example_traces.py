import cv2
import matplotlib.pyplot as plt
import numpy as np

from pynwb import NWBHDF5IO
from spikeinterface.extractors import read_nwb_recording
from spikeinterface.preprocessing import filter, common_reference, whiten, center, scale
from spikeinterface.widgets import plot_timeseries

from update_project.general.session_loader import SessionLoader
from update_project.general.results_io import ResultsIO
from update_project.general.plots import get_color_theme
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts


overwrite = False
plot_raw = False
plot_spikes = True

# set params
downsample = 15  # amount to downsample the raw ephys trace (usually 30kHz)
n_trials = 3  # number of trials to plot from each trial type
channels = [(0, 32), (32, 64)]
video_duration = 5  # seconds of full video at 1X speed
raw_window = 2  # seconds to show as max length at any time
spike_window = 0.2  # seconds to show as max length for spiking
fps = 60  # frames per second
video_speed = 0.25  # speed to slow video down to for plotting
colors = get_color_theme()

# setup sessions
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = [210518] # , 210913, 220617, 220623]
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

# loop through individual sessions
for name in session_names:
    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()
    results_io = ResultsIO(creator_file=__file__, session_id=session_db.get_session_id(name),
                           folder_name='example_trials', )

    # # get spike data
    recording_raw = read_nwb_recording(session_db.get_session_path(name), electrical_series_name='raw_ecephys')
    recording_s = center(recording_raw, mode='mean')
    recording_cmr = common_reference(recording_s, operator='median', groups=[list(np.r_[:64]), list(np.r_[64:128])])
    recording_f = filter(recording_cmr, band=[300, 6000], btype='bandpass')
    spike_data = whiten(recording_f)

    # get channel data
    electrode_df = nwbfile.electrodes.to_dataframe()
    channel_regions = electrode_df['location'].to_numpy()
    channel_regions = channel_regions[channel_regions != 'none']
    CA1_channels = np.argwhere(channel_regions == 'CA1').squeeze()

    # get acquisition data
    trials_df = nwbfile.trials.to_dataframe()
    trial_inds = trials_df.query('maze_id == 4 & update_type == 3').index[:n_trials].to_numpy()
    lfp_data, lfp_times = align_by_time_intervals_ts(nwbfile.processing['ecephys']['LFP']['LFP'],
                                                     nwbfile.intervals['trials'][trial_inds],
                                                     return_timestamps=True)
    raw_data, raw_times_no_downsample = align_by_time_intervals_ts(nwbfile.acquisition['raw_ecephys'],
                                                     nwbfile.intervals['trials'][trial_inds],
                                                     return_timestamps=True)
    raw_data = [r[::downsample] for r in raw_data]
    raw_times = [r[::downsample] for r in raw_times_no_downsample]
    samprate = nwbfile.acquisition['raw_ecephys'].rate / downsample

    # plot lfp data
    time_window = raw_window * samprate  # n seconds
    time_window_small = spike_window * samprate * downsample
    n_frames = int(video_duration * fps / video_speed)
    chunk_size = samprate / fps * video_speed
    chunk_size_small = samprate * downsample / fps * video_speed
    for ind, trial in enumerate(trial_inds):
        for ch in channels:
            channel_inds = np.r_[CA1_channels[ch[0]]:CA1_channels[ch[-1] - 1] + 1]  # np.r_[0:32, 96:128]  use for lfp and raw channel plots

            # plot the raw traces
            plot_filenames = []
            plot_filenames_spikes = []
            if plot_raw:
                for frame in range(n_frames):
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

                    stop_time = int(chunk_size * (frame + 1))
                    start_time = 0 if chunk_size * frame < time_window else int(stop_time - time_window)
                    times = raw_times[ind][start_time:stop_time]

                    for i, reg, offset in zip(channel_inds, channel_regions[channel_inds], range(len(channel_inds))):
                        ax.plot(times, raw_data[ind][start_time:stop_time, i] - 0.5 * offset, color='k', linewidth=0.75)
                    ax.set(xlim=(times[0], times[0] + raw_window), ylim=((-len(channel_inds) / 2) - 1, 1), ylabel='channels',
                           xlabel='time (s)', yticks=[], xticks=[])
                    for loc in ['right', 'top', 'left', 'bottom']:
                        ax.spines[loc].set_visible(False)

                    # save files and append to list
                    kwargs = results_io.get_figure_args(filename='ephys_trace',
                                                        results_type='session',
                                                        results_name='temp_video_files',
                                                        additional_tags=f'trial{trial}_{ch}_{frame}',
                                                        format='png')
                    kwargs.update(dpi=150)
                    fig.savefig(**kwargs)
                    plt.close(fig)
                    plot_filenames.append(str(kwargs['fname']))

            # plot some example spiking
            if plot_spikes:
                for frame in range(n_frames):
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

                    stop_time = int((chunk_size * (frame)) + time_window_small)  # start with a full view
                    start_time = int(stop_time - time_window_small)
                    times = raw_times_no_downsample[ind][start_time:stop_time]

                    frame_range = (times * spike_data.get_sampling_frequency()).astype('int64')
                    traces = spike_data.get_traces(start_frame=frame_range[0], end_frame=frame_range[-1] + 1,
                                                   channel_ids=list(channel_inds))
                    for i, reg, offset in zip(channel_inds, channel_regions[channel_inds], range(len(channel_inds))):
                        if len(times) > len(traces[:, offset]):
                            plot_times = times[:-1]
                        elif len(times) < len(traces[:, offset]):
                            plot_traces = traces[:-1, offset] - 15 * offset
                        else:
                            plot_times = times
                            plot_traces = traces[:, offset] - 15 * offset
                        ax.plot(plot_times, plot_traces, color='k', linewidth=0.75)

                    # ts = plot_timeseries(spike_data, backend='matplotlib', time_range=[times[0], times[-1]],
                    #                      mode='line', channel_ids=list(channel_inds), ax=ax)
                    ax.set(xlim=(times[0], times[0] + spike_window), yticks=[], xticks=[],
                           ylim=(-15*len(channel_inds), 15))
                    for loc in ['right', 'top', 'left', 'bottom']:
                        ax.spines[loc].set_visible(False)

                    # if it's the last file, save as a pdf so it's manipulable
                    if frame == (n_frames - 1):
                        kwargs = results_io.get_figure_args(filename='ephys_trace_spikes',
                                                            results_type='session',
                                                            results_name='temp_video_files',
                                                            additional_tags=f'trial{trial}_{ch}_{frame}',
                                                            format='pdf')
                        fig.savefig(**kwargs)

                    # save files and append to list
                    kwargs = results_io.get_figure_args(filename='ephys_trace_spikes',
                                                        results_type='session',
                                                        results_name='temp_video_files',
                                                        additional_tags=f'trial{trial}_{ch}_{frame}',
                                                        format='png')
                    kwargs.update(dpi=150)
                    fig.savefig(**kwargs)
                    plt.close(fig)
                    plot_filenames_spikes.append(str(kwargs['fname']))

            # make videos for each plot type
            if plot_raw and plot_spikes:
                data_to_plot = ['spikes', 'raw']
            elif plot_raw:
                data_to_plot = ['raw']
            elif plot_spikes:
                data_to_plot = ['spikes']

            for type in data_to_plot:
                fnames = plot_filenames if type == 'raw' else plot_filenames_spikes
                video_filename = str(results_io.get_figure_args(filename=f'{type}_ephys_video',
                                                                results_type='session',
                                                                additional_tags=f'trial{trial}_{ch}',
                                                                format='mp4')['fname'])
                test_frame = cv2.imread(fnames[0])
                height, width, layers = test_frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
                for filename in fnames:
                    video.write(cv2.imread(filename))

                cv2.destroyAllWindows()
                video.release()
