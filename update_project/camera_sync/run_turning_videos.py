# import libraries
import cv2
import glob
import numpy as np
import pandas as pd
import warnings
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pathlib import Path
from pynwb import NWBHDF5IO

from update_project.camera_sync.cam_utils import convert_mat_file_to_dict, get_trial_intervals
from update_project.general.session_loader import SessionLoader
from update_project.general.results_io import ResultsIO
from update_project.general.plots import get_color_theme

n_trials = 25  # number of trials to plot from each trial type
video_duration = 5  # seconds of full video
colors = get_color_theme()
base_path = "y:/singer"  # "/Volumes/labs/"

# setup sessions
animals = [29]
dates_included = [211027]  # [210518, 210908, 210912, 210913, 220617, 220623]
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded,
                           behavior_only=True)
session_names = session_db.load_session_names()

# loop through individual sessions
for name in session_names:
    # load nwb file
    session_id = session_db.get_session_id(name)
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()
    results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name='camera', )

    trials_df = nwbfile.trials.to_dataframe()
    trial_inds = trials_df.query('maze_id == 4 & update_type == 3').index[:n_trials].to_numpy()

    # load virmen and camera data
    video_dir = Path(f"{base_path}/CameraData/UpdateTask/{session_id}/")
    virmen_filename = Path(f"{base_path}/Virmen Logs/UpdateTask/{session_id}_1/virmenDataRaw.mat")

    # import virmen data
    if virmen_filename.is_file():
        virmen_data = convert_mat_file_to_dict(virmen_filename)
        virmen_df = pd.DataFrame(virmen_data['virmenData']['data'], columns=virmen_data['virmenData']['dataHeaders'])
    else:
        warnings.warn('File can not be found')

    # import video frames
    video_file = video_dir / "*.mp4"
    video_file = str(video_file)
    video_file = glob.glob(video_file)[0]
    cap = cv2.VideoCapture(video_file)

    # check that the virmen data and the camera data have the same number of frames so that we can properly align them
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    virmen_frame_count = sum(virmen_df.cameraTrigger)

    assert video_frame_count == virmen_frame_count

    # get trial intervals
    camera_trig = virmen_df.index[virmen_df.cameraTrigger == 1].to_list()
    trial_intervals = get_trial_intervals(virmen_df, camera_trig, 'switch', event='update', n_trials=n_trials,
                                          time_window=(video_duration / 2))

    # save images for each frame from those trials
    camera_frames_all = []
    for i, interval in enumerate(trial_intervals):
        print(f"Saving frames for interval {i}")
        camera_frames = np.where(np.logical_and(camera_trig > interval[0], camera_trig < interval[1]))[0]
        camera_frames_all.append(camera_frames)
        frame_no = camera_frames[0]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        while frame_no <= camera_frames[-1]:
            ret, frame = cap.read()
            filename = str(results_io.get_figure_args(filename='turning',
                                                      results_type='session',
                                                      results_name=f'temp_video_files/trial{i}',
                                                      additional_tags=f'{frame_no}',
                                                      format='png')['fname'])
            cv2.imwrite(filename, frame)
            frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

    # make plot for each camera frame
    n_frames_min = np.min([len(f) for f in camera_frames_all])
    fps = n_frames_min / video_duration
    plot_filenames, plot_filenames_first_trial = [], []
    for frame_ind in range(n_frames_min):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(nrows=int(np.sqrt(n_trials)), ncols=int(np.sqrt(n_trials)),
                               figsize=(9, 6))
        for trial, interval in enumerate(trial_intervals):
            frame = camera_frames_all[trial][frame_ind]
            row_ind = int(np.floor(trial / np.sqrt(n_trials)))
            col_ind = int(trial % np.sqrt(n_trials))
            if trial < n_trials:
                # read in the image
                filename = str(results_io.get_figure_args(filename='turning',
                                                          results_type='session',
                                                          results_name=f'temp_video_files/trial{trial}',
                                                          additional_tags=f'{frame}',
                                                          format='png')['fname'])
                img = mpimg.imread(filename)

                ax[row_ind][col_ind].imshow(img)
                ax[row_ind][col_ind].set(yticks=[], xticks=[])
                for loc in ['right', 'top', 'left', 'bottom']:
                    ax[row_ind][col_ind].spines[loc].set_visible(False)
        plt.tight_layout()

        # save files and append to list
        kwargs = results_io.get_figure_args(filename='turning', results_type='session',
                                            results_name=f'temp_video_files', additional_tags=f'{frame_ind}',
                                            format='png')
        kwargs.update(dpi=150)
        plt.savefig(**kwargs)
        plt.close(fig)
        plot_filenames.append(str(kwargs['fname']))

        # plot only first trial in higher res
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
        for trial, interval in enumerate(trial_intervals):
            frame = camera_frames_all[trial][frame_ind]
            row_ind = int(np.floor(trial / np.sqrt(n_trials)))
            col_ind = int(trial % np.sqrt(n_trials))
            if trial < n_trials:
                # read in the image
                filename = str(results_io.get_figure_args(filename='turning',
                                                          results_type='session',
                                                          results_name=f'temp_video_files/trial{trial}',
                                                          additional_tags=f'{frame}',
                                                          format='png')['fname'])
                img = mpimg.imread(filename)

                ax.imshow(img)
                ax.set(yticks=[], xticks=[])
                for loc in ['right', 'top', 'left', 'bottom']:
                    ax.spines[loc].set_visible(False)

        # save files and append to list
        kwargs = results_io.get_figure_args(filename='turning_first_trial_only', results_type='session',
                                            results_name=f'temp_video_files', additional_tags=f'{frame_ind}',
                                            format='png')
        kwargs.update(dpi=150)
        plt.savefig(**kwargs)
        plt.close(fig)
        plot_filenames_first_trial.append(str(kwargs['fname']))

    # use images to create movie
    video_filename = str(results_io.get_figure_args(filename='turning', results_type='session', format='mp4')['fname'])
    test_frame = cv2.imread(plot_filenames[0])
    height, width, layers = test_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    for filename in plot_filenames:
        video.write(cv2.imread(filename))

    cv2.destroyAllWindows()
    video.release()

    # use images to create movie
    video_filename = str(results_io.get_figure_args(filename='turning_first_trial_only',
                                                    results_type='session',
                                                    format='mp4')['fname'])
    test_frame = cv2.imread(plot_filenames_first_trial[0])
    height, width, layers = test_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    for filename in plot_filenames_first_trial:
        video.write(cv2.imread(filename))

    cv2.destroyAllWindows()
    video.release()

    test = 3