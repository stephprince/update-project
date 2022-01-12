# import libraries
import cv2
import glob
import numpy as np
import pandas as pd
import warnings

from pathlib import Path

from cam_utils import convert_mat_file_to_dict, get_trial_intervals
from cam_plot_utils import plot_update_cam_all, plot_update_cam_simple, plot_update_cam_angle_and_veloc, write_camera_video

# set inputs
animal = 29
date = 211027
session = 1
trial_type = "update"

save_new_video_images = 0 # set to 1 to overwrite existing video frame temp files

base_path = "y:/" #"/Volumes/labs/"
video_dir = Path(f"{base_path}singer/CameraData/UpdateTask/S{animal}_{date}/")
virmen_filename = Path(f"{base_path}singer/Virmen Logs/UpdateTask/S{animal}_{date}_{session}/virmenDataRaw.mat")
output_dir = Path(f"{base_path}singer/CameraData/UpdateTask/S{animal}_{date}/{trial_type}/")

# import virmen data
if virmen_filename.is_file():
    virmen_data = convert_mat_file_to_dict(virmen_filename)
    virmen_df = pd.DataFrame(virmen_data['virmenData']['data'], columns=virmen_data['virmenData']['dataHeaders'])
else:
    warnings.warn('File can not be found')

# import video frames
video_file = video_dir / "*.mp4"
video_file = str(video_file)
video_file = glob.glob(video_file)[session-1]
cap = cv2.VideoCapture(video_file)

# check that the virmen data and the camera data have the same number of frames so that we can properly align them
video_frame_count= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
virmen_frame_count = sum(virmen_df.cameraTrigger)

assert video_frame_count == virmen_frame_count

# get trial intervals
camera_trig = virmen_df.index[virmen_df.cameraTrigger == 1].to_list()
trial_intervals = get_trial_intervals(virmen_df, camera_trig, trial_type)

# save new video images
if save_new_video_images:
    # save images for each frame from those trials
    for i, interval in enumerate(trial_intervals):
        print(f"Saving frames for trial {i}")
        # create folder
        Path(f"{output_dir}/temp/trial{i}/").mkdir(parents=True, exist_ok=True)
        img_dir = Path(f"{output_dir}/temp/trial{i}/")

        camera_frames = np.where(np.logical_and(camera_trig > interval[0], camera_trig < interval[1]))[0]
        frame_no = camera_frames[0]

        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
        while frame_no <= camera_frames[-1]:
            ret, frame = cap.read()
            name = Path(f"{img_dir}/frame{frame_no}.png")
            name = str(name)
            cv2.imwrite(name, frame)
            frame_no += 1

    cap.release()
    cv2.destroyAllWindows()

# use images to create movie
for i, interval in enumerate(trial_intervals):
    #get relevant trial data
    trial_mask = np.logical_and(camera_trig > interval[0], camera_trig < interval[1])
    camera_frames = np.where(trial_mask)[0]
    camera_inds_in_df = np.array(camera_trig)[trial_mask]

    # make plot for each camera frame
    print(f"Making gif for trial {i}")
    Path(f"{output_dir}/results/camera/trial{i}/").mkdir(parents=True, exist_ok=True)
    out_dir = Path(f"{output_dir}/results/camera/trial{i}/")
    plot_filenames_all = []
    plot_filenames_simple = []
    plot_filenames_angle_veloc = []
    for ind, frame_no in enumerate(camera_frames):
        filename, fps = plot_update_cam_all(camera_inds_in_df, output_dir, frame_no, ind, interval, virmen_df, i)
        plot_filenames_all.append(filename)

        filename = plot_update_cam_simple(camera_inds_in_df, output_dir, frame_no, ind, interval, virmen_df, i)
        plot_filenames_simple.append(filename)

        filename = plot_update_cam_angle_and_veloc(camera_inds_in_df, output_dir, frame_no, ind, interval, virmen_df, i)
        plot_filenames_angle_veloc.append(filename)

    # make videos for each plot type
    write_camera_video(out_dir, fps, plot_filenames_all, 'all')
    write_camera_video(out_dir, fps, plot_filenames_simple, 'simple')
    write_camera_video(out_dir, fps, plot_filenames_angle_veloc, 'angle_veloc')
