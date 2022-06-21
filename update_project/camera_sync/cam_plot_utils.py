import cv2
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

def create_track_boundaries():
    # establish track boundaries
    coords = [[2, 1], [2, 245], [10, 260], [10, 280], [8, 280], [0.5, 267], [-0.5, 267], [-8, 280], [-10, 280],
              [-10, 260], [-2, 245], [-2, 1], [2, 1]]
    xs, ys = zip(*coords)

    return xs, ys

def extract_virmen_data(camera_inds_in_df, ind, interval, virmen_df):
    # extract the current position
    current_ind = camera_inds_in_df[ind]
    x_pos = virmen_df.xPos[interval[0]:current_ind]
    y_pos = virmen_df.yPos[interval[0]:current_ind]
    x_pos[:5] = 0
    y_pos[:5] = 5

    # extract view angle
    dx_angle = -math.cos(virmen_df.viewAngle[current_ind] + math.pi / 2)
    dy_angle = math.sin(virmen_df.viewAngle[current_ind] + math.pi / 2)
    view_angle_num = np.round(np.rad2deg(virmen_df.viewAngle[current_ind]))
    view_angle = virmen_df.viewAngle[interval[0]:current_ind]

    # extract time
    time = virmen_df.time[interval[0]:current_ind]
    trial_start_time = virmen_df.time[interval[0]]
    trial_end_time = virmen_df.time[interval[-1]]
    trial_duration = (trial_end_time - trial_start_time) * 60 * 60 * 24 # to convert to seconds
    time_elapsed = (time - trial_start_time) * 60 * 60 * 24  # to convert to seconds

    # extract lick data (if 1, lick detected)
    licks = np.diff(virmen_df.numLicks[interval[0]:current_ind], prepend=virmen_df.numLicks[interval[0]])
    lick_events = time_elapsed[licks == 1]

    # extract velocity
    rot_veloc = virmen_df.rotVeloc[interval[0]:current_ind]
    trans_veloc = virmen_df.transVeloc[interval[0]:current_ind]

    return {"x_pos": x_pos, "y_pos": y_pos, "lick_events": lick_events,
            "dx_angle": dx_angle, "dy_angle": dy_angle, "view_angle": view_angle, "view_angle_num": view_angle_num,
            "time_elapsed": time_elapsed, "trial_duration": trial_duration,
            "rot_veloc": rot_veloc, "trans_veloc": trans_veloc,
            }

def plot_update_cam_all(camera_inds_in_df, output_dir, frame_no, ind, interval, virmen_df, itrial):
    # read in the image
    frame_filename = Path(f"{output_dir}/temp/trial{itrial}/frame{frame_no}.png")
    img = mpimg.imread(frame_filename)

    # extract relevant data
    vr_data = extract_virmen_data(camera_inds_in_df, ind, interval, virmen_df)
    track_bounds_xs, track_bounds_ys = create_track_boundaries()
    fps = np.shape(camera_inds_in_df)[0]/vr_data["trial_duration"]

    # plot the data
    mosaic = """
            BBAAAACC
            BBAAAADD
            BBEEFFGG
            """
    #plt.style.use('dark_background')
    lwidth = 3
    ax_dict = plt.figure(constrained_layout=True, figsize=(20, 10)).subplot_mosaic(mosaic)

    ax_dict["A"].imshow(img)
    ax_dict["A"].set_xticks([])
    ax_dict["A"].set_yticks([])

    ax_dict["B"].plot(vr_data["x_pos"], vr_data["y_pos"], color="steelblue", linewidth=lwidth)
    ax_dict["B"].plot(track_bounds_xs, track_bounds_ys, color='black')
    ax_dict["B"].set_title('Track Position', fontsize=20)
    ax_dict["B"].set_ylim(0, 300)
    ax_dict["B"].set_xlim(10.1, -10.1)
    ax_dict["B"].axis('off')

    ax_dict["C"].plot(vr_data["time_elapsed"], vr_data["rot_veloc"], color='slateblue', linewidth=lwidth)
    ax_dict["C"].set_xlabel('Time (s)', fontsize=14)
    ax_dict["C"].set_ylabel('Speed (au)', fontsize=14)
    ax_dict["C"].set_title('Rotational velocity', fontsize=20)
    ax_dict["C"].set_xlim(0, vr_data["trial_duration"])
    ax_dict["C"].set_ylim(-2.75, 2.75)

    ax_dict["D"].plot(vr_data["time_elapsed"], vr_data["trans_veloc"], color='darkviolet', linewidth=lwidth)
    ax_dict["D"].set_xlabel('Time (s)', fontsize=14)
    ax_dict["D"].set_ylabel('Speed (au)', fontsize=14)
    ax_dict["D"].set_title('Translational velocity', fontsize=20)
    ax_dict["D"].set_xlim(0, vr_data["trial_duration"])
    ax_dict["D"].set_ylim(0, 60)

    ax_dict["E"].eventplot(vr_data["lick_events"], color="hotpink", linewidths=2)
    ax_dict["E"].set_title('Licking activity', fontsize=20)
    ax_dict["E"].set_xlim(0, vr_data["trial_duration"])
    ax_dict["E"].axis('off')

    ax_dict["F"].arrow(0, 0, vr_data["dx_angle"], vr_data["dy_angle"], head_width=0.1, head_length=0.1, linewidth=lwidth,
                       color='seagreen')
    ax_dict["F"].set_title('View Angle', fontsize=20)
    ax_dict["F"].set_xlim(-1.25, 1.25)
    ax_dict["F"].set_ylim(0, 1.25)
    ax_dict["F"].axis('off')
    ax_dict["F"].text(0.85, 0.85, f"{vr_data['view_angle_num']}\N{DEGREE SIGN}", fontsize=14)

    ax_dict["G"].plot(vr_data["time_elapsed"], vr_data["view_angle"], color='seagreen', linewidth=lwidth)
    ax_dict["G"].set_title('View Angle', fontsize=20)
    ax_dict["G"].set_xlabel('Time (s)', fontsize=14)
    ax_dict["G"].set_ylabel('View Angle', fontsize=14)
    ax_dict["G"].set_ylim(-1.5, 1.5)
    ax_dict["G"].set_xlim(0, vr_data["trial_duration"])

    # save the entire plot for each camera frame
    plot_filename = Path(f"{output_dir}/results/camera/trial{itrial}/trial{itrial}_frame{frame_no}_all.png")
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename, fps

def plot_update_cam_simple(camera_inds_in_df, output_dir, frame_no, ind, interval, virmen_df, itrial):
    # font sizes
    axes_size = 18
    title_size = 22

    # read in the image
    frame_filename = Path(f"{output_dir}/temp/trial{itrial}/frame{frame_no}.png")
    img = mpimg.imread(frame_filename)

    # extract relevant data
    vr_data = extract_virmen_data(camera_inds_in_df, ind, interval, virmen_df)
    track_bounds_xs, track_bounds_ys = create_track_boundaries()

    # plot the data
    mosaic = """
            BAAAAA
            BAAAAA
            BAAAAA
            BAAAAA
            BAAAAA
            BCCCCC
            """
    lwidth = 3
    ax_dict = plt.figure(constrained_layout=True, figsize=(20, 14)).subplot_mosaic(mosaic)

    ax_dict["A"].imshow(img)
    ax_dict["A"].set_xticks([])
    ax_dict["A"].set_yticks([])

    ax_dict["B"].plot(vr_data["x_pos"], vr_data["y_pos"], color="steelblue", linewidth=lwidth)
    ax_dict["B"].plot(track_bounds_xs, track_bounds_ys, color='black')
    ax_dict["B"].set_title('Track Position', fontsize=title_size)
    ax_dict["B"].set_ylim(0, 300)
    ax_dict["B"].set_xlim(10.1, -10.1)
    ax_dict["B"].axis('off')

    ax_dict["C"].eventplot(vr_data["lick_events"], color="hotpink", linewidths=lwidth)
    ax_dict["C"].set_ylabel('Lick events', fontsize=axes_size)
    ax_dict["C"].set_xlabel('Time elapsed (s)', fontsize=axes_size)
    ax_dict["C"].set_xlim(0, vr_data["trial_duration"])

    # save the entire plot for each camera frame
    plot_filename = Path(f"{output_dir}/results/camera/trial{itrial}/trial{itrial}_frame{frame_no}_simple.png")
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

def plot_update_cam_angle_and_veloc(camera_inds_in_df, output_dir, frame_no, ind, interval, virmen_df, itrial):
    # font sizes
    axes_size = 18
    title_size = 22

    # read in the image
    frame_filename = Path(f"{output_dir}/temp/trial{itrial}/frame{frame_no}.png")
    img = mpimg.imread(frame_filename)

    # extract relevant data
    vr_data = extract_virmen_data(camera_inds_in_df, ind, interval, virmen_df)
    track_bounds_xs, track_bounds_ys = create_track_boundaries()

    # plot the data
    mosaic = """
                AAAAAABB
                AAAAAACC
                """
    lwidth = 3
    ax_dict = plt.figure(constrained_layout=True, figsize=(20, 10)).subplot_mosaic(mosaic)

    ax_dict["A"].imshow(img)
    ax_dict["A"].set_xticks([])
    ax_dict["A"].set_yticks([])

    ax_dict["B"].plot(vr_data["time_elapsed"], vr_data["rot_veloc"], color='slateblue', linewidth=lwidth)
    ax_dict["B"].set_xlabel('Time (s)', fontsize=axes_size)
    ax_dict["B"].set_ylabel('Speed (au)', fontsize=axes_size)
    ax_dict["B"].set_title('Rotational velocity', fontsize=title_size)
    ax_dict["B"].set_xlim(0, vr_data["trial_duration"])
    ax_dict["B"].set_ylim(-2.75, 2.75)

    ax_dict["C"].plot(vr_data["time_elapsed"], vr_data["view_angle"], color='darkviolet', linewidth=lwidth)
    ax_dict["C"].set_title('View Angle', fontsize=title_size)
    ax_dict["C"].set_xlabel('Time (s)', fontsize=axes_size)
    ax_dict["C"].set_ylabel('View Angle', fontsize=axes_size)
    ax_dict["C"].set_ylim(-1.5, 1.5)
    ax_dict["C"].set_xlim(0, vr_data["trial_duration"])

    # save the entire plot for each camera frame
    plot_filename = Path(f"{output_dir}/results/camera/trial{itrial}/trial{itrial}_frame{frame_no}_angle_veloc.png")
    plt.savefig(plot_filename)
    plt.close()

    return plot_filename

def write_camera_video(out_dir, plot_filenames, vid_filename, fps=5.0):
    video_filename = Path(f"{out_dir}/{vid_filename}.mp4")
    frame = cv2.imread(str(plot_filenames[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))
    for filename in plot_filenames:
        video.write(cv2.imread(str(filename)))

    cv2.destroyAllWindows()
    video.release()

def write_camera_gif(out_dir, plot_filenames, vid_filename):
    # build gif
    gif_filename = Path(f"{out_dir}/trial{itrial}_{vid_filename}.gif")
    with imageio.get_writer(gif_filename, mode='I') as writer:
        for filename in plot_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)