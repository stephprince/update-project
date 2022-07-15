import matplotlib.pyplot as plt
import numpy as np

from bisect import bisect
from pynwb import NWBHDF5IO
from track_linearization import make_track_graph, plot_track_graph, get_linearized_position, plot_graph_as_1D

from update_project.session_loader import SessionLoader
from update_project.results_io import ResultsIO

# setup sessions
animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = [210913]  # 210913
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()
plot_colors = 'y'  # 'x' is the other option, which metric to base plotting colors off of

for name in session_names:
    # load nwbfile
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()
    results_io = ResultsIO(creator_file=__file__, session_id=session_db.get_session_id(name), folder_name='position-linearization',)

    # organize position data by trial
    n_trials = 5
    spatial_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
    start_times = nwbfile.trials.start_time[:]
    stop_times = nwbfile.trials.stop_time[:]
    position_by_trial = []
    for start, stop in zip(start_times, stop_times):
        idx_start = bisect(spatial_series.timestamps, start)
        idx_stop = bisect(spatial_series.timestamps, stop, lo=idx_start)
        position_by_trial.append(spatial_series.data[idx_start:idx_stop, :])
    position = np.vstack(position_by_trial[:n_trials])  # first 5 trials

    if plot_colors == 'x':
        home_arm_mask = np.logical_and(position[:, 0] <= 1, position[:, 0] >= -1)
        left_arm_mask = position[:, 0] < -1
        right_arm_mask = position[:, 0] > 1  # so the colors are going to indicate the change in X
    elif plot_colors == 'y':
        home_arm_mask = position[:, 1] < 255
        left_arm_mask = np.logical_and(position[:, 1] > 255, position[:, 0] <= -1)
        right_arm_mask = np.logical_and(position[:, 1] > 255, position[:, 0] >= 1)

    # make track graph and visualize with position data
    node_positions = [(0, 0),       # start of home arm
                      (0, 255),     # end of home arm
                      (-30, 285),   # left arm
                      (30, 285)]    # right arm
    edges = [(0, 1),
             (1, 2),
             (1, 3)]
    track_graph = make_track_graph(node_positions, edges)

    fig, ax = plt.subplots()
    plot_track_graph(track_graph, ax=ax, draw_edge_labels=True)
    plt.scatter(position[home_arm_mask, 0], position[home_arm_mask, 1], s=10, zorder=11)
    plt.scatter(position[left_arm_mask, 0], position[left_arm_mask, 1], s=10, zorder=11)
    plt.scatter(position[right_arm_mask, 0], position[right_arm_mask, 1], s=10, zorder=11)
    plt.suptitle(f'Track graph - first {n_trials} trials  - {name}', fontsize=14)
    kwargs = results_io.get_figure_args(filename=f'track-graph', results_type='session', format='pdf')
    plt.savefig(**kwargs)
    plt.close()

    # linearize y_position and visualize with position data
    position_df = get_linearized_position(position=position, track_graph=track_graph)

    fig, ax = plt.subplots()
    plot_graph_as_1D(track_graph, ax=ax, axis="y", other_axis_start=position_df.index.max() + 1)
    ax.scatter(position_df.index[home_arm_mask], position_df.linear_position[home_arm_mask], s=5, zorder=2, clip_on=False)
    ax.scatter(position_df.index[left_arm_mask], position_df.linear_position[left_arm_mask], s=5, zorder=2, clip_on=False)
    ax.scatter(position_df.index[right_arm_mask], position_df.linear_position[right_arm_mask], s=5, zorder=2, clip_on=False)
    plt.suptitle(f'Linearized position - first {n_trials} trials  - {name}', fontsize=14)
    kwargs = results_io.get_figure_args(filename=f'linearized-position', results_type='session', format='pdf')
    plt.savefig(**kwargs)
    plt.close()
