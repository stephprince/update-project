import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from pynwb import NWBHDF5IO

from nwbwidgets.utils.timeseries import bisect_timeseries_by_times, align_by_trials

# load the nwb files
nwb_filename = Path('../../data//test_behavior.nwb')

io = NWBHDF5IO(nwb_filename,'r')
nwbfile = io.read()

# compile behavioral data
df_trials = nwbfile.trials.to_dataframe()  # get trial dataframe
df_trials = df_trials[df_trials['maze_id'] == 4]  # filter dataframe to only have trials from the delay maze (maze id 4)

### plot behavioral performance ###

# make a line plot of performance over time
window = 30  # number of trials to use to calculate the moving average
df_trials['prop_correct'] = df_trials.correct.rolling(window, min_periods=1).mean()  # this column will be proportion correct trials calculated using a moving average

# make violin plot of performance across trial types
update_types = []
trials_by_update_type = []
for key, group in df_trials.groupby('update_type'):
    update_types.append(key)
    trials_by_update_type.append(group['prop_correct'])

fig_0, ax_0 = plt.subplots(nrows=1, ncols=1)
violin_parts = ax_0.violinplot(trials_by_update_type, update_types, showmeans=True)

# plot horizontal dashed line across the middle
horizontal_dashed_line_x = np.linspace(start=0.5, stop=3.5, num=100)
horizontal_dashed_line_y = np.full(shape=(100,), fill_value=0.5)
ax_0.plot(horizontal_dashed_line_x, horizontal_dashed_line_y, color='black', linestyle='dashed', alpha=0.5)

# double check if the numbers are mapped to the right labels
update_types_dict = {1.0: "Update",
                     2.0: "Non-update",
                     3.0: "Visual-guided"}
ax_0.set_title("Behavioral Performance Violin Plot")
ax_0.set_xlabel("Trial Type")
ax_0.set_ylabel("Proportion Correct")
ax_0.set_xticks([1.0, 2.0, 3.0])
ax_0.set_xticklabels([update_types_dict[1.0], update_types_dict[2.0], update_types_dict[3.0]])
ax_0.set_xlim([0.5, 3.5])
ax_0.set_ylim([0.0, 1.0])

# customize color of violins
colors = ['red', 'green', 'blue']
for i, pc in enumerate(violin_parts['bodies']):
    pc.set_color(colors[i])
violin_parts['cmeans'].set_color(colors)
violin_parts['cbars'].set_color(colors)
violin_parts['cmaxes'].set_color(colors)
violin_parts['cmins'].set_color(colors)

### plot trajectories ###

# make line plot of position throughout the trial
view_angle = nwbfile.processing['behavior']['view_angle'].spatial_series['view_angle']
view_angle_by_trials = align_by_trials(view_angle)

# make a line plot/average of view angles around the update period
window = 5  # look 5 seconds before and 5 seconds after the update occurred
start_times = df_trials['t_update'].dropna() - window
time = nwbfile.processing['behavior']['time']
view_angle_around_update = bisect_timeseries_by_times(view_angle, start_times, window*2)
time_around_update = bisect_timeseries_by_times(time, start_times, window*2)

# fig = plt.figure(constrained_layout=True, figsize=(20, 10))
# ax = fig.add_subplot(111)
fig_1, ax_1 = plt.subplots(nrows=1, ncols=1)
for idx, trace in enumerate(view_angle_around_update):
    print("idx = {}, len = {}".format(idx, len(trace))) # the length of the traces are not uniform - ERROR
    time_from_aligned = time_around_update[idx] - time_around_update[idx][0] - 5
    if trace[-1] > 0.0:
        ax_1.plot(time_from_aligned, trace, color='red', alpha=0.4) # alpha controls transparency, 1.0 = opaque
    else:
        ax_1.plot(time_from_aligned, trace, color='blue', alpha=0.4)

# plot vertical dashed line down the middle
vertical_dashed_line_x = np.zeros(100)
vertical_dashed_line_y = np.linspace(start=-0.8, stop=0.8, num=100)
ax_1.plot(vertical_dashed_line_x, vertical_dashed_line_y, color='black', linestyle='dashed', alpha=0.5)

ax_1.set_title("View Angle Line Plot")
ax_1.set_xlabel("Position Around Update") # units? is this time? where does update happen?
ax_1.set_ylabel("View Angle (rad)")
ax_1.set_xlim([-5.0, 5.0])
ax_1.set_ylim([-0.8, 0.8])

plt.show()