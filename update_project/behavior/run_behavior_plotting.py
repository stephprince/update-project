import matplotlib.pyplot as plt

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


### plot trajectories ###

# make line plot of position throughout the trial
view_angle = nwbfile.processing['behavior']['view_angle'].spatial_series['view_angle']
view_angle_by_trials = align_by_trials(view_angle)

# make a line plot/average of view angles around the update period
window = 5  # look 5 seconds before and 5 seconds after the update occurred
start_times = df_trials['t_update'].dropna()-window
time = nwbfile.processing['behavior']['time']
view_angle_around_update = bisect_timeseries_by_times(view_angle, start_times, window*2)
time_around_update = bisect_timeseries_by_times(time, start_times, window*2)

fig = plt.figure(constrained_layout=True, figsize=(20, 10))
ax = fig.add_subplot(111)
for idx, trace in enumerate(view_angle_around_update):
    time_from_aligned = time_around_update[idx]-time_around_update[idx][0]
    ax.plot(time_from_aligned, trace)

plt.show()

