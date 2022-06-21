import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from pynwb import NWBHDF5IO
from scipy.signal import resample

from update_project.general.plots import plot_distributions
from update_project.session_loader import SessionLoader
from update_project.results_io import ResultsIO


# define main plotting function
def plot_velocity_distributions(velocity_df, thresholds=[], group=None, show_median=True):
    # plot the distributions
    nrows, ncols = (3, 3)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))

    title = 'Rotational velocity distribution'
    xlabel = 'Labview speed'
    plot_distributions(velocity_df, axes=axes, column_name='rotational', group=group, row_ids=[0, 1, 2],
                       col_ids=[0, 0, 0], xlabel=xlabel, title=title, stripplot=False, show_median=show_median)
    ylims = axes[1][0].get_ylim()
    for t in thresholds:
        axes[1][0].plot([t, t], ylims, linestyle='dashed', color=[0, 0, 0, 0.5], label=f'{t} cutoff')
    axes[1][0].set_ylim(ylims)
    axes[1][0].legend()

    title = 'Translational velocity distribution'
    xlabel = 'Labview speed'
    plot_distributions(velocity_df, axes=axes, column_name='translational', group=group, row_ids=[0, 1, 2],
                       col_ids=[1, 1, 1], xlabel=xlabel, title=title, stripplot=False, show_median=show_median)
    ylims = axes[1][1].get_ylim()
    for t in thresholds:
        axes[1][1].plot([t, t], ylims, linestyle='dashed', color=[0, 0, 0, 0.5], label=f'{t} cutoff')
    axes[1][1].set_ylim(ylims)
    axes[1][1].legend()

    title = 'Summed velocity distributions'
    xlabel = 'Labview speed'
    plot_distributions(velocity_df, axes=axes, column_name='summed', group=group, row_ids=[0, 1, 2],
                       col_ids=[2, 2, 2], xlabel=xlabel, title=title, stripplot=False, show_median=show_median)
    ylims = axes[1][2].get_ylim()
    for t in thresholds:
        axes[1][2].plot([t, t], ylims, linestyle='dashed', color=[0, 0, 0, 0.5], label=f'{t} cutoff')
    axes[1][2].set_ylim(ylims)
    axes[1][2].legend()


# setup sessions
animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = []  # 210913
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

downsample_factor = 20
thresholds = [1000, 2500, 5000]
plot_session_data = False

group_data_list = []
for name in session_names:
    print(f'Getting data for session {name}')

    # load nwbfile
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()
    base_path = Path().absolute().parent / 'results'
    results_io = ResultsIO(creator_file=__file__, base_path=base_path, session_id=session_db.get_session_id(name),
                           folder_name='movement-threshold')

    # get velocity data
    rot_velocity = np.abs(nwbfile.acquisition['rotational_velocity'].data[:])
    trans_velocity = np.abs(nwbfile.acquisition['translational_velocity'].data[:])
    rot_velocity_resampled = resample(rot_velocity, int(len(rot_velocity) / downsample_factor))
    trans_velocity_resampled = resample(trans_velocity, int(len(trans_velocity) / downsample_factor))
    velocity_dict = dict(rotational=rot_velocity_resampled,
                         translational=trans_velocity_resampled,
                         summed=(rot_velocity_resampled + trans_velocity_resampled),
                         session_id=session_db.get_session_id(name),
                         animal=name[1])
    velocity_df = pd.DataFrame(velocity_dict)
    group_data_list.append(velocity_df)

    if plot_session_data:
        plot_velocity_distributions(velocity_df, thresholds=thresholds)
        plt.suptitle(f'Velocity distributions  - {name}', fontsize=14)
        kwargs = results_io.get_figure_args(filename=f'velocity_distributions', results_type='session', format='pdf')
        plt.savefig(**kwargs)
        plt.close()

# plot all animal distribution
group_df = pd.concat(group_data_list, axis=0, ignore_index=True)
group_data_list = []  # clear variable to save memory

# plot the distributions
plot_velocity_distributions(group_df, thresholds=thresholds, group='animal', show_median=False)
plt.suptitle(f'Velocity distributions  - all sessions', fontsize=14)
kwargs = results_io.get_figure_args(filename=f'velocity_distributions', additional_tags=g, format='pdf')
plt.savefig(**kwargs)
plt.close()
