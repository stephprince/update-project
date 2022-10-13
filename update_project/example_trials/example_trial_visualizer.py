import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools

from pathlib import Path
from scipy.stats import sem
from nwbwidgets.analysis.spikes import compute_smoothed_firing_rate

from update_project.results_io import ResultsIO
from update_project.general.plots import get_color_theme
from update_project.example_trials.example_trial_aggregator import ExampleTrialAggregator
from update_project.single_units.psth_visualizer import show_psth_raster

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')
rcparams = mpl.rcParams


class ExampleTrialVisualizer:

    def __init__(self, data, exclusion_criteria=None, align_window=10, align_times=['t_update']):
        self.data = data
        self.colors = get_color_theme()
        self.virtual_track = data[0]['analyzer'].virtual_track
        self.align_times = align_times
        self.align_window = align_window
        self.plot_groups = dict(update_type=[['non_update'], ['switch'], ['stay']],
                                turn_type=[[1, 2]],
                                outcomes=[[0], [1]])
        self.n_example_trials = 10

        self.exclusion_criteria = exclusion_criteria
        self.aggregator = ExampleTrialAggregator()
        self.aggregator.run_aggregation(data, exclusion_criteria=exclusion_criteria, align_window=self.align_window,
                                        align_times=align_times)
        self.results_io = ResultsIO(creator_file=__file__, folder_name=Path().absolute().stem)

    def plot(self):
        for plot_types in list(itertools.product(*self.plot_groups.values())):
            plot_group_dict = {k: v for k, v in zip(self.plot_groups.keys(), plot_types)}
            tags = '_'.join([''.join([k, str(v)]) for k, v in zip(self.plot_groups.keys(), plot_types)])
            self.plot_update_trials(plot_group_dict, tags)

    def plot_update_trials(self, plot_group_dict, tags):
        data = self.aggregator.select_group_aligned_data(filter_dict=plot_group_dict)
        prev_group = ''
        trial_count = 0

        for g_name, g_data in data.groupby(['session_id', 'update_type', 'trial_id']):
            fig, axes = plt.subplots(10, 1, figsize=(8.5, 8.5), layout='constrained')

            # plot LFP and theta
            times = g_data['new_times'].to_numpy()[0]
            axes[0].plot(times, g_data['theta_amplitude'].to_numpy()[0], color='k')
            axes[1].plot(times, g_data['theta_phase'].to_numpy()[0], color='k')

            # plot MUA and rasters
            bin_times = np.linspace(times[0], times[-1], 100)
            fr = np.array([compute_smoothed_firing_rate(x, bin_times, 0.05) for x in g_data['spikes'].to_numpy()])
            axes[2].plot(bin_times, np.nansum(fr, axis=0))

            show_psth_raster(g_data['spikes'].to_list(), ax=axes[3], start=times[0], end=times[-1],
                             group_inds=g_data['max_selectivity_type'].map({'nan':0, 'switch':1, 'stay': 2}).to_list(),
                             labels=g_data['max_selectivity_type'].unique().to_list(),
                             colors=[self.colors[c] for c in g_data['max_selectivity_type'].unique().to_list()])

            # plot decoding data


            # plot behavioral data
            axes[7].plot(g_data['new_times'], g_data['rotational_velocity'], color='k')


            # only plot certain number of trials per session and update type
            if g_name[:2] == prev_group:
                trial_count += 1
                if trial_count >= self.n_example_trials:
                    continue
            else:
                trial_count = 0
                prev_group = g_name[:2]

            # file saving info
            self.results_io.save_fig(fig=fig, axes=axes, filename=f'group_tuning_curves_trial',
                                     additional_tags=f'{tags}_{"_".join(g_name)}', tight_layout=False)
