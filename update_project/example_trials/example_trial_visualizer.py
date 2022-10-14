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
                                correct=[[0], [1]])
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

        if np.size(data):
            data['trial_id'] = data['trial_id'].astype(str)
            bounds = data['bound_values'].unique()
            times = data['new_times'].to_numpy()[0]  # for full sampled data
            bin_times = np.linspace(times[0], times[-1], 500)  # for spiking rasters
            decoding_times = data['times'].to_numpy()[0]  # for decoding data
            feat = data['feature_name'].values[0]
            prev_group = ''
            trial_count = 0

            for g_name, g_data in data.groupby(['session_id', 'trial_id']):
                if len(data['region'].unique()) == 2:
                    # only plot certain number of trials per session and update type
                    if g_name[0] == prev_group:
                        trial_count += 1
                        if trial_count >= self.n_example_trials:
                            continue
                    else:
                        trial_count = 0
                        prev_group = g_name[0]

                    fig, axes = plt.subplots(14, 1, figsize=(15, 20), sharex='col', layout='constrained',
                                             gridspec_kw={'height_ratios': [1, 1, 1, 4, 4, 3, 3, 1, 1, 1, 1, 2, 1, 1]})

                    # plot theta
                    axes[0].plot(times, g_data['theta_amplitude'].to_numpy()[0], color='k', label='theta amplitude')
                    axes[0].set(ylabel='CA1 theta')

                    # plot MUA and rasters
                    ax_locs = dict(CA1=[1, 3, 5, 7, 9], PFC=[2, 4, 6, 8, 10])
                    for r_name, r_data in g_data.groupby('region'):
                        r_data.sort_values('place_field_peak_ind', inplace=True, na_position='first')
                        fr = np.array([compute_smoothed_firing_rate(x, bin_times, 0.01) for x in r_data['spikes'].to_numpy()])
                        axes[ax_locs[r_name][0]].step(bin_times, np.nansum(fr, axis=0), color='k', label=f'{r_name} MUA', where='mid')
                        axes[ax_locs[r_name][0]].fill_between(bin_times, np.nansum(fr, axis=0), color='k', alpha=0.2, step='mid')
                        axes[ax_locs[r_name][0]].set(ylabel=f'{r_name} mua')

                        show_psth_raster(r_data['spikes'].to_list(), ax=axes[ax_locs[r_name][1]], start=times[0], end=times[-1],
                                         group_inds=r_data['max_selectivity_type'].map({np.nan:0, 'switch':1, 'stay': 2}),
                                         colors=[self.colors[str(c)] for c in g_data['max_selectivity_type'].unique()],
                                         show_legend=False)
                        axes[ax_locs[r_name][1]].set(ylabel=f'{r_name} units', xlabel='')

                        # plot decoding data
                        prob_map = r_data['probability'].values[0]
                        n_bins = np.shape(prob_map)[0]
                        feat_limits = self.virtual_track.get_limits(feat)
                        prob_lims = np.linspace(feat_limits[0], feat_limits[1], n_bins)
                        heatmap_times = [decoding_times[0] - (np.diff(decoding_times)[0] / 2),
                                         decoding_times[-1] + (np.diff(decoding_times)[0] / 2)]
                        im = axes[ax_locs[r_name][2]].imshow(prob_map * n_bins, cmap=self.colors['cmap'], aspect='auto',
                                            origin='lower', vmin=0.25, vmax=3, extent=[heatmap_times[0], heatmap_times[-1],
                                                                                       prob_lims[0], prob_lims[-1]], zorder=1)
                        for b in bounds:
                            axes[ax_locs[r_name][2]].axhline(b[0], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)
                            axes[ax_locs[r_name][2]].axhline(b[1], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)


                        axes[ax_locs[r_name][2]].plot(decoding_times, r_data['feature'].values[0], color='w', linestyle='dashed',
                                                      label=f'true {feat}')
                        axes[ax_locs[r_name][2]].plot(decoding_times, r_data['decoding'].values[0], color='w', label=f'predicted {feat}')
                        plt.colorbar(im, ax=axes[ax_locs[r_name][2]], label='prob / chance', pad=0.01, fraction=0.046, location='right')
                        axes[ax_locs[r_name][2]].set(title=r_name, ylabel=r_data['feature_name'].values[0])

                        stay_prob = r_data.query('choice == "initial_stay"')['prob_over_chance'].to_numpy()[0]
                        switch_prob = r_data.query('choice == "switch"')['prob_over_chance'].to_numpy()[0]
                        axes[ax_locs[r_name][3]].plot(decoding_times, stay_prob, color=self.colors['stay'], label='stay')
                        axes[ax_locs[r_name][3]].plot(decoding_times, switch_prob, color=self.colors['switch'], label='switch')
                        axes[ax_locs[r_name][3]].set(ylabel='prob / chance')

                        axes[ax_locs[r_name][4]].plot(decoding_times, r_data['error'].to_numpy()[0], color='r', label='decoding error')
                        axes[ax_locs[r_name][4]].set(ylabel='|True - predicted|')

                    # plot behavioral data
                    speed_threshold = 1000  # TODO - get from bayesian decoder
                    speed_total = abs(g_data['translational_velocity'].values[0]) + abs(g_data['rotational_velocity'].values[0])
                    axes[11].plot(times, speed_total, color='purple', label='movement')
                    axes[11].fill_between(times, speed_threshold, speed_total, where=(speed_total > speed_threshold),
                                          color='purple', alpha=0.2)
                    axes[11].set(ylabel='roll + pitch (au)')
                    axes[11].axhline(speed_threshold, linestyle='dashed', color='k', label='movement threshold')
                    axes[12].plot(times, g_data['rotational_velocity'].values[0], color='m', label='rotation velocity')
                    axes[12].set(ylabel='roll (au)')
                    axes[13].plot(times, g_data['translational_velocity'].values[0], color='b', label='translational velocity')
                    axes[13].set(ylabel='pitch (au)')

                    event_labels = ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made', 'stop_time']
                    colors = dict(start_time='w', t_delay='k', t_update='c', t_delay2='k', t_choice_made='y')
                    event_times = (g_data[event_labels].iloc[0, :] - g_data['t_update'].iloc[0]).to_dict()
                    if plot_group_dict['update_type'][0] == 'non_update':
                        colors.update(t_update='k')
                    for ax in axes:
                        for ind, e in enumerate(event_labels[:-1]):
                            ax.axvspan(event_times[e], event_times[event_labels[ind + 1]], facecolor=colors[e], alpha=0.1, zorder=0)

                    [ax.legend(loc='upper right') for ax in axes]
                    [ax.axvline(0, linestyle='dashed', color='k', linewidth=0.5, zorder=3) for ax in axes]
                    fig.supxlabel('Time around update cue (s)')
                    fig.suptitle(f'Example trial - {tags} - {g_name}')

                    # file saving info
                    self.results_io.save_fig(fig=fig, axes=axes, filename=f'example_trial_{feat}',
                                             additional_tags=f'{tags}_{"_".join(g_name)}', tight_layout=False)
