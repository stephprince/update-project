import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import seaborn as sns

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
            feat = data['feature_name'].values[0]
            prev_group = ''
            trial_count = 0

            for g_name, g_data in data.groupby(['session_id', 'trial_id']):
                if len(g_data['region'].unique()) == 2:
                    if g_name[0] == prev_group:
                        trial_count += 1
                        if trial_count >= self.n_example_trials:
                            continue  # only plot certain number of trials per session and update type
                    else:
                        trial_count = 0
                        prev_group = g_name[0]

                    fig = plt.figure(figsize=(15, 20), layout='constrained')
                    sfigs = fig.subfigures(1, 2, width_ratios=[4, 1])

                    # plot all neural data/behavioral metrics over time
                    axes = sfigs[0].subplots(14, 1, sharex='col',
                                             gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 4, 4, 3, 3, 1, 1, 2, 1, 1]})
                    self._plot_single_trial_timeseries(fig=sfigs[0], axes=axes, data=g_data, plot_group=plot_group_dict)
                    fig.suptitle(f'Example trial - {tags} - {g_name[0]}')

                    # plot general info accompanying trial timeseries
                    axes2 = sfigs[1].subplots(13, 1, gridspec_kw={'height_ratios': [1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2]})
                    self._plot_session_info(fig=sfigs[1], axes=axes2, data=data.query(f'session_id == "{g_name[0]}"'),
                                            g_data=g_data, session_id=g_name[0])
                    fig.suptitle(f'Session averages - {g_name}')

                    # file saving info
                    self.results_io.save_fig(fig=fig, axes=axes, filename=f'example_trial_{feat}',
                                             additional_tags=f'{tags}_{"_".join(g_name)}', tight_layout=False)

    def _plot_session_info(self, fig, axes, data, g_data, session_id):
        feat = g_data['feature_name'].values[0]
        locations = self.virtual_track.cue_end_locations.get(feat, dict())
        ax_adjust = dict(CA1=0, PFC=1)

        # plot theta modulation for this trial
        theta_phase_data = self.aggregator.single_unit_agg.calc_theta_phase_data(g_data, by_trial=True)
        for t_name, t_data in theta_phase_data.groupby(['region', 'new_times', 'max_selectivity_type'], dropna=False):
            axes[0].plot(t_data['phase_mid'] / np.pi, t_data['theta_amplitude'], color='k', label='theta')
            axes[0].fill_between((t_data['phase_mid'] / np.pi),
                                  t_data['theta_amplitude'] + t_data['theta_amplitude_err'],
                                  t_data['theta_amplitude'] -t_data['theta_amplitude_err'],
                                  color='k', alpha=0.2)

            lstyles = ['dashed', 'solid']
            lstyle = lstyles[np.where(np.array(['pre-update', 'post-update']) == t_name[1])[0][0]]
            axes[1 + ax_adjust[t_name[0]]].plot((t_data['phase_mid'] / np.pi), t_data['spike_counts'],
                                                color=self.colors[str(t_name[2])], linestyle=lstyle)
            axes[1 + ax_adjust[t_name[0]]].fill_between((t_data['phase_mid'] / np.pi),
                                                        t_data['spike_counts'] + t_data['spike_counts_err'],
                                                        t_data['spike_counts'] - t_data['spike_counts_err'],
                                                        color=self.colors[str(t_name[2])], alpha=0.2)
            axes[1 + ax_adjust[t_name[0]]].xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g $\pi$'))
            axes[1 + ax_adjust[t_name[0]]].xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=1.0))
            axes[1 + ax_adjust[t_name[0]]].relim()
            axes[1 + ax_adjust[t_name[0]]].set(ylabel='mean spikes (trial only)', xlabel='theta_phase', title=t_name[0])
            axes[1 + ax_adjust[t_name[0]]].legend()

        # plot firing rates
        psth_data = self.aggregator.single_unit_agg.get_aligned_psth(data)
        psth_times = psth_data['psth_times'].to_numpy()[0]
        for s_name, s_data in psth_data.groupby(['region', 'max_selectivity_type'], dropna=False):
            psth_mat_all = np.vstack(s_data['psth_mean'])
            psth_mat_all = psth_mat_all[~np.isnan(psth_mat_all).any(axis=1), :]
            psth_mean = np.nanmean(psth_mat_all, axis=0)
            psth_sem = sem(psth_mat_all, axis=0, nan_policy='omit')

            # average + heatmap of all cells
            axes[3 + ax_adjust[s_name[0]]].plot(psth_times, np.nanmean(psth_mat_all, axis=0),
                                              color=self.colors[str(s_name[1])])
            axes[3 + ax_adjust[s_name[0]]].fill_between(psth_times, psth_mean + psth_sem, psth_mean - psth_sem,
                                                      color=self.colors[str(s_name[1])], alpha=0.3)
            axes[3 + ax_adjust[s_name[0]]].set(ylabel='mean fr')

        # plot tuning curves for this session
        tuning_data = self.aggregator.single_unit_agg.group_tuning_curves.query(f'session_id == "{session_id}"')
        for r_name, r_data in tuning_data.groupby('region'):
            r_data.sort_values('place_field_peak_ind', na_position='first', inplace=True)
            curve = np.stack(r_data['tuning_curves'].to_numpy())
            tuning_curve_scaled = curve / np.max(curve, axis=1)[:, None]

            y_limits = [0, np.shape(tuning_curve_scaled)[0]]
            x_limits = [np.round(np.min(r_data['tuning_bins'].to_numpy()[0]), 2),
                        np.round(np.max(r_data['tuning_bins'].to_numpy()[0]), 2)]
            im = axes[5 + ax_adjust[r_name]].imshow(tuning_curve_scaled, aspect='auto',
                                                    cmap=self.colors['cmap_r'], origin='lower', vmin=0.1,
                                                    vmax=0.9,
                                                    extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]])
            axes[5 + ax_adjust[r_name]].set(title=r_name)

            for key, value in locations.items():
                axes[5 + ax_adjust[r_name]].axvline(value, linestyle='dashed', color='k', alpha=0.5)
            plt.colorbar(im, ax=axes[5 + ax_adjust[r_name]], pad=0.04, location='right', fraction=0.046,
                         label='normalized firing rate')

        # plot confusion matrix for this session
        decoding_data = self.aggregator.decoder_agg.group_df.query(f'session_id == "{session_id}"')
        decoding_data['region'] = decoding_data['region'].apply(lambda x:
                                                                ast.literal_eval(x)[0] if isinstance(x, str) else x[0])
        bins = g_data['bins'].values[0]
        limits = [np.round(np.min(bins), 2), np.round(np.max(bins), 2)]
        for r_name, r_data in decoding_data.groupby('region'):
            # summary_df = r_data['summary_df'].values[0].loc[data['timestamps'].values[0][0]:data['timestamps'].values[0][-1]]
            # confusion_mat = self.aggregator.decoder_agg._get_confusion_matrix(summary_df, bins) * len(bins)
            confusion_mat = r_data['confusion_matrix'].to_numpy()[0] * len(bins)
            im = axes[7 + ax_adjust[r_name]].imshow(confusion_mat, cmap=self.colors['cmap'], origin='lower',
                                                    vmin=0.5, vmax=3, extent=[limits[0], limits[1], limits[0], limits[1]])
            axes[7 + ax_adjust[r_name]].set(title=r_name, xlabel=f'true {feat}', ylabel=f'decoded {feat}')

            # plot annotation lines
            for key, value in locations.items():
                axes[7 + ax_adjust[r_name]].axhline(value, linestyle='dashed', color='k', alpha=0.5)
                axes[7 + ax_adjust[r_name]].axvline(value, linestyle='dashed', color='k', alpha=0.5)
            plt.colorbar(im, ax=axes[7 + ax_adjust[r_name]], pad=0.04, location='right', fraction=0.046, label='prob / chance')

        # plot decoding around update average
        for r_name, r_data in data.groupby('region'):
            prob_map = (r_data
                .groupby('choice')
                .apply(lambda x: np.nanmean(np.stack(x['probability']), axis=0))
                .values[0])  # only get one of the switch/stay averages bc. they should be the ame
            goal_mean = (r_data
                         .groupby('choice')
                         .apply(lambda x: np.nanmean(np.stack(x['prob_over_chance']), axis=0)))
            goal_err = (r_data
                         .groupby('choice')
                         .apply(lambda x: pd.Series(sem(np.stack(x['prob_over_chance']), nan_policy='omit')))
                        .squeeze(axis=0))
            prob_lims = np.linspace(limits[0], limits[-1], np.shape(prob_map)[0])
            times = r_data['times'].to_numpy()[0]

            im_prob = axes[9 + ax_adjust[r_name]].imshow(prob_map * np.shape(prob_map)[0], cmap=self.colors['cmap'], aspect='auto',
                                          origin='lower', vmin=0.6, vmax=2.8,
                                          extent=[times[0], times[-1], prob_lims[0], prob_lims[-1]])
            axes[9 + ax_adjust[r_name]].axvline(0, color='k', linestyle='dashed', alpha=0.5)
            axes[9 + ax_adjust[r_name]].set(title=r_name, ylabel=feat, xlabel='Time aorund update')

            axes[11 + ax_adjust[r_name]].plot(times, goal_mean['switch'], color=self.colors['switch'], label='switch')
            axes[11 + ax_adjust[r_name]].fill_between(times, goal_mean['switch'] + goal_err.loc['switch'].to_numpy(),
                                 goal_mean['switch'] - goal_err.loc['switch'].to_numpy(),  color=self.colors['switch'],
                                 alpha=0.2)
            axes[11 + ax_adjust[r_name]].plot(times, goal_mean['initial_stay'], color=self.colors['initial_stay'], label='initial')
            axes[11 + ax_adjust[r_name]].fill_between(times, goal_mean['initial_stay'] + goal_err.loc['initial_stay'].to_numpy(),
                                 goal_mean['initial_stay'] - goal_err.loc['initial_stay'].to_numpy(),
                                 color=self.colors['initial_stay'], alpha=0.2)
            axes[11 + ax_adjust[r_name]].axhline(1, linestyle='dashed', color='k', alpha=0.5)
            axes[11 + ax_adjust[r_name]].axvline(0, color='k', linestyle='dashed', alpha=0.5)
            axes[11 + ax_adjust[r_name]].set(title=r_name, ylabel=feat, xlabel='Time aorund update')

            plt.colorbar(im_prob, ax=axes[9 + ax_adjust[r_name]], label='prob / chance', pad=0.01, fraction=0.046, location='right')
            for key, value in locations.items():
                axes[9 + ax_adjust[r_name]].axhline(value, linestyle='dashed', color='k', alpha=0.5)

    def _plot_single_trial_timeseries(self, fig, axes, data, plot_group):
        # times setup
        bounds = data['bound_values'].unique()
        times = data['new_times'].to_numpy()[0]  # for full sampled data
        bin_times = np.linspace(times[0], times[-1], 500)  # for spiking rasters
        decoding_times = data['times'].to_numpy()[0]  # for decoding data
        feat = data['feature_name'].values[0]
        fr_times = np.linspace(bin_times[0], bin_times[-1], 250)

        # plot theta
        axes[0].plot(times, data['theta_amplitude'].to_numpy()[0], color='k', label='theta amplitude')
        axes[0].set(ylabel='CA1 theta')

        # plot MUA and rasters
        ax_locs = dict(CA1=[1, 3, 5, 7, 9], PFC=[2, 4, 6, 8, 10])
        for r_name, r_data in data.groupby('region'):
            # plot firing rates
            gen_data = r_data.query('choice == "initial_stay"')  # only use one choice bc otherwise duplicate data
            gen_data.sort_values('place_field_peak_ind', inplace=True, na_position='first')
            psth = (gen_data
                    .groupby(['unit_id'])
                    .apply(
                lambda x: self.aggregator.single_unit_agg._calc_psth(x['spikes'], bin_times[0], bin_times[-1], ntt=250)))
            gen_data['psth_mean'] = pd.json_normalize(psth)['psth_mean'].to_numpy()  # use psth bc z-scores data
            psth_times = pd.json_normalize(psth)['psth_times'].values[0]

            fr = np.array([compute_smoothed_firing_rate(x, bin_times, 0.01) for x in gen_data['spikes'].to_numpy()])
            axes[ax_locs[r_name][0]].step(bin_times, np.nanmean(fr, axis=0), color='k', label=f'{r_name} MUA',
                                          where='mid')
            axes[ax_locs[r_name][0]].fill_between(bin_times,  np.nanmean(fr, axis=0), color='k', alpha=0.2, step='mid')
            axes[ax_locs[r_name][0]].set(ylabel=f'{r_name} mua')

            for s_name, s_data in gen_data.groupby('max_selectivity_type'):
                fr = np.array([compute_smoothed_firing_rate(x, fr_times, 0.02) for x in s_data['spikes'].to_numpy()])
                fr_mean = np.nanmean(fr, axis=0)
                fr_sem = sem(fr, axis=0)
                # fr_mean = np.nanmean(np.vstack(s_data['psth_mean']), axis=0)
                # fr_sem = sem(np.vstack(gen_data['psth_mean']), axis=0, nan_policy='omit')
                axes[ax_locs[r_name][1]].plot(fr_times, fr_mean, color=self.colors[s_name], label=f'{r_name} {s_name} FR')
                axes[ax_locs[r_name][1]].fill_between(fr_times, fr_mean + fr_sem, fr_mean - fr_sem, alpha=0.3,
                                                      color=self.colors[s_name])
                axes[ax_locs[r_name][1]].set(ylabel=f'{r_name} goal FR (z-scored)')

            show_psth_raster(gen_data['spikes'].to_list(), ax=axes[ax_locs[r_name][2]], start=times[0], end=times[-1],
                             group_inds=gen_data['max_selectivity_type'].map({np.nan: 0, 'switch': 1, 'stay': 2}),
                             colors=[self.colors[c] for c in ['nan', 'switch', 'stay']],
                             show_legend=False)
            axes[ax_locs[r_name][2]].set(ylabel=f'{r_name} units', xlabel='')

            # plot decoding data
            prob_map = gen_data['probability'].values[0]
            feat_limits = self.virtual_track.get_limits(feat)
            prob_lims = np.linspace(feat_limits[0], feat_limits[1], np.shape(prob_map)[0])
            heatmap_times = [decoding_times[0] - (np.diff(decoding_times)[0] / 2),
                             decoding_times[-1] + (np.diff(decoding_times)[0] / 2)]
            im = axes[ax_locs[r_name][3]].imshow(prob_map * np.shape(prob_map)[0], cmap=self.colors['cmap'],
                                                 aspect='auto', origin='lower', vmin=0.5, vmax=3,
                                                 extent=[heatmap_times[0], heatmap_times[-1],
                                                         prob_lims[0], prob_lims[-1]], zorder=1)
            for b in bounds:
                axes[ax_locs[r_name][3]].axhline(b[0], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)
                axes[ax_locs[r_name][3]].axhline(b[1], linestyle='dashed', color='k', alpha=0.5, linewidth=0.5)

            axes[ax_locs[r_name][3]].plot(decoding_times, gen_data['feature'].values[0], color='w', linestyle='dashed',
                                          label=f'true {feat}')
            # axes[ax_locs[r_name][3]].plot(decoding_times, gen_data['decoding'].values[0], color='w',
            #                               label=f'predicted {feat}')
            axes[ax_locs[r_name][3]].plot(decoding_times, prob_lims[prob_map.argmax(axis=0)], color='w', label=f'predicted {feat}')
            plt.colorbar(im, ax=axes[ax_locs[r_name][3]], label='prob / chance', pad=0.01, fraction=0.046,
                         location='right')
            axes[ax_locs[r_name][3]].set(title=r_name, ylabel=gen_data['feature_name'].values[0])

            # plot decoding probabilities
            stay_prob = r_data.query('choice == "initial_stay"')['prob_over_chance'].to_numpy()[0]
            switch_prob = r_data.query('choice == "switch"')['prob_over_chance'].to_numpy()[0]
            axes[ax_locs[r_name][4]].plot(decoding_times, stay_prob, color=self.colors['stay'], label='stay')
            axes[ax_locs[r_name][4]].plot(decoding_times, switch_prob, color=self.colors['switch'], label='switch')
            axes[ax_locs[r_name][4]].set(ylabel='prob / chance')

        # plot behavioral data
        speed_threshold = 1000  # TODO - get from bayesian decoder
        speed_total = abs(data['translational_velocity'].values[0]) + abs(data['rotational_velocity'].values[0])
        axes[11].plot(times, speed_total, color='purple', label='movement')
        axes[11].fill_between(times, speed_threshold, speed_total, where=(speed_total > speed_threshold),
                             color='purple', alpha=0.2)
        axes[11].set(ylabel='roll + pitch (au)')
        axes[11].axhline(speed_threshold, linestyle='dashed', color='k', label='movement threshold')
        axes[12].plot(times, data['rotational_velocity'].values[0], color='m', label='rotation velocity')
        axes[12].set(ylabel='roll (au)')
        axes[13].plot(times, data['translational_velocity'].values[0], color='b', label='translational velocity')
        axes[13].set(ylabel='pitch (au)')

        event_labels = ['start_time', 't_delay', 't_update', 't_delay2', 't_choice_made', 'stop_time']
        colors = dict(start_time='w', t_delay='k', t_update='c', t_delay2='k', t_choice_made='y')
        event_times = (data[event_labels].iloc[0, :] - data['t_update'].iloc[0]).to_dict()
        if plot_group['update_type'][0] == 'non_update':
            colors.update(t_update='k')
        for ax in axes:
            for ind, e in enumerate(event_labels[:-1]):
                ax.axvspan(event_times[e], event_times[event_labels[ind + 1]], facecolor=colors[e], alpha=0.1, zorder=0)

        [ax.legend(loc='upper right') for ax in axes]
        [ax.axvline(0, linestyle='dashed', color='k', linewidth=0.5, zorder=3) for ax in axes]
        fig.supxlabel('Time around update cue (s)')
