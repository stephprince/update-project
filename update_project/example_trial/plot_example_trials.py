import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO

from update_project.session_loader import SessionLoader
from update_project.results_io import ResultsIO
from update_project.general.plots import get_color_theme
from update_project.general.psth import show_start_aligned_psth, get_annotation_times
from update_project.general.units import align_by_time_intervals as align_by_time_intervals_units
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts


def plot_example_trials():
    # setup sessions
    animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
    dates_included = [210913]  # 210913
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()

    n_trials = 1  # number of trials to plot from each trial type
    # loop through individual sessions
    for name in session_names:
        # load nwb file
        io = NWBHDF5IO(session_db.get_session_path(name), 'r')
        nwbfile = io.read()
        results_io = ResultsIO(creator_file=__file__, session_id=session_db.get_session_id(name),
                               folder_name='example-trial', )

        # get example trials
        trials_df = nwbfile.trials.to_dataframe()
        mask = pd.concat([trials_df['update_type'] == 1, trials_df['maze_id'] == 4], axis=1).all(axis=1)
        trial_inds_dict = dict(non_update=np.array(trials_df.index[mask][:n_trials]),
                               switch=np.array(trials_df.index[trials_df['update_type'] == 2][:n_trials]),
                               stay=np.array(trials_df.index[trials_df['update_type'] == 3][:n_trials]))
        trial_inds_df = pd.DataFrame(trial_inds_dict).melt()
        trial_inds_df.sort_values('value', inplace=True, ignore_index=True)
        trial_inds = trial_inds_df['value'].values

        # get channel data
        electrode_df = nwbfile.electrodes.to_dataframe()
        ripple_channel = electrode_df.index[electrode_df['ripple_channel'] == 1][0]
        channel_regions = electrode_df['location'].values
        channel_regions = channel_regions[channel_regions != 'none']

        # get acquisition data
        print(f'Loading acquisition data for {session_db.get_session_id(name)}...')
        raw_data, raw_times = align_by_time_intervals_ts(nwbfile.acquisition['raw_ecephys'],
                                                         nwbfile.intervals['trials'][trial_inds],
                                                         return_timestamps=True)
        lfp_data, lfp_times = align_by_time_intervals_ts(nwbfile.processing['ecephys']['LFP']['LFP'],
                                                         nwbfile.intervals['trials'][trial_inds],
                                                         return_timestamps=True)
        lick_data = align_by_time_intervals_ts(nwbfile.acquisition['licks'],
                                               nwbfile.intervals['trials'][trial_inds])

        # get processed data
        print(f'Loading processed data for {session_db.get_session_id(name)}...')
        position_data, vr_times = align_by_time_intervals_ts(nwbfile.processing['behavior']['position']['position'],
                                                             nwbfile.intervals['trials'][trial_inds],
                                                             return_timestamps=True)
        decomp_data = align_by_time_intervals_ts(nwbfile.processing['ecephys']['decomposition_amplitude'],
                                                nwbfile.intervals['trials'][trial_inds])
        band_df = nwbfile.processing['ecephys']['decomposition_amplitude'].bands.to_dataframe()
        band_ind = np.array(band_df.index[band_df['band_name'] == 'theta'])[0]
        theta_data = [d[:, ripple_channel, band_ind] for d in decomp_data]

        # get units data
        print(f'Loading unit data for {session_db.get_session_id(name)}...')
        unit_group = []
        spikes_aligned = []
        for unit_index in range(len(nwbfile.units)):
            spikes_aligned.append(
                align_by_time_intervals_units(nwbfile.units, unit_index, nwbfile.intervals['trials'][trial_inds]))
            unit_group.append(nwbfile.units['region'][unit_index])
        annotation_times = get_annotation_times(nwbfile.intervals['trials'][trial_inds])

        # get digital data
        dig_data = []
        for ind, row in trial_inds_df.iterrows():
            df = trials_df.iloc[row['value']]
            cue_ts, delay_ts, update_ts = np.zeros(len(lfp_times[ind])), np.zeros(len(lfp_times[ind])), np.zeros(
                len(lfp_times[ind]))
            cue_on = np.logical_and(lfp_times[ind] >= df['start_time'], lfp_times[ind] <= df['t_delay'])
            delay_on = np.logical_and(lfp_times[ind] >= df['t_delay'], lfp_times[ind] <= df['t_update'])
            update_on = np.logical_and(lfp_times[ind] >= df['t_update'], lfp_times[ind] <= df['t_delay2'])
            delay_2_on = np.logical_and(lfp_times[ind] >= df['t_delay2'], lfp_times[ind] <= df['t_choice_made'])
            cue_ts[cue_on] = 1
            delay_ts[np.logical_or(delay_on, delay_2_on)] = 1
            update_ts[update_on] = 1
            dig_data.append(dict(cue=cue_ts, delay=delay_ts, update=update_ts))

        # plot example trials
        print(f'Plotting example data for {session_db.get_session_id(name)}...')
        colors = get_color_theme()
        for trial_type_name in ['non_update', 'switch', 'stay']:
            trial_type_inds = np.array(trial_inds_df.index[trial_inds_df['variable'] == trial_type_name])
            mosaic = """
            A
            A
            A
            A
            A
            B
            C
            C
            C
            D
            E
            """
            axes = plt.figure(figsize=(17, 22)).subplot_mosaic(mosaic, sharex=True)
            for plot_id, ind in enumerate(trial_type_inds):
                # plot raw ephys channels
                channels_to_plot = channel_regions[np.r_[0:32, 96:128]]  # only plot first and last shank
                for ch, reg in enumerate(channels_to_plot[::2]):  # skip every other channel to visualize
                    axes['A'].plot(raw_times[ind]-raw_times[ind][0], raw_data[ind][:, ch] + 0.25 * ch, color=colors[reg], linewidth=0.5)
                    axes['A'].plot(raw_times[ind]-raw_times[ind][0], raw_data[ind][:, ch] + 0.25 * ch, color=colors[reg], linewidth=0.5)

                # plot filtered ephys from hippocampal ripple channel (to show theta)
                axes['B'].plot(lfp_times[ind]-lfp_times[ind][0], lfp_data[ind][:, ripple_channel], linewidth=0.5)
                axes['B'].plot(lfp_times[ind]-lfp_times[ind][0], theta_data[ind] + 1)

                # plot single unit activity
                trial_spikes = [s[ind] for s in spikes_aligned]
                note_times = {k: [v.iloc[ind]] for k, v in annotation_times.items()}
                group_mapping = {'CA1': 1, 'PFC': 2}
                group_ids = [group_mapping[u] for u in unit_group]
                show_start_aligned_psth(trial_spikes, note_times, group_ids, end=np.ptp(lfp_times[ind]),
                                        axes=[axes['C']])

                # plot analog and digital channels
                times = lfp_times[ind]-lfp_times[ind][0]
                axes['D'].plot(times, dig_data[ind]['cue'], color=colors['general'][0], label='cue')
                axes['D'].plot(times, dig_data[ind]['delay'] + 1.5, color=colors['general'][1], label='delay')
                axes['D'].plot(times, dig_data[ind]['update'] + 3, color=colors['general'][2], label='update')
                axes['D'].plot(times, lick_data[ind] / np.max(lick_data[ind]) + 4.5,
                               color=colors['general'][3], label='licks')

                # plot position
                axes['E'].plot(vr_times[ind]-vr_times[ind][0], position_data[ind][:, 0], color=colors['general'][4])
                axes['E'].plot(vr_times[ind]-vr_times[ind][0], position_data[ind][:, 1], color=colors['general'][5])

            axes['A'].set_xlim([0, np.max(times)])
            axes['A'].set_title(f'Example trial with ephys - {trial_type_name} - {session_db.get_session_id(name)}')
            tags = f'{trial_type_name}_plot{plot_id}'
            fig = axes['A'].get_figure()
            results_io.save_fig(fig=fig, axes=axes, filename=f'example-trial-with-ephys', additional_tags=tags)


if __name__ == '__main__':
    plot_example_trials()
