import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import warnings

from pathlib import Path
from pynwb import NWBHDF5IO
from matplotlib import ticker

from update_project.session_loader import SessionLoader
from update_project.results_io import ResultsIO
from update_project.general.units import bin_spikes
from update_project.general.lfp import get_theta
from update_project.general.acquisition import get_velocity

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')


def get_phase_reference():
    # setup sessions
    animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
    dates_included = []  # 210913
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()
    speed_threshold = 1000
    downsample_factor = 2

    for name in session_names:
        # load nwb file
        print(f'Getting phase reference data for {session_db.get_session_id(name)}')
        io = NWBHDF5IO(session_db.get_session_path(name), 'r')
        nwbfile = io.read()
        results_io = ResultsIO(creator_file=__file__, session_id=session_db.get_session_id(name),
                               folder_name='phase-reference', )

        # get theta phase values
        theta_df = get_theta(nwbfile)
        theta_df = theta_df.iloc[::downsample_factor, :]

        # get spiking activity in CA1
        binned_spikes, units, cell_types = [], [], []
        for unit_index in range(len(nwbfile.units)):
            if nwbfile.units['region'][unit_index] == 'CA1':
                spikes = bin_spikes(nwbfile.units.get_unit_spike_times(unit_index), theta_df.index.values)
                binned_spikes.append(spikes)
                units.append(str(unit_index))
                cell_types.append(nwbfile.units['cell_type'][unit_index])

        if np.size(units):
            spike_df = pd.DataFrame(np.vstack(binned_spikes).T, columns=units, index=theta_df.index.values)
        else:
            warnings.warn('No CA1 units found in this recording so cannot adjust theta phase reference')
            spike_df = pd.DataFrame(columns=units, index=theta_df.index.values)
        pyr_cols = [str(u) for u, c in zip(units, cell_types) if c in ['Pyramidal Cell']]
        int_cols = [str(u) for u, c in zip(units, cell_types) if c in ['Wide Interneuron', 'Narrow Interneuron']]
        spike_df['total_spikes'] = spike_df.sum(axis=1)  # get CA1 group FR
        spike_df['total_spikes_pyr'] = spike_df[pyr_cols].sum(axis=1)  # get CA1 group FR
        spike_df['total_spikes_int'] = spike_df[int_cols].sum(axis=1)  # get CA1 group FR

        # get locomotor periods and apply to theta/spiking activity
        velocity = get_velocity(nwbfile)
        velocity_resampled = velocity.iloc[::downsample_factor]
        movement = velocity_resampled > speed_threshold
        theta_spiking_df = theta_df.join(spike_df)[movement]
        binned_spikes, spike_df = [], []  # clear out memory

        # histogram spiking activity by theta phase
        phase_bins = np.linspace(theta_spiking_df['phase'].min(), theta_spiking_df['phase'].max(), 12)
        theta_df_bins = pd.cut(theta_spiking_df['phase'], phase_bins)
        theta_hist_df = theta_spiking_df.groupby(theta_df_bins).apply(lambda x: np.mean(x))
        theta_hist_df = theta_hist_df.rename_axis('phase_interval').reset_index()

        # find phase of maximal CA1 firing and make adjustment so that phase is 0 degrees
        phase_adj = theta_hist_df['total_spikes'].idxmax()
        theta_hist_df['phase_adj'] = phase_adj
        phase_vect = theta_hist_df['phase'].to_numpy()/np.pi
        phase_labels = ['raw', 'post_adjustment']
        cell_type_dict = dict(all=dict(col_name='total_spikes', col_ind=units, color='k', cmap='Greys'),
                              pyr=dict(col_name='total_spikes_pyr', col_ind=pyr_cols, color='r', cmap='Reds'),
                              int=dict(col_name='total_spikes_int', col_ind=int_cols, color='b', cmap='Blues'))

        # save the data for each session
        fname = results_io.get_data_filename(filename='theta-phase_ref_adjustment', results_type='session', format='pkl')
        with open(fname, 'wb') as f:
            [pickle.dump(theta_hist_df, f)]

        # plot data (1 col for pre/post adjustment, 1 row for theta avg, 1 row for spike average, 1 row for spike indiv)
        fig, axes = plt.subplots(nrows=5, ncols=2, squeeze=False)
        for col_ind, phase_shift in enumerate([0, phase_adj]):
            theta_hist_df_adjusted = theta_hist_df.reindex(np.roll(theta_hist_df.index, -phase_shift)).reset_index()

            axes[0][col_ind].plot(phase_vect, theta_hist_df_adjusted['amplitude'], color='k')
            axes[0][col_ind].set(ylabel='amplitude', title=f'theta - {phase_labels[col_ind]}')

            counter = 0
            for cell_name, cell_data in cell_type_dict.items():
                peak = phase_vect[theta_hist_df_adjusted[cell_data['col_name']].idxmax()]
                unit_mod = np.array(theta_hist_df_adjusted[cell_data['col_ind']]).T
                unit_mod_scaled = unit_mod / np.nanmax(unit_mod, axis=1)[:, None]
                group_mod = theta_hist_df_adjusted[cell_data['col_name']]/theta_hist_df_adjusted[cell_data['col_name']].max()
                axes[1][col_ind].plot(phase_vect, group_mod, color=cell_data['color'], label=cell_name)
                axes[1][col_ind].axvline(peak, color=cell_data['color'], linestyle='dashed', zorder=0)
                axes[1][col_ind].set(ylabel='relative mod', title='total spiking modulation')

                im = axes[2+counter][col_ind].imshow(unit_mod_scaled, origin='lower', aspect='auto', vmin=0.1, vmax=0.9,
                                                     extent=[phase_vect[0], phase_vect[-1], 0, len(cell_data['col_ind'])],
                                                     cmap=cell_data['cmap'])
                axes[2+counter][col_ind].set(ylabel=f'{cell_name} units', title='unit spiking modulation')
                fig.colorbar(im, ax=axes[2+counter][col_ind], label=f'norm FR', pad=0.01, location='right', fraction=0.046)
                counter += 1

        ax_list = axes.flat
        for ax in ax_list:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))

        fig.suptitle(f'{results_io.session_id} - theta phase spiking modulation in CA1')
        fig.tight_layout()
        results_io.save_fig(fig=fig, axes=axes, filename=f'theta-phase-modulation', results_type='session')


if __name__ == '__main__':
    get_phase_reference()
