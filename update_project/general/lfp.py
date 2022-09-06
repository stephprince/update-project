import numpy as np
import pandas as pd

from update_project.results_io import ResultsIO


def get_theta(nwbfile, adjust_reference=False, session_id=''):
    electrode_df = nwbfile.electrodes.to_dataframe()
    ripple_channel = electrode_df.index[electrode_df['ripple_channel'] == 1][0]

    band_df = nwbfile.processing['ecephys']['decomposition_amplitude'].bands.to_dataframe()
    band_ind = np.array(band_df.index[band_df['band_name'] == 'theta'])[0]
    amp = nwbfile.processing['ecephys']['decomposition_amplitude'].data[:, ripple_channel, band_ind]
    phase = nwbfile.processing['ecephys']['decomposition_phase'].data[:, ripple_channel, band_ind]
    rate = nwbfile.processing['ecephys']['decomposition_amplitude'].rate
    timestamps = np.arange(0, len(amp) / rate, 1 / rate)

    if adjust_reference:
        phase = adjust_theta_reference(phase, session_id)

    theta_dict = dict(amplitude=pd.Series(index=timestamps[:], data=amp),
                      phase=pd.Series(index=timestamps[:], data=phase))

    return pd.DataFrame.from_dict(theta_dict)


def adjust_theta_reference(phase, session_id):
    # load phase reference data
    results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name='phase-reference')
    fname = results_io.get_data_filename(filename='theta-phase_ref_adjustment', results_type='session', format='pkl')
    import_data = results_io.load_pickled_data(fname)
    theta_hist_df = [d for d in import_data][0]

    # get amount to adjust
    #phase_mins = theta_hist_df['phase_interval'].apply(lambda x: x.left).to_numpy()
    phase_mins = np.linspace(-np.pi, np.pi, 12)
    phase_peak = phase_mins[theta_hist_df['phase_adj'].to_numpy()[0]]
    new_ref = phase_mins[int(len(phase_mins)/2 - 1)]

    # adjust phase
    phase_adjusted = phase.copy()
    amount_to_shift = new_ref - phase_peak  #-np.pi - phase_peak
    phase_adjusted = phase_adjusted + amount_to_shift
    vals_left = phase_adjusted < -np.pi
    vals_right = phase_adjusted > np.pi
    phase_adjusted[vals_left] = phase_adjusted[vals_left] + np.pi * 2
    phase_adjusted[vals_right] = phase_adjusted[vals_right] - np.pi * 2

    assert (np.unique(np.histogram(phase, phase_mins)[0]) == np.unique(np.histogram(phase_adjusted, phase_mins)[0])).all()

    return phase_adjusted
