import pandas as pd

from pathlib import Path


def get_commitment_data(nwbfile, results_io):
    # load view angle
    time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')
    view_angle = time_series.data[:]

    # load dynamic choice from saved output
    choice_path = Path(__file__).parent.parent.parent / 'results' / 'choice'
    fname = results_io.get_data_filename(filename=f'dynamic_choice_output_choice', results_type='session', format='pkl',
                                         diff_base_path=choice_path)
    import_data = results_io.load_pickled_data(fname)
    for v, i_data in zip(['output_data', 'agg_data', 'decoder_data', 'params'], import_data):  # TODO - make this robust
        if v == 'decoder_data':
            choice_data = i_data
    choice_commitment = choice_data - 0.5  # convert to -0.5 and +0.5 for flipping between trials

    commitment = pd.DataFrame(dict(view_angle=pd.Series(index=time_series.timestamps[:],
                                                        data=view_angle),
                                   choice_commitment=pd.Series(index=time_series.timestamps[:],
                                                               data=choice_commitment)))

    return commitment


def get_view_angle(nwbfile):
    time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')
    view_angle = pd.DataFrame(dict(view_angle=pd.Series(index=time_series.timestamps[:], data=time_series.data[:])))

    return view_angle


def get_position(nwbfile):
    time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
    position = pd.DataFrame(dict(x_position=pd.Series(index=time_series.timestamps[:], data=time_series.data[:, 0]),
                                 y_position=pd.Series(index=time_series.timestamps[:], data=time_series.data[:, 1])))

    return position
