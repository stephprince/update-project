import itertools

from pathlib import Path
from pynwb import NWBHDF5IO

from update_project.results_io import ResultsIO
from update_project.session_loader import SessionLoader
from update_project.decoding.bayesian_decoder import BayesianDecoder
from update_project.decoding.bayesian_decoder_visualizer import BayesianDecoderVisualizer

# setup sessions
animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = []  # 210913
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

# setup parameters - NOTE: not all parameters included here, to see defaults look inside the decoder class
overwrite = False
plot = False
features = ['x_position', 'y_position', 'view_angle', 'choice', 'turn_type']
regions = [['CA1'], ['PFC'], ['CA1', 'PFC']]
units_thresh = [0, 25, 50]  # TODO - test these out
trials_thresh = [0, 25, 50]  # TODO - test these out
params = dict(speed_threshold=0,
              firing_threshold=0,
              units_types=dict(cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']),
              encoder_trial_types=dict(update_type=[1], correct=[0, 1]),
              encoder_bin_num=30,
              decoder_trial_types=dict(update_type=[1, 2, 3], correct=[0, 1]),
              decoder_bin_size=0.25,
              decoder_bin_type='time',
              decoder_test_size=0.25,
              )

# run decoder for all sessions (itertools equivalent to nested for-loop)
group_data = []
for name, reg, feat in itertools.product(session_names, regions, features):
    # update params based on loop
    params['units_types'].update(region=reg)

    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    # run decoder
    decoder = BayesianDecoder(nwbfile=nwbfile, params=params, session_id=session_db.get_session_id(name),
                              features=[feat])  # initialize decoder class
    decoder.run_decoding(overwrite=overwrite)   # build decoding model

    # plot data
    if plot:
        visualizer = BayesianDecoderVisualizer(decoder, type='session')
        visualizer.plot()

    # save to group output
    session_decoder_output = dict(session_id=session_db.get_session_id(name),
                                  region=tuple(reg),  # convert to tuple for later grouping
                                  feature=feat,
                                  decoder=decoder)
                                 #units_threshold=units,
                                 #trials_threshold=trials)
    group_data.append(session_decoder_output)  # save for group plotting

# get decoder group summary data
group_visualizer = BayesianDecoderVisualizer(group_data, type='group')
group_visualizer.plot(group_by=dict(region=regions, feature=features))
