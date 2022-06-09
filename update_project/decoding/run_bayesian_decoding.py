from pynwb import NWBHDF5IO

from update_project.session_loader import SessionLoader
from update_project.decoding.bayesian_decoder import BayesianDecoder
from update_project.decoding.bayesian_decoder_visualizer import BayesianDecoderVisualizer

# setup sessions
animals = [25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = []  # 210913
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = session_db.load_sessions()

# setup parameters - NOTE: not all parameters included here, to see defaults look inside the decoder class
overwrite = False
features = ['x_position', 'y_position', 'view_angle', 'choice', 'turn_type']
params = dict(units_threshold=0,
              speed_threshold=0,
              firing_threshold=0,
              units_types=dict(region=['CA1', 'PFC'],
                               cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']),
              encoder_trial_types=dict(update_type=[1], correct=[0, 1]),
              encoder_bin_num=30,
              decoder_trial_types=dict(update_type=[1, 2, 3], correct=[0, 1]),
              decoder_bin_size=0.25,
              decoder_bin_type='time',
              decoder_test_size=0.25,
              )

# run decoder for all sessions
group_data = dict()
for name, session in unique_sessions:

    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    # run decoder on all features
    group_data[session_db.get_session_id(name)] = dict()
    for feat in features:
        decoder = BayesianDecoder(nwbfile=nwbfile, params=params, session_id=session_db.get_session_id(name),
                                  features=[feat])  # initialize decoder class
        decoder.run_decoding(overwrite=overwrite)   # build decoding model

        visualizer = BayesianDecoderVisualizer(decoder, type='session')
        visualizer.plot()  # plot data

        group_data[session_db.get_session_id(name)][feat] = decoder  # save for group plotting

# get decoder group summary data
group_visualizer = BayesianDecoderVisualizer(group_data, type='group', features=features, sessions=group_data.keys)
group_visualizer.plot()
