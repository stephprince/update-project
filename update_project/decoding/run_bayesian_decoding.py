import itertools

from pynwb import NWBHDF5IO

from update_project.session_loader import SessionLoader
from update_project.decoding.bayesian_decoder import BayesianDecoder
from update_project.decoding.bayesian_decoder_visualizer import SessionVisualizer, GroupVisualizer

# setup flags
overwrite = False
plot = False
group = False

# setup sessions
animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = []  # 210913
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

# setup parameters - NOTE: not all parameters included here, to see defaults look inside the decoder class
features = ['y_position', 'x_position', 'view_angle']  # 'choice', 'turn_type']
regions = [['CA1']]  # [['CA1'], ['PFC'], ['CA1', 'PFC']]
units_thresh = [0, 20, 40]
trials_thresh = [0, 25, 50, 100]
testing_params = dict(encoder_trial_types=[[1], [0, 1]],
                      encoder_bins=[30, 50],
                      decoder_bins=[0.025, 0.050, 0.10, 0.25],
                      speed_thresholds=[1000, 2000, 5000], )

# run decoder for all sessions (itertools equivalent to nested for-loop)
group_data = []
loop_counter = 0
for name, reg, feat, trial_types, enc_bins, dec_bins, speed in itertools.product(session_names, regions, features,
                                                                                 *list(testing_params.values())):
    print(f'Running decoding for loop {loop_counter} out of '
          f'{len(list(itertools.product(session_names, regions, features, *list(testing_params.values()))))}')
    loop_counter += 1

    # load nwb file
    session_id = session_db.get_session_id(name)
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    # run decoder
    params = dict(units_types=dict(region=reg, cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']),
                  encoder_trial_types=dict(update_type=[1], correct=trial_types),
                  encoder_bin_num=enc_bins,  # num feature bins
                  decoder_bin_size=dec_bins,  # time length of decoding bins
                  speed_threshold=speed)  # cutoff for moving/not-moving times
    decoder = BayesianDecoder(nwbfile=nwbfile, params=params, session_id=session_id,
                              features=[feat])  # initialize decoder class
    decoder.run_decoding(overwrite=overwrite)  # build decoding model

    # plot data
    if plot:
        visualizer = SessionVisualizer(decoder)
        visualizer.plot()

    # save to group output
    if group:
        params = {k: v for k, v in zip(testing_params.keys(), [tuple(trial_types), enc_bins, dec_bins, speed])}
        session_decoder_output = dict(session_id=session_id,
                                      region=tuple(reg),  # convert to tuple for later grouping
                                      feature=feat,
                                      decoder=decoder,
                                      **params)
        group_data.append(session_decoder_output)  # save for group plotting

# get decoder group summary data
for units, trials in itertools.product(units_thresh, trials_thresh):
    if group:
        group_visualizer = GroupVisualizer(group_data,
                                           exclusion_criteria=dict(units_threshold=units, trials_threshold=trials),
                                           params=list(testing_params.keys()))
        group_visualizer.plot(group_by=dict(region=regions, feature=features))
