#import dill
import itertools

#from pathos.helpers import cpu_count
#from pathos.pools import ProcessPool as Pool
from pynwb import NWBHDF5IO

from update_project.session_loader import SessionLoader
from update_project.decoding.bayesian_decoder import BayesianDecoder
from update_project.decoding.bayesian_decoder_visualizer import SessionVisualizer, GroupVisualizer


def run_bayesian_decoding():
    # setup flags
    overwrite = False  # when False, this will only load data if the parameters match
    plot = False  # this only plots on a session by session basis
    group = True  # this compiles the data for group plotting
    parallel = False  # cannot use in conjunction with group currently

    # setup sessions
    animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
    dates_included = []  # 210913
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()

    # setup parameters - NOTE: not all parameters included here, to see defaults look inside the decoder class
    features = ['y_position', 'x_position']
    regions = [['CA1']]  # [['CA1'], ['PFC'], ['CA1', 'PFC']]
    units_thresh = [0, 20, 40]
    trials_thresh = [0, 25, 50, 100]
    testing_params = dict(encoder_trial_types=[[1], [0, 1]],
                          encoder_bins=[50],  # [30, 50],
                          decoder_bins=[0.025, 0.050, 0.25],  # [0.025, 0.050, 0.10, 0.25],
                          speed_thresholds=[1000])  # [1000, 2000, 5000], )

    # run decoder for all sessions
    args = itertools.product(session_names, regions, features, *list(testing_params.values()))  # like a nested for-loop
    if parallel:
        pool = Pool(nodes=int(cpu_count() / 2) - 1)
        pool_results = pool.map(lambda x: bayesian_decoding_parallel(plot, overwrite, session_db, testing_params, *x), args)
        pool.close()
        pool.join()
        group_data = [p.get() for p in pool_results]
    elif group:
        group_data = []
        for name, reg, feat, trial_types, enc_bins, dec_bins, speed in args:
            group_data.append(bayesian_decoding_parallel(plot, overwrite, session_db, testing_params, name, reg, feat,
                                                         trial_types, enc_bins, dec_bins, speed))

    # get decoder group summary data
    for units, trials in itertools.product(units_thresh, trials_thresh):
        if group:
            group_visualizer = GroupVisualizer(group_data,
                                               exclusion_criteria=dict(units=units, trials=trials),
                                               params=list(testing_params.keys()))
            group_visualizer.plot(group_by=dict(region=regions, feature=features))


def bayesian_decoding_parallel(plot, overwrite, session_db, testing_params, name, reg, feat, trial_types,
                               enc_bins, dec_bins, speed):
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
    params = {k: v for k, v in zip(testing_params.keys(), [tuple(trial_types), enc_bins, dec_bins, speed])}
    session_decoder_output = dict(session_id=session_id,
                                  region=tuple(reg),  # convert to tuple for later grouping
                                  feature=feat,
                                  decoder=decoder,
                                  **params)

    return session_decoder_output


if __name__ == '__main__':
    run_bayesian_decoding()
