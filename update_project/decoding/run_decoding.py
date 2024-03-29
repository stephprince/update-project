import itertools

from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.decoding.bayesian_decoder_analyzer import BayesianDecoderAnalyzer
from update_project.decoding.bayesian_decoder_visualizer import BayesianDecoderVisualizer


def run_decoding():
    # setup flags
    overwrite = False  # when False, this will only load data if the parameters match
    plot = False  # this only plots on a session by session basis
    group = True  # this compiles the data for group plotting
    parallel = False  # cannot be run in conjunction with group currently

    # setup sessions
    animals = [17, 20, 25, 28, 29, 33, 34]
    dates_included = []
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()

    # setup parameters - NOTE: not all parameters included here, to see defaults look inside the decoder class
    features = ['choice']
    regions = [['PFC']]
    exclusion_criteria = dict(units=20, trials=50)  # include sessions with this minimum number of units/trials
    testing_params = dict(encoder_bin_num=[50],  # switch to 40 for x-position
                          decoder_bin_size=[0.2],
                          dec_test_size=[0.2])

    # run decoder for all sessions
    args = itertools.product(session_names, regions, features, *list(testing_params.values()))  # like a nested for-loop
    if parallel:
        pool = Pool(nodes=int(cpu_count() / 2) - 2)
        pool.map(lambda x: bayesian_decoding(plot, overwrite, parallel, session_db, testing_params, *x), args)
        pool.close()
        pool.join()

    group_data = []
    for arg_list in args:
        group_data.append(bayesian_decoding(plot, overwrite, parallel, session_db, testing_params, *arg_list))

    if group:
        group_visualizer = BayesianDecoderVisualizer(group_data,
                                                     exclusion_criteria=exclusion_criteria,
                                                     params=list(testing_params.keys()),)
        group_visualizer.plot(group_by=dict(region=regions, feature=features))

    print(f'Finished running {__file__}')


def bayesian_decoding(plot, overwrite, parallel, session_db, testing_params, name, reg, feat, enc_bins, dec_bins, dec_test_size):
    # load nwb file
    session_id = session_db.get_session_id(name)
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    # run decoder
    params = dict(units_types=dict(region=reg, cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']),
                  encoder_bin_num=enc_bins,  # num feature bins
                  decoder_bin_size=dec_bins,
                  decoder_test_size=dec_test_size)  # time length of decoding bins
    analyzer = BayesianDecoderAnalyzer(nwbfile=nwbfile, params=params, session_id=session_id,
                                      features=[feat])  # initialize decoder class
    analyzer.run_analysis(overwrite=overwrite)  # build decoding model

    # save to group output
    if parallel:
        return None  # cannot return output for parallel processing bc contains h5py object which cannot be pickled
    else:
        params = {k: v for k, v in zip(list(testing_params.keys()), [enc_bins, dec_bins, dec_test_size])}
        session_decoder_output = dict(session_id=session_id,
                                      animal=session_db.get_animal_id(name),
                                      analyzer=analyzer,
                                      **params)
        return session_decoder_output


if __name__ == '__main__':
    run_decoding()
