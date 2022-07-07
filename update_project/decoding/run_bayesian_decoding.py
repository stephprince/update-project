import dill
import itertools

from pathos.helpers import cpu_count
from pathos.pools import ProcessPool as Pool
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
    features = ['x_position', 'y_position', 'view_angle']
    regions = [['PFC']]  # run PFC later
    exclusion_criteria = dict(units=20, trials=50)  # include sessions with this minimum number of units/trials
    testing_params = dict(encoder_bins=[40, 60],
                          decoder_bins=[0.050, 0.25])

    # run decoder for all sessions
    args = itertools.product(session_names, regions, features, *list(testing_params.values()))  # like a nested for-loop
    if parallel:
        pool = Pool(nodes=int(cpu_count() / 2) - 1)
        pool.map(lambda x: bayesian_decoding(plot, overwrite, parallel, session_db, testing_params, *x), args)
        pool.close()
        pool.join()

    if group:
        group_data = []
        for arg_list in args:
            group_data.append(bayesian_decoding(plot, overwrite, parallel, session_db, testing_params, *arg_list))

        group_visualizer = GroupVisualizer(group_data,
                                           exclusion_criteria=exclusion_criteria,
                                           params=list(testing_params.keys()))
        group_visualizer.plot(group_by=dict(region=regions, feature=features))

    print(f'Finished running {__file__}')


def bayesian_decoding(plot, overwrite, parallel, session_db, testing_params, name, reg, feat, enc_bins, dec_bins):
    # load nwb file
    session_id = session_db.get_session_id(name)
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    # run decoder
    params = dict(units_types=dict(region=reg, cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']),
                  encoder_bin_num=enc_bins,  # num feature bins
                  decoder_bin_size=dec_bins,)  # time length of decoding bins
    decoder = BayesianDecoder(nwbfile=nwbfile, params=params, session_id=session_id,
                              features=[feat])  # initialize decoder class
    decoder.run_decoding(overwrite=overwrite)  # build decoding model

    # plot data
    if plot:
        visualizer = SessionVisualizer(decoder)
        visualizer.plot()

    # save to group output
    if parallel:
        return None  # cannot return output for parallel processing bc contains h5py object which cannot be pickled
    else:
        params = {k: v for k, v in zip(testing_params.keys(), [enc_bins, dec_bins])}
        session_decoder_output = dict(session_id=session_id,
                                      animal=session_db.get_animal_id(name),
                                      region=tuple(reg),  # convert to tuple for later grouping
                                      feature=feat,
                                      decoder=decoder,
                                      **params)
        return session_decoder_output

    return None


if __name__ == '__main__':
    run_bayesian_decoding()
