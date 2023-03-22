import itertools

from multiprocessing import Pool
from update_project.general.session_loader import SessionLoader


def test_func(name, reg, feat, trial_types, enc_bins, dec_bins, speed):
    # save to group output
    session_decoder_output = dict(session_id=name,
                                  region=tuple(reg),  # convert to tuple for later grouping
                                  feature=feat,)

    return session_decoder_output


if __name__ == '__main__':
    # setup flags
    overwrite = False  # when False, this will only load data if the parameters match
    plot = False  # this only plots on a session by session basis
    group = False  # this compiles the data for group plotting

    # setup sessions
    animals = [29]  # 17, 20, 25, 28, 29
    dates_included = []  # 210913
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()

    # setup parameters - NOTE: not all parameters included here, to see defaults look inside the decoder class
    features = ['y_position']  # , 'x_position', 'view_angle']  # ['choice', 'turn_type']
    regions = [['CA1']]  # [['CA1'], ['PFC'], ['CA1', 'PFC']]
    units_thresh = [0, 20, 40]
    trials_thresh = [0, 25, 50, 100]
    testing_params = dict(encoder_trial_types=[[1], [0, 1]],
                          encoder_bins=[30, 50],
                          decoder_bins=[0.025, 0.050, 0.10, 0.25],
                          speed_thresholds=[1000, 2000, 5000], )

    # run decoder for all sessions (itertools equivalent to nested for-loop)
    pool = Pool(6)  # initialize parallel processing
    group_data = pool.starmap(test_func, itertools.product(session_names, regions, features, *list(testing_params.values())))

    pool.close()
    pool.join()
    results = [r for r in group_data]