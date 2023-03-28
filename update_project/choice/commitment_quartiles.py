import pandas as pd
import pickle

from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.general.results_io import ResultsIO
from update_project.general.preprocessing import get_commitment_data
from update_project.general.trials import get_trials_dataframe


def get_commitment_quartiles(overwrite=False, session_id=None):
    # setup sessions
    animals = [17, 20, 25, 28, 29, 33, 34]
    dates_included = []  # 210913
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()
    align_time = 't_update'

    df_list = []
    for name in session_names:
        results_io = ResultsIO(creator_file=__file__, session_id=session_db.get_session_id(name),
                               folder_name='choice', )

        if overwrite:
            # load nwb file
            print(f'Getting commitment data for {session_db.get_session_id(name)}')
            io = NWBHDF5IO(session_db.get_session_path(name), 'r')
            nwbfile = io.read()

            # get view angle at update
            commitment_data = get_commitment_data(nwbfile, results_io)
            trials = get_trials_dataframe(nwbfile, with_pseudoupdate=True)
            update_trials = trials.dropna(subset=align_time)
            update_trials['view_angle_at_update'] = (commitment_data['view_angle']
                                                     .loc[list(update_trials[align_time])]
                                                     .to_numpy())

            # merge with full trial table
            trials = trials.join(update_trials['view_angle_at_update'], how='left')
            trials['session_id'] = session_db.get_session_id(name)

            fname = results_io.get_data_filename(filename='commitment_quantiles', results_type='session', format='pkl')
            with open(fname, 'wb') as f:
                [pickle.dump(trials, f)]
        else:
            fname = results_io.get_data_filename(filename='commitment_quantiles', results_type='session', format='pkl')
            data = results_io.load_pickled_data(fname)
            for d in data:
                trials = d

        df_list.append(trials)

    # calculate group quartiles and save data
    group_df = pd.concat(df_list, axis=0)
    group_df['view_angle_quantile'] = pd.qcut(group_df['view_angle_at_update'], q=4,
                                              labels=[f'q{i}' for i in range(4)][::-1])  # reversed
    if session_id:
        return group_df.query(f'session_id == "{session_id}"')
    else:
        return group_df


if __name__ == '__main__':
    get_commitment_quartiles()
