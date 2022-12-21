import numpy as np
import pandas as pd

from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.experiment_metadata.experiment_metadata_analyzer import ExperimentMetadataAnalyzer

# setup flags
overwrite = True  # when False, this will only load data if the parameters match

# setup sessions
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = []
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

# loop through individual sessions
group_data = []
for name in session_names:
    # load nwb file
    session_id = session_db.get_session_id(name)
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    # run analysis
    analyzer = ExperimentMetadataAnalyzer(nwbfile=nwbfile, session_id=session_id)
    analyzer.run_analysis(overwrite=overwrite)  # build decoding model

    # save to group output
    group_data.append(analyzer.metadata_df)

# table with animal ID, # sessions, approx recording locations, # PFC units, # CA1 units, # behavioral trials
group_df = pd.concat(group_data, axis=1).transpose()
columns_to_sum = group_df.loc[:, 'total units':].columns.to_list()
group_df_by_animal = (group_df
                      .groupby('animal')
                      .agg({'session_id': len,
                            'PFC location': lambda x: x.values[0],  # same for all of them bc just from the last day
                            'CA1 location': lambda x: x.values[0],
                            **{x: np.sum for x in columns_to_sum}}))
group_df_by_animal.to_csv(analyzer.results_io.get_results_path() / 'experiment_metadata_summary.csv')
