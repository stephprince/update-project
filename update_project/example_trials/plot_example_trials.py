from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.single_units.single_unit_analyzer import SingleUnitAnalyzer
from update_project.decoding.bayesian_decoder import BayesianDecoder
from update_project.example_trials.example_trial_visualizer import ExampleTrialVisualizer

# setup sessions
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = []
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

feature = 'y_position'
regions = [['CA1'], ['PFC']]
exclusion_criteria = dict(units=50, trials=50)  # higher than others bc just plotting examples and both PFC + HPC
overwrite = True
align_window = 5
align_times = ['t_update']

group_data = []
for name in session_names:
    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    for reg in regions:
        unit_types = dict(region=reg, cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron'])

        # load existing data
        analyzer = SingleUnitAnalyzer(nwbfile=nwbfile, session_id=session_db.get_session_id(name), feature=feature,
                                      params=dict(align_window=align_window, align_times=align_times,
                                                  units_types=unit_types))
        analyzer.run(overwrite=overwrite, export_data=False)  # don't use existing data but also don't save it

        decoder = BayesianDecoder(nwbfile=nwbfile, session_id=session_db.get_session_id(name), features=[feature],
                                  params=dict(units_types=unit_types))
        decoder.run_decoding(export_data=False)

        # save to group output
        session_data = dict(session_id=session_db.get_session_id(name),
                            animal=session_db.get_animal_id(name),
                            analyzer=analyzer,
                            decoder=decoder,
                            region=reg,
                            feature_name=feature)
        group_data.append(session_data)

visualizer = ExampleTrialVisualizer(group_data, exclusion_criteria=exclusion_criteria, align_window=align_window,
                                    align_times=align_times)
group_data = []  # clear out memory
visualizer.plot()
