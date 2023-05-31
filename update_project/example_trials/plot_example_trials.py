from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.example_trials.example_trial_analyzer import ExampleTrialAnalyzer
from update_project.example_trials.example_trial_visualizer import ExampleTrialVisualizer

# setup sessions
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = []
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

feature = 'y_position'
regions = [['CA1'], ['PFC']]
overwrite = True  # want to overwrite usually bc different settings than normal and don't save at the end
params = dict(both_regions_only=True,
              exclusion_criteria=dict(units=20, trials=50),
              single_unit_params=dict(align_window=15, align_times=['t_update']),
              decoding_params=dict(encoder_bin_num=50, decoder_bin_size=0.2, decoder_test_size=0.2))

group_data = []
for name in session_names:
    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    analyzer = ExampleTrialAnalyzer(nwbfile=nwbfile,
                                    session_id=session_db.get_session_id(name),
                                    feature=feature,
                                    regions=regions,
                                    params=params)
    analyzer.run_analysis(overwrite=overwrite)

    session_data = dict(session_id=session_db.get_session_id(name),
                        animal=session_db.get_animal_id(name),
                        analyzer=analyzer,
                        feature_name=feature,
                        **params)
    group_data.append(session_data)

visualizer = ExampleTrialVisualizer(group_data)
group_data = []  # clear out memory
visualizer.plot()