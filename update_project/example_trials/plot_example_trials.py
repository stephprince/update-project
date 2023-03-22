from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.single_units.single_unit_analyzer import SingleUnitAnalyzer
from update_project.decoding.bayesian_decoder_analyzer import BayesianDecoderAnalyzer
from update_project.example_trials.example_trial_visualizer import ExampleTrialVisualizer

# setup sessions
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = []
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

feature = 'y_position'
regions = [['CA1'], ['PFC']]
exclusion_criteria = dict(units=20, trials=50)  # higher than others bc just plotting examples and both PFC + HPC
overwrite = True
single_unit_params = dict(align_window=2.5, align_times=['t_update'])
decoding_params = dict(encoder_bin_num=50, decoder_bin_size=0.20, decoder_test_size=0.2)

group_data = []
for name in session_names:
    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    for reg in regions:
        decoding_params.update(units_types=dict(region=reg,
                                               cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))
        single_unit_params.update(units_types=dict(region=reg,
                                               cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))

        # load existing data
        analyzer = SingleUnitAnalyzer(nwbfile=nwbfile, session_id=session_db.get_session_id(name), feature=feature,
                                      params=single_unit_params)
        analyzer.run_analysis(overwrite=overwrite, export_data=False)  # don't use existing data but also don't save it

        decoder = BayesianDecoderAnalyzer(nwbfile=nwbfile, session_id=session_db.get_session_id(name), features=[feature],
                                          params=decoding_params)
        decoder.run_analysis(export_data=False)

        # save to group output
        session_data = dict(session_id=session_db.get_session_id(name),
                            animal=session_db.get_animal_id(name),
                            analyzer=analyzer,
                            decoder=decoder,
                            region=reg,
                            feature_name=feature)
        group_data.append(session_data)

visualizer = ExampleTrialVisualizer(group_data, exclusion_criteria=exclusion_criteria, align_window=single_unit_params['align_window'],
                                    align_times=single_unit_params['align_times'])
group_data = []  # clear out memory
visualizer.plot()