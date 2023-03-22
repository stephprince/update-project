from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.single_units.single_unit_analyzer import SingleUnitAnalyzer
from update_project.single_units.single_unit_visualizer import SingleUnitVisualizer

# setup sessions
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = []
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

overwrite = True
plot = True
feature = 'y_position'
params = dict(units_types=dict(region=['CA1', 'PFC'],
                               cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))

group_data = []
for name in session_names:
    # load nwb file
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    analyzer = SingleUnitAnalyzer(nwbfile=nwbfile, session_id=session_db.get_session_id(name), feature=feature,
                                  params=params)
    analyzer.run_analysis(overwrite=overwrite)

    # save to group output
    session_data = dict(session_id=session_db.get_session_id(name),
                        animal=session_db.get_animal_id(name),
                        analyzer=analyzer,
                        feature_name=feature,
                        **params)
    group_data.append(session_data)

if plot:
    visualizer = SingleUnitVisualizer(group_data)
    visualizer.plot()