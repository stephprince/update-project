from pynwb import NWBHDF5IO

from update_project.general.session_loader import SessionLoader
from update_project.behavior.behavior_analyzer import BehaviorAnalyzer
from update_project.behavior.behavior_visualizer import BehaviorVisualizer


def run_behavior_analysis():
    # setup flags
    overwrite = True  # when False, this will only load data if the parameters match
    plot = False  # this only plots on a session by session basis
    group = True  # this compiles the data for group plotting

    # setup sessions
    animals = [17, 20, 25, 28, 29, 33, 34]
    dates_included = [210913]
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
        behavior = BehaviorAnalyzer(nwbfile=nwbfile, session_id=session_id)
        behavior.run_analysis(overwrite=overwrite)  # build decoding model

        # save to group output
        session_data = dict(session_id=session_id,
                            animal=session_db.get_animal_id(name),
                            behavior=behavior)
        group_data.append(session_data)

        # plot data
        if plot:
            visualizer = BehaviorVisualizer([session_data], session_id=session_id)
            visualizer.plot()

    # get group summary data
    if group:
        group_visualizer = BehaviorVisualizer(group_data)
        group_visualizer.plot()


if __name__ == '__main__':
    run_behavior_analysis()
