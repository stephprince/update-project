from update_project.general.session_loader import SessionLoader
from update_task_figure_generator import UpdateTaskFigureGenerator


def run_demo(animals, dates_included=None, overwrite=False):
    # load relevant sessions
    session_db = SessionLoader(animals=animals, dates_included=dates_included, demo=True)
    example_session = session_db.get_session_id(session_db.load_session_names()[0])

    # initialize general analysis class and run for each of the specified steps
    analyzer = UpdateTaskFigureGenerator(sessions=session_db, overwrite=overwrite)

    # generate all manuscript figures
    analyzer.plot_demo_figure(session=example_session)


if __name__ == '__main__':
    animals = [17, 20, 25, 28, 29, 33, 34]
    dates_included = [210913]

    run_demo(animals, dates_included=dates_included, overwrite=True)
