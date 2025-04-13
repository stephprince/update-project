from update_project.general.session_loader import SessionLoader
from update_task_figure_generator import UpdateTaskFigureGenerator


def run_all_figures(animals, dates_included=None, overwrite=False):
    # load relevant sessions
    session_db = SessionLoader(animals=animals, dates_included=dates_included)

    # initialize general analysis class and run for each of the specified steps
    analyzer = UpdateTaskFigureGenerator(sessions=session_db, overwrite=overwrite)

    # generate all manuscript figures
    analyzer.plot_main_figures()
    analyzer.plot_supplemental_figures()


if __name__ == '__main__':
    animals = [17, 20, 25, 28, 29, 33, 34]
    run_all_figures(animals)
