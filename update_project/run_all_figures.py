from update_project.general.session_loader import SessionLoader
from update_task_figure_generator import UpdateTaskFigureGenerator

overwrite = False  # whether or not to overwrite existing data structures
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = []
session_db = SessionLoader(animals=animals, dates_included=dates_included)

# initialize general analysis class and run for each of the specified steps
analyzer = UpdateTaskFigureGenerator(sessions=session_db, overwrite=overwrite)

# generate all manuscript figures
analyzer.plot_main_figures()
analyzer.plot_supplemental_figures()
