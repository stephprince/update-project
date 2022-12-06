from update_project.general.session_loader import SessionLoader
from update_task_analyzer import UpdateTaskAnalyzer

overwrite = False  # whether or not to overwrite existing data structures
animals = [17, 20, 25, 28, 29, 33, 34]
dates_included = [210913]
session_db = SessionLoader(animals=animals, dates_included=dates_included)

# initialize general analysis class and run for each of the specificed steps
analyzer = UpdateTaskAnalyzer(analysis=dict(Behavior=dict(),
                                            ChoiceEstimate=dict(target_var='choice'),
                                            Decoder=dict(features=['y_position', 'x_position', 'choice'], ),
                                            SingleUnits=dict()),
                              sessions=session_db,
                              overwrite=overwrite)

# generate all manuscript figures
analyzer.plot_main_figures()
analyzer.plot_supplemental_figures()
