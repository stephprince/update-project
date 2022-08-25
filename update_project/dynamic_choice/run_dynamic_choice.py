import matplotlib.pyplot as plt

from pathlib import Path
from pynwb import NWBHDF5IO

from update_project.session_loader import SessionLoader
from update_project.dynamic_choice.dynamic_choice_rnn import DynamicChoiceRNN
from update_project.dynamic_choice.dynamic_choice_visualizer import DynamicChoiceVisualizer

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')

# setup sessions
animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = [210520, 210909]
dates_excluded = []
session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
session_names = session_db.load_session_names()

overwrite = False
plot = True
grid_search = False

group_data = []
for name in session_names:
    # load nwb file
    print(f'Getting dynamic choice data for {session_db.get_session_id(name)}')
    io = NWBHDF5IO(session_db.get_session_path(name), 'r')
    nwbfile = io.read()

    rnn = DynamicChoiceRNN(nwbfile=nwbfile, session_id=session_db.get_session_id(name))
    rnn.run(overwrite=overwrite, grid_search=grid_search)

    # save to group output
    session_data = dict(session_id=session_db.get_session_id(name),
                        animal=session_db.get_animal_id(name),
                        rnn=rnn)
    group_data.append(session_data)

if plot:
    visualizer = DynamicChoiceVisualizer(group_data, grid_search=grid_search)
    visualizer.plot()
