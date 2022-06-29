import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from git import Repo
from pathlib import Path

from update_project.session_loader import get_session_info
from archive.decoding import get_decoding_error_summary, plot_decoding_error_summary

# set inputs
animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
dates_included = []
dates_excluded = []
overwrite_data = False

# load session info
base_path = Path('Y:/singer/NWBData/UpdateTask/')
spreadsheet_filename = '/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
all_session_info = get_session_info(filename=spreadsheet_filename, animals=animals,
                                    dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

# get output figure info
repo = Repo(search_parent_directories=True)
short_hash = repo.head.object.hexsha[:10]
this_filename = __file__  # to add to metadata to know which script generated the figure

# get bins data
nb_bins = 30  # position
bin_size = 0.5  # seconds

# loop through sessions to accumulate data
intermediate_data_path = Path().absolute().parent.parent / 'results' / 'decoding' / 'group_summary' / 'intermediate_data'
Path(intermediate_data_path).mkdir(parents=True, exist_ok=True)

all_decoding_data = {}
position_bins = {}
decoding_matrix_prob = {}
session_rmse = {}
for dimension in ['x', 'y']:
    # load up the data
    filename = f'decoding_error_group_summary_{dimension}position.pkl'
    if overwrite_data or not Path(intermediate_data_path / filename).is_file():
        all_decoding_data[dimension] = get_decoding_error_summary(base_path, dimension, nb_bins, bin_size, unique_sessions)
        all_decoding_data[dimension].to_pickle(intermediate_data_path / filename)
    else:
        all_decoding_data[dimension] = pd.read_pickle(intermediate_data_path / filename)

    position_bins[dimension] = np.linspace(np.min(all_decoding_data[dimension]['actual_position']),
                                           np.max(all_decoding_data[dimension]['actual_position']), nb_bins + 1)
    bins = pd.cut(all_decoding_data[dimension]['actual_position'], position_bins[dimension])
    decoding_matrix_prob[dimension] = all_decoding_data[dimension]['prob_dist'].groupby(bins).apply(
        lambda x: np.nanmean(np.vstack(x.values), axis=0)).values

    session_rmse[dimension] = all_decoding_data[dimension][['animal', 'session', 'session_rmse']].groupby(['animal', 'session']).mean().reset_index()

# plot data for all sessions
mosaic = """
AABCC
AADEE
FFGHH
FFIJJ
"""
axes = plt.figure(figsize=(20, 18)).subplot_mosaic(mosaic)

locations = {'x': {'left arm': -2, 'home arm': 2, 'right arm': 33},  # actually -1,+1 but add for bins
             'y': {'initial cue': 120.35, 'delay cue': 145.35, 'update cue': 215.35, 'delay2 cue': 250.35, 'choice cue': 285}}

plot_decoding_error_summary(all_decoding_data['x'], decoding_matrix_prob['x'], position_bins['x'], session_rmse['x'],
                            locations['x'], nb_bins, 'x', axes, ['A', 'B', 'C', 'D', 'E'])
plot_decoding_error_summary(all_decoding_data['y'], decoding_matrix_prob['y'], position_bins['y'], session_rmse['y'],
                            locations['y'], nb_bins, 'y', axes, ['F', 'G', 'H', 'I', 'J'])

plt.suptitle(f'Group decoding accuracy - non update trials only', fontsize=20)
plt.tight_layout()

# save figure
figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / 'group_summary'
Path(figure_path).mkdir(parents=True, exist_ok=True)
filename = figure_path / f'decoding_error_summary_position_git{short_hash}.pdf'
plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})
