import matplotlib.pyplot as plt
import numpy as np

from git import Repo
from pynwb import NWBHDF5IO
from pathlib import Path

from update_project.session_loader import get_session_info
from plots import show_event_aligned_psth, show_start_aligned_psth
from units import align_by_time_intervals

# set inputs
animals = [29]
dates_included = []
dates_excluded = []  # need to run S20_210511, S20_210519, S25_210909, S28_211118 later, weird spike sorting table issue

# load session info
base_path = Path('Y:/singer/NWBData/UpdateTask/')
spreadsheet_filename = '/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
all_session_info = get_session_info(filename=spreadsheet_filename, animals=animals,
                                    dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

# loop through sessions and run conversion
for name, session in unique_sessions:

    # load file
    session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
    filename = base_path / f'{session_id}.nwb'
    io = NWBHDF5IO(str(filename), 'r')
    nwbfile = io.read()

    # get info for figure saving and labelling
    repo = Repo(search_parent_directories=True)
    short_hash = repo.head.object.hexsha[:10]
    this_filename = __file__  # to add to metadata to know which script generated the figure
    figure_path = Path().absolute().parent.parent / 'results' / 'decoding' / f'{session_id}'
    Path(figure_path).mkdir(parents=True, exist_ok=True)

    # get trial group indices, convert to different numbers for plotting color purposes
    labels = np.array(['left', 'right', 'incorrect', 'correct', 'non-update', 'switch', 'stay'])
    group_inds = {'turn': nwbfile.intervals['trials']['turn_type'][:] - 1,
                  'outcome': nwbfile.intervals['trials']['correct'][:].astype(int) + 2,
                  'update': nwbfile.intervals['trials']['update_type'][:] + 3}

    # get trials to use for different alignment events
    alignment_trial_inds = {'start_time': range(len(nwbfile.intervals['trials'])),
                            't_delay': np.argwhere(~np.isnan(nwbfile.intervals['trials']['t_delay'][:])).squeeze(),
                            't_update': np.argwhere(nwbfile.intervals['trials']['update_type'][:] != 1).squeeze(),
                            't_delay2': np.argwhere(~np.isnan(nwbfile.intervals['trials']['t_delay2'][:])).squeeze(),
                            't_choice_made': range(len(nwbfile.intervals['trials']))}

    # get delay times and use to sort trials
    annotation_times = dict()
    for key in alignment_trial_inds.keys():
        annotation_times[key] = nwbfile.trials[key][:] - nwbfile.trials['start_time'][:]
    annotation_times.pop('start_time')

    sort_index = np.argsort(annotation_times['t_delay'])
    annotations_sorted = {k: v[sort_index] for k, v in annotation_times.items()}
    groups_sorted = {k: v[sort_index] for k, v in group_inds.items()}

    update_trial_inds = np.argwhere(~np.isnan(annotations_sorted['t_update'])).squeeze()
    annotations_update = {k: v[update_trial_inds] for k, v in annotations_sorted.items()}
    groups_update = {k: v[update_trial_inds] for k, v in groups_sorted.items()}

    # make individual neuron psths
    window_long = 25  # seconds
    window_short = 3  # seconds before and after to plot
    for unit_index in range(len(nwbfile.units)):
        # get spikes aligned to event times and sorted
        spikes_aligned_long = align_by_time_intervals(nwbfile.units, unit_index, nwbfile.intervals['trials'],
                                                      start_label='start_time', stop_label='start_time',
                                                      start=0, end=window_long)
        spikes_aligned_sorted = [spikes_aligned_long[ind] for ind in sort_index]
        spikes_aligned_update = [spikes_aligned_sorted[ind] for ind in update_trial_inds]

        spikes_aligned = dict()
        for label, trial_mask in alignment_trial_inds.items():
            spikes_aligned[label] = align_by_time_intervals(nwbfile.units, unit_index, nwbfile.intervals['trials'],
                                                            start_label=label, stop_label=label,
                                                            start=-window_short, end=window_short,
                                                            rows_select=trial_mask)

        for group_name, group in group_inds.items():
            # plot the data
            mosaic = """
                    AAAAA
                    BBBBB
                    CDEFG
                    HIJKL
                    """
            ax_dict = plt.figure(constrained_layout=True, figsize=(20, 16)).subplot_mosaic(mosaic)

            # plot psth aligned at start for all trials
            show_start_aligned_psth(spikes_aligned_sorted, annotations_sorted, groups_sorted[group_name], window_long, labels,
                                    ax_dict['A'])
            ax_dict['A'].set_title(f'PSTH for unit {unit_index} session {session_id} - {group_name}', fontsize=16)

            # plot psth aligned at start for update only trials
            show_start_aligned_psth(spikes_aligned_update, annotations_update, groups_update[group_name], window_long, labels,
                                    ax_dict['B'])
            ax_dict['B'].set_ylabel('trials (update only)')

            # plot psth aligned at event times
            show_event_aligned_psth(spikes_aligned, window_short, group, alignment_trial_inds, labels, ax_dict=ax_dict,
                                    ax_start_key='C')
            ax_dict['C'].set_ylabel('Firing rate (spikes/s)')
            ax_dict['H'].set_ylabel('trials')

            # save figure
            filename = figure_path / f'psth_{group_name}_unit{unit_index}_git{short_hash}.pdf'
            plt.savefig(filename, dpi=300, metadata={'Creator': this_filename})

        plt.close('all')

    io.close()
