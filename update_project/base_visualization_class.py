import matplotlib as mpl
import matplotlib.pyplot as plt

from abc import abstractmethod, ABC
from pathlib import Path

from update_project.general.plots import get_color_theme
from update_project.general.virtual_track import UpdateTrack


class BaseVisualizationClass(ABC):
    def __init__(self, data):
        # setup style sheet and matplotlib plotting parameters
        plt.style.use(Path(__file__).parent / 'general' / 'prince-paper.mplstyle')
        self.rcparams = mpl.rcParams
        self.new_line = '\n'

        self.data = data
        self.colors = get_color_theme()
        self.virtual_track = UpdateTrack()

        self.plot_groups = dict(update_type=[['non_update'], ['switch'], ['stay']],
                                turn_type=[[1], [2], [1, 2]],
                                correct=[[0], [1], [0, 1]])
        self.plot_group_comparisons = dict(update_type=[['non_update'], ['switch'], ['stay']],
                                           turn_type=[[1, 2]],
                                           correct=[[0], [1]])
        self.data_comparisons = dict(update_type=dict(update_type=['switch', 'stay', 'non_update'], correct=[1]),
                                     correct=dict(update_type=['switch'], correct=[0, 1]))
        self.plot_group_comparisons_full = dict(update_type=dict(update_type=['switch', 'stay', 'non_update'],
                                                                 turn_type=[1, 2], correct=[1],
                                                                 time_label=['t_update']),
                                                correct=dict(update_type=['switch'], turn_type=[1, 2], correct=[1, 0],
                                                             time_label=['t_update']),
                                                all_update=dict(update_type=['switch', 'stay', 'non_update'],
                                                                turn_type=[1, 2], correct=[1, 0],
                                                                time_label=['t_update']))
        self.label_maps = dict(update_type=dict(switch='switch', stay='stay', non_update='delay only'),
                               correct={1: 'correct', 0: 'incorrect'})

    @abstractmethod
    def plot(self):
        """Child AnalysisInterface classes should override this to plot all the specific figures for that analysis"""
        pass


