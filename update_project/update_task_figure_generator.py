import matplotlib.pyplot as plt

from pathlib import Path
from pynwb import NWBHDF5IO

from update_project.general.results_io import ResultsIO
from update_project.behavior.behavior_analyzer import BehaviorAnalyzer
from update_project.behavior.behavior_visualizer import BehaviorVisualizer
from update_project.choice.choice_analyzer import ChoiceAnalyzer
from update_project.choice.choice_visualizer import ChoiceVisualizer
from update_project.decoding.bayesian_decoder_analyzer import BayesianDecoderAnalyzer
from update_project.decoding.bayesian_decoder_visualizer import BayesianDecoderVisualizer
from update_project.single_units.single_unit_analyzer import SingleUnitAnalyzer
from update_project.single_units.single_unit_visualizer import SingleUnitVisualizer


class UpdateTaskFigureGenerator:
    def __init__(self, analysis, sessions, overwrite=False):
        self.analysis_kwargs = analysis
        self.sessions = sessions
        self.overwrite = overwrite
        self.results_io = ResultsIO(creator_file=__file__, folder_name='manuscript_figures')

        self.analysis_classes = dict(Behavior=BehaviorAnalyzer,
                                     Choice=ChoiceAnalyzer,
                                     Decoder=BayesianDecoderAnalyzer,
                                     SingleUnits=SingleUnitAnalyzer)
        self.visualization_classes = dict(Behavior=BehaviorVisualizer,
                                          Choice=ChoiceVisualizer,
                                          Decoder=BayesianDecoderVisualizer,
                                          SingleUnits=SingleUnitVisualizer)

    def plot_main_figures(self):
        self.plot_figure_1()  # figure 1 - experimental paradigm
        self.plot_figure_2()  # figure 2 - HPC position coding during decision updating
        self.plot_figure_3()  # figure 3 - PFC choice coding during decision updating
        self.plot_figure_4()  # figure 4 - HPC + PFC contributions to accurate decision updating

    def plot_supplemental_figures(self):
        self.plot_supp_figure_1()  # supp figure 1
        self.plot_supp_figure_2()  # supp figure 2
        self.plot_supp_figure_3()  # supp figure 3
        self.plot_supp_figure_4()  # supp figure 4

    def run_analysis_pipeline(self, analysis_to_run, overwrite=False):
        print(f'Running {analysis_to_run} analysis interface')
        session_names = self.sessions.load_session_names()
        analysis_interface = self.analysis_classes[analysis_to_run]
        visualization_interface = self.visualization_classes[analysis_to_run]

        group_data = []
        for name in session_names:
            # load nwb file
            io = NWBHDF5IO(self.sessions.get_session_path(name), 'r')
            nwbfile = io.read()

            # run analysis
            analyzer = analysis_interface(nwbfile=nwbfile,
                                          session_id=self.sessions.get_session_id(name),
                                          **self.analysis_kwargs[analysis_to_run])
            analyzer.run_analysis(overwrite=overwrite)
            session_data = dict(session_id=self.sessions.get_session_id(name),
                                animal=self.sessions.get_animal_id(name),
                                analyzer=analyzer)
            group_data.append(session_data)

        visualizer = visualization_interface(group_data)

        return visualizer

    def plot_figure_1(self, overwrite=False):
        visualizer = self.run_analysis_pipeline(analysis_to_run='Behavior', overwrite=overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 5))
        sfigs = fig.subfigures(nrows=2, ncols=2, width_ratios=[2.25, 1], height_ratios=[1, 1.25])
        axes_top_left = sfigs[0][0].subplots(nrows=1, ncols=1)
        axes_top_right = sfigs[0][1].subplots(nrows=1, ncols=1)
        axes_bottom_left = sfigs[1][0].subplots(nrows=1, ncols=3, sharey=True)
        axes_bottom_right = sfigs[1][1].subplots(nrows=1, ncols=1)

        # plot data
        axes_top_left.axis('off')
        axes_top_left.annotate('Task schematic with example trial types', xycoords='axes fraction', xy=(0.4, 0.5),
                             horizontalalignment='center')
        axes_top_right = visualizer.plot_performance(ax=axes_top_right)  # behavioral performance distributions
        axes_bottom_left = visualizer.plot_trajectories_by_position(ax=axes_bottom_left)  # 20 example view trajectories of each trial type
        sfigs[1][0].suptitle('Example trial trajectories', fontsize=12)
        sfigs[1][0].supxlabel('fraction of track', fontsize=10)
        sfigs[1][0].axes[0].set(ylabel='view angle (degrees)')
        axes_bottom_right = visualizer.plot_trajectories_by_event(ax=axes_bottom_right)  # average trajectories aligned to update

        # add text annotations
        sfig_list = dict(A=sfigs[0][0], B=sfigs[0][1], C=sfigs[1][0], D=sfigs[1][1])
        for label, s in sfig_list.items():
            s.text(0, 0.94, label, weight='bold', fontsize=12, transform=s.transSubfigure)

        # figure saving
        self.results_io.save_fig(fig=fig, filename=f'figure_1', tight_layout=False, results_type='manuscript')

    def plot_figure_2(self):
        visualizer = self.run_analysis_pipeline(analyses_to_run=['Decoder'])  # TODO - ad way to implement kwargs here

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1)
        axes_top = sfigs[0].subplot_mosaic('AABBCCDD')
        axes_middle = sfigs[1].subplots(nrows=1, ncols=2)
        axes_bottom = sfigs[2].subplots(nrows=1, ncols=5)

        axes_top = visualizer.plot_position(ax=axes_top)
        axes_middle = visualizer.plot_trajectories_by_position(ax=axes_middle)
        axes_bottom = visualizer.plot_trajectories_by_event(ax=axes_bottom)

        # figure saving
        self.results_io.save_fig(fig=fig, filename=f'figure_2', tight_layout=False)
