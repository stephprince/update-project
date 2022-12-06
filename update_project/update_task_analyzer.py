import matplotlib.pyplot as plt

from pathlib import Path
from pynwb import NWBHDF5IO

from update_project.general.results_io import ResultsIO
from update_project.behavior.behavior_analysis_interface import BehaviorAnalysisInterface  # TODO: fix and refactor these
from update_project.decoding.bayesian_decoder_analysis_interface import BayesianDecoderAnalysisInterface
from update_project.dynamic_choice.choice_analysis_interface import ChoiceAnalysisInterface
from update_project.single_units.single_unit_analysis_interface import SingleUnitAnalysisInterface


class UpdateTaskAnalyzer:
    """Primary analysis class for the update task data"""

    def __init__(self, analysis, sessions, overwrite=False):
        self.analysis_kwargs = analysis
        self.sessions = sessions
        self.overwrite = overwrite
        self.results_io = ResultsIO(creator_file=__file__, folder_name='manuscript_figures')

        self.analysis_interface_classes = dict(Behavior=BehaviorAnalysisInterface,
                                               Choice=ChoiceAnalysisInterface,
                                               Decoder=BayesianDecoderAnalysisInterface,
                                               SingleUnits=SingleUnitAnalysisInterface)
        self.visualization_interface_classes = dict(Behavior=BehaviorVisualizationInterface,
                                                    Choice=ChoiceAnalysisInterface,
                                                    Decoder=BayesianDecoderVisualizationInterface,
                                                    SingleUnits=SingleUnitVisualizationInterface)

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

    def run_analysis_pipeline(self, analyses_to_run=None, overwrite=False):
        session_names = self.sessions.load_session_names()
        analyses_to_run = analyses_to_run or list(self.analysis_interface_classes.keys())  # if none specified, run all

        for analysis_name in analyses_to_run:
            analysis_interface = self.analysis_interface_classes[analysis_name]
            visualizer_interface = self.visualization_interface_classes[analysis_name]
            print(f'Running analysis for the {analysis_name} analysis interface')

            group_data = []
            for name in session_names:
                # load nwb file
                io = NWBHDF5IO(self.sessions.get_session_path(name), 'r')
                nwbfile = io.read()

                # run analysisZZZ
                analyzer = analysis_interface.run_analysis(nwbfile=nwbfile,
                                                           overwrite=overwrite,
                                                           **self.analysis_kwargs[analysis_name])
                session_data = dict(session_id=self.sessions.get_session_id(name),
                                    animal=self.sessions.get_animal_id(name),
                                    analyzer=analyzer)
                group_data.append(session_data)

            visualizer = visualizer_interface(group_data)

        return visualizer

    def plot_figure_1(self):
        visualizer = self.run_analysis_pipeline(analyses_to_run=['Behavior'])

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
        sfigs = fig.subfigures(nrows=3, ncols=1)
        axes_top = sfigs[0].subplot_mosaic('.....B')
        axes_middle = sfigs[1].subplots(nrows=1, ncols=5)
        axes_bottom = sfigs[2].subplots(nrows=1, ncols=5)

        axes_top['B'] = visualizer.plot_performance(ax=axes_top['B'])
        axes_middle = visualizer.plot_trajectories_by_position(ax=axes_middle)
        axes_bottom = visualizer.plot_trajectories_by_event(ax=axes_bottom)

        # figure saving
        self.results_io.save_fig(fig=fig, filename=f'figure_1', tight_layout=False)

    def plot_figure_2(self):
        visualizer = self.run_analysis_pipeline(analyses_to_run=['Decoder'])

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1)
        axes_top = sfigs[0].subplot_mosaic('AABBCCDD')
        axes_middle = sfigs[1].subplots(nrows=1, ncols=2)
        axes_bottom = sfigs[2].subplots(nrows=1, ncols=5)

        axes_top['B'] = visualizer.plot_performance(ax=axes_top['B'])
        axes_middle = visualizer.plot_trajectories_by_position(ax=axes_middle)
        axes_bottom = visualizer.plot_trajectories_by_event(ax=axes_bottom)

        # figure saving
        self.results_io.save_fig(fig=fig, filename=f'figure_1', tight_layout=False)
