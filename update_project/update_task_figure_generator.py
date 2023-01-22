import matplotlib.pyplot as plt

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
    def __init__(self, sessions, overwrite=False):
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
        # self.plot_figure_1()  # figure 1 - experimental paradigm
        # self.plot_figure_2()  # figure 2 - HPC position coding during decision updating
        # self.plot_figure_3()  # figure 3 - PFC choice coding during decision updating
        self.plot_figure_4()  # figure 4 - HPC + PFC contributions to accurate decision updating

    def plot_supplemental_figures(self):
        # self.plot_supp_figure_1()  # supp figure 1 - behavioral supplement
        # self.plot_supp_figure_2()  # supp figure 2 - HPC position coding supplement
        self.plot_supp_figure_3()  # supp figure 3 - HPC position coding supplement - session/animal breakdown
        self.plot_supp_figure_4()  # supp figure 4 - PFC position coding supplement
        self.plot_supp_figure_5()  # supp figure 5 - PFC choice coding supplement
        self.plot_supp_figure_6()  # supp figure 6 - HPC + PFC supplement
        self.plot_supp_figure_7()  # supp figure 6 - HPC + PFC supplement

    def run_analysis_pipeline(self, analysis_to_run, analysis_kwargs=dict(), overwrite=False):
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
                                          **analysis_kwargs)
            analyzer.run_analysis(overwrite=overwrite)
            session_data = dict(session_id=self.sessions.get_session_id(name),
                                animal=self.sessions.get_animal_id(name),
                                analyzer=analyzer)
            group_data.append(session_data)

        visualizer = visualization_interface(group_data)

        return visualizer

    def plot_figure_1(self):
        visualizer = self.run_analysis_pipeline(analysis_to_run='Behavior', overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
        sfigs = fig.subfigures(nrows=2, ncols=2, width_ratios=[2.25, 1], height_ratios=[1, 1.1])

        # plot data
        sfigs[0][0] = self.plot_placeholder(sfigs[0][0], text='Task schematic with example trial types')
        sfigs[0][1] = visualizer.plot_performance(sfigs[0][1])  # behavioral performance distributions
        sfigs[1][0] = visualizer.plot_trajectories_by_position(sfigs[1][0])  # example  trajectories of each trial type
        sfigs[1][1] = visualizer.plot_trajectories_by_event(sfigs[1][1])  # average trajectories aligned to update

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_1', tight_layout=False, results_type='manuscript')

    def plot_figure_2(self):
        visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                analysis_kwargs=dict(features=['y_position'],
                                                                     params=dict(region=['CA1'])),
                                                overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 5))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        sfigs_row3 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])

        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='Recording setup + position schematic')
        sfigs_row0[1] = visualizer.plot_decoding_output_heatmap(sfigs_row0[1])
        sfigs[1] = visualizer.plot_goal_coding(sfigs[1])
        sfigs_row3[0] = visualizer.plot_goal_coding_stats(sfigs_row3[0], tags='CA1_position')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_2', tight_layout=False, results_type='manuscript')

    def plot_figure_3(self):
        visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                analysis_kwargs=dict(features=['choice'],
                                                                     params=dict(region=['PFC'])),
                                                overwrite=self.overwrite)
        choice_visualizer = self.run_analysis_pipeline(analysis_to_run='Choice', overwrite=False)  # do not overwrite choice data

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1.25])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2)
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2)
        sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=2)

        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='LSTM network schematic')
        sfigs_row0[1] = choice_visualizer.plot_choice_commitment(sfigs_row0[1])  # TODO - check which trial types included
        sfigs_row1[0] = self.plot_placeholder(sfigs[1][0], text='Recording setup + choice schematic')
        sfigs_row1[1] = visualizer.plot_decoding_output_heatmap(sfigs_row1[1])
        sfigs[2] = visualizer.plot_goal_coding(sfigs[2])
        sfigs_row3[0] = visualizer.plot_goal_coding_stats(sfigs_row3[0], tags='PFC_choice')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_3', tight_layout=False, results_type='manuscript')

    def plot_figure_4(self):
        # pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
        #                                             analysis_kwargs=dict(features=['choice'],
        #                                                                  params=dict(region=['PFC'])),
        #                                             overwrite=self.overwrite)
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6))
        sfigs = fig.subfigures(nrows=4, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1.5, 1])
        #
        # sfigs[0][0] = self.plot_placeholder(sfigs[0][0], text='Region schematic with HPC')
        # sfigs[0][1] = hpc_visualizer.plot_goal_coding(sfigs[0][1], comparison='correct')
        # sfigs[1][0] = self.plot_placeholder(sfigs[1][0], text='Region schematic with PFC')
        # sfigs[1][1] = pfc_visualizer.plot_goal_coding(sfigs[1][1], comparison='correct')
        # sfigs[2][0] = hpc_visualizer.plot_goal_coding_stats(sfigs[2][0], comparison='correct', title='hippocampal',
        #                                                     tags='HPC_position_accuracy')
        # sfigs[2][1] = pfc_visualizer.plot_goal_coding_stats(sfigs[2][1], comparison='correct', title='prefrontal',
        #                                                     tags='PFC_choice_accuracy')
        sfigs[3][0] = hpc_visualizer.plot_goal_coding_prediction(sfigs[3][0], comparison='all_update')
        # sfigs[3][1] = pfc_visualizer.plot_goal_coding_prediction(sfigs[3][1], comparison='all_update')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_4', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_1(self):
        visualizer = self.run_analysis_pipeline(analysis_to_run='Behavior', overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
        sfigs = fig.subfigures(nrows=2, ncols=2, width_ratios=[1.5, 1], height_ratios=[1, 3])

        # plot data
        sfigs[0][0] = visualizer.plot_event_durations(sfigs[0][0])  # average trajectories aligned to update
        sfigs[0][1] = visualizer.plot_performance_by_animals(sfigs[0][1])  # behavioral performance distributions
        sfigs[1][0] = visualizer.plot_performance_by_delay(sfigs[1][0])  # example trajectories of each trial type
        sfigs[1][1] = visualizer.plot_trajectories_all_metrics(sfigs[1][1])  # example trajectories of each trial type

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_1', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_2(self):
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=4, ncols=1, height_ratios=[1, 1.5, 1.5, 1])
        sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=2)

        # plot data
        sfigs[0] = self.plot_placeholder(sfigs[0], text='probe tract visualization + histological images')
        sfigs[1] = hpc_visualizer.plot_tuning_curves(sfigs[1], title='hippocampal position')
        sfigs[2] = hpc_visualizer.plot_decoding_validation(sfigs[2])
        sfigs_row3[0] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row3[0], update_type='stay')
        sfigs_row3[1] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row3[1], update_type='non_update')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_2', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_3(self):
        encoder_trials_all = dict(update_type=[1, 2, 3], correct=[0, 1], maze_id=[3, 4])
        hpc_visualizer_encode_all = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                               analysis_kwargs=dict(features=['y_position'],
                                                                                    params=dict(region=['CA1'],
                                                                                                encoder_trial_types=encoder_trials_all,
                                                                                                decoder_test_size=0.5)),
                                                               overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1])

        # plot data
        sfigs[0][0] = hpc_visualizer_encode_all.plot_decoding_output_heatmap(sfigs[1][0])
        sfigs[0][1] = hpc_visualizer_encode_all.plot_goal_coding(sfigs[1][1])
        sfigs[1][0] = hpc_visualizer_encode_all.plot_goal_coding_stats(sfigs[2][0], tags='CA1_position',
                                                                       time_window=(0,1))

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_3', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_4(self):
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 3])

        # plot data
        sfigs[0] = hpc_visualizer.plot_goal_coding(sfigs[0], groups='session_id', heatmap=True)  # results by session
        sfigs[1] = hpc_visualizer.plot_goal_coding(sfigs[1], groups='animal', heatmap=True) # results by animal
        sfigs[2] = hpc_visualizer.plot_goal_coding(sfigs[2], groups='animal')  # results by animal non-heatmap

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_4', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_5(self):
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)
        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=4, ncols=2, width_ratios=[1.25, 1], height_ratios=[1.5, 1, 1, 1])

        sfigs[1][0] = pfc_visualizer.plot_tuning_curves(sfigs[1][0], title='prefrontal position')
        sfigs[1][1] = pfc_visualizer.plot_decoding_validation(sfigs[1][1])
        sfigs[2][0] = self.plot_placeholder(sfigs[2][0], text='PFC schematic')
        sfigs[2][1] = pfc_visualizer.plot_decoding_output_heatmap(sfigs[2][1])
        sfigs[3][0] = pfc_visualizer.plot_goal_coding(sfigs[3][0], ylim=(0.0, 0.18825))  # pulled upper lim from HPC plot
        sfigs[3][1] = pfc_visualizer.plot_goal_coding_stats(sfigs[3][1], tags='PFC_position')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_5', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_6(self):
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['x_position'],
                                                                         params=dict(region=['HPC'])),
                                                    overwrite=self.overwrite)
        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=4, ncols=2, width_ratios=[1.25, 1], height_ratios=[1.5, 1, 1, 1])

        sfigs[1][0] = hpc_visualizer.plot_tuning_curves(sfigs[1][0], title='hippocampal lateral position')
        sfigs[1][1] = hpc_visualizer.plot_decoding_validation(sfigs[1][1])
        sfigs[2][0] = self.plot_placeholder(sfigs[2][0], text='HPC schematic')
        sfigs[2][1] = hpc_visualizer.plot_decoding_output_heatmap(sfigs[2][1])
        sfigs[3][0] = hpc_visualizer.plot_goal_coding(sfigs[3][0])  # pulled upper lim from HPC plot
        sfigs[3][1] = hpc_visualizer.plot_goal_coding_stats(sfigs[3][1], tags='HPC_lateral_position')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_6', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_7(self):
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['choice'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1, height_ratios=[1.5, 1.5, 1])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2)

        # plot data
        sfigs[0] = pfc_visualizer.plot_tuning_curves(sfigs[0], title='prefrontal cortex choice')
        sfigs[1] = pfc_visualizer.plot_decoding_validation(sfigs[1])
        sfigs_row2[0] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row2[0], update_type='stay')  # stay and non-update trials
        sfigs_row2[1] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row2[1], update_type='non_update')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_7', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_8(self):
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['choice'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 3])

        # plot data
        sfigs[0] = pfc_visualizer.plot_goal_coding(sfigs[0], groups='session_id', heatmap=True)  # results by session
        sfigs[1] = pfc_visualizer.plot_goal_coding(sfigs[1], groups='animal', heatmap=True)  # results by animal
        sfigs[2] = pfc_visualizer.plot_goal_coding(sfigs[2], groups='animal')  # results by animal non-heatmap

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_8', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_9(self):
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['choice'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6))
        sfigs = fig.subfigures(nrows=3, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1, 1.5])

        sfigs[0][0] = hpc_visualizer.plot_goal_coding(sfigs[0][0], comparison='correct')
        sfigs[0][1] = pfc_visualizer.plot_goal_coding(sfigs[0][1], comparison='correct')
        sfigs[1][0] = hpc_visualizer.plot_goal_coding_stats(sfigs[1][0], comparison='correct', title='hippocampal', tags='PFC_position_accuracy')
        sfigs[1][1] = pfc_visualizer.plot_goal_coding_stats(sfigs[1][1], comparison='correct', title='prefrontal', tags='CA1_choice_accuracy')
        sfigs[2][0] = self.plot_placeholder(sfigs[1][1], text='correct vs. incorrect by session')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_9', tight_layout=False, results_type='manuscript')

    @staticmethod
    def plot_placeholder(sfig, text):
        axes = sfig.subplots(nrows=1, ncols=1)  # TODO - move this part to inside the visualization
        axes.axis('off')
        axes.annotate(text, xycoords='axes fraction', xy=(0.4, 0.5), horizontalalignment='center')

        return sfig

    @staticmethod
    def add_panel_labels(sfigs):
        panel_list = {k: v for k, v in zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', sfigs.flatten())}
        for label, s in panel_list.items():
            s.text(0, 0.94, label, weight='bold', fontsize=12, transform=s.transSubfigure)
