import matplotlib.pyplot as plt

from pynwb import NWBHDF5IO

from update_project.general.results_io import ResultsIO
from update_project.behavior.behavior_analyzer import BehaviorAnalyzer
from update_project.behavior.behavior_visualizer import BehaviorVisualizer
from update_project.choice.choice_analyzer import ChoiceAnalyzer
from update_project.choice.choice_visualizer import ChoiceVisualizer
from update_project.decoding.bayesian_decoder_analyzer import BayesianDecoderAnalyzer
from update_project.example_trials.example_trial_analyzer import ExampleTrialAnalyzer
from update_project.decoding.bayesian_decoder_visualizer import BayesianDecoderVisualizer
from update_project.single_units.single_unit_analyzer import SingleUnitAnalyzer
from update_project.single_units.single_unit_visualizer import SingleUnitVisualizer
from update_project.example_trials.example_trial_visualizer import ExampleTrialVisualizer


class UpdateTaskFigureGenerator:
    def __init__(self, sessions, overwrite=False):
        self.sessions = sessions
        self.overwrite = overwrite
        self.results_io = ResultsIO(creator_file=__file__, folder_name='manuscript_figures')

        self.analysis_classes = dict(Behavior=BehaviorAnalyzer,
                                     Choice=ChoiceAnalyzer,
                                     Decoder=BayesianDecoderAnalyzer,
                                     SingleUnits=SingleUnitAnalyzer,
                                     Examples=ExampleTrialAnalyzer)
        self.visualization_classes = dict(Behavior=BehaviorVisualizer,
                                          Choice=ChoiceVisualizer,
                                          Decoder=BayesianDecoderVisualizer,
                                          SingleUnits=SingleUnitVisualizer,
                                          Examples=ExampleTrialVisualizer)

    def plot_main_figures(self):
        self.plot_figure_1()              # figure 1 - experimental paradigm + behavior supplement
        self.plot_figure_2()              # figure 2 - HPC/PFC position coding + decoding validation + animal breakdown
        self.plot_figure_3()              # figure 3 - HPC theta modulation
        self.plot_figure_4()              # figure 4 - PFC/HPC choice coding + decoding validation + animal breakdown
        self.plot_figure_5_and_6_and_7()  # figure 5 - HPC + PFC stay + accuracy prediction + choice commitment + supp

    def plot_supplemental_figures(self):
        # rest of supplemental figures are plotted within their corresponding main figure functions to minimize compute
        self.plot_supp_figure_all_trials()  # supp figure 4 - HPC position coding - all trial types supplement
        self.plot_supp_figure_lateral_position()  # supp figure 5 - HPC/PFC lateral position coding supplement

    def plot_demo_figure(self, session='S25_210913'):
        behavior_visualizer = self.run_analysis_pipeline(analysis_to_run='Behavior', overwrite=self.overwrite)

        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1, height_ratios=[1, 1.1, 2.5])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[2.25, 1])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[2.25, 1])

        # plot data
        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='Task schematic with example trial types')
        sfigs_row0[1] = behavior_visualizer.plot_performance(sfigs_row0[1], tags='demo_prop_correct')  # performance
        sfigs_row1[0] = behavior_visualizer.plot_trajectories_by_position(sfigs_row1[0], example_session=session)  # example  trajectories
        sfigs_row1[1] = behavior_visualizer.plot_trajectories_by_event(sfigs_row1[1])  # average trajectories

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'demo_figure', tight_layout=False, results_type='manuscript')

    def run_analysis_pipeline(self, analysis_to_run, analysis_kwargs=dict(), visualization_kwargs=dict(),
                              session_names=None, overwrite=False):
        print(f'Running {analysis_to_run} analysis interface')
        session_names = session_names or self.sessions.load_session_names()
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

        visualizer = visualization_interface(group_data,
                                             **visualization_kwargs)

        return visualizer

    def plot_figure_1(self, with_supplement=True):
        behavior_visualizer = self.run_analysis_pipeline(analysis_to_run='Behavior', overwrite=self.overwrite)
        example_visualizer = self.run_analysis_pipeline(analysis_to_run='Examples', overwrite=self.overwrite,
                                                        session_names=[('S', 29, 211118)])

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1, height_ratios=[1, 1.1, 2.5])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[2.25, 1])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[2.25, 1])

        # plot data
        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='Task schematic with example trial types')
        sfigs_row0[1] = behavior_visualizer.plot_performance(sfigs_row0[1], tags='fig1_prop_correct')  # performance
        sfigs_row1[0] = behavior_visualizer.plot_trajectories_by_position(sfigs_row1[0])  # example  trajectories
        sfigs_row1[1] = behavior_visualizer.plot_trajectories_by_event(sfigs_row1[1])  # average trajectories
        sfigs[2] = example_visualizer.plot_behavior_trial(sfigs[2], trial_id=175)  # or example 155

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_1', tight_layout=False, results_type='manuscript')

        if with_supplement:
            # figure structure
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
            sfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[1, 3])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1.5, 1])
            sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1.5])

            # plot data
            sfigs_row0[0] = behavior_visualizer.plot_event_durations(sfigs_row0[0])  # trajectories aligned to update
            sfigs_row0[1] = behavior_visualizer.plot_performance_by_animals(sfigs_row0[1])  # behavioral performance
            sfigs_row1[0] = behavior_visualizer.plot_performance_by_delay(sfigs_row1[0])  # example trajectories
            sfigs_row1[1] = behavior_visualizer.plot_trajectories_all_metrics(sfigs_row1[1])  # example trajectories

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_1', tight_layout=False, results_type='manuscript')

    def plot_figure_2(self, with_supplement=True):
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)
        example_visualizer = self.run_analysis_pipeline(analysis_to_run='Examples', overwrite=self.overwrite,
                                                        session_names=[('S', 25, 210910)]) # ('S', 29, 211118)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=4, ncols=1, height_ratios=[1.5, 1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 4])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=4, width_ratios=[1, 2, 2, 2])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=3, width_ratios=[1, 4, 2])
        sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=4, width_ratios=[1, 2, 2, 2])

        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='Recording setup + position schematic')
        sfigs_row0[1] = example_visualizer.plot_decoding_trial(sfigs_row0[1], region='CA1', trial_id=231) #154, 207, 231, 168
        sfigs_row1[0] = self.plot_placeholder(sfigs_row1[0], text='Position schematic')
        sfigs_row1[1] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row1[1], update_type='switch')
        sfigs_row1[2] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row1[2], update_type='non_update')
        sfigs_row1[3] = self.plot_placeholder(sfigs_row1[3], text='')
        sfigs_row2[0] = self.plot_placeholder(sfigs_row2[0], text='Trial schematic')
        sfigs_row2[1] = hpc_visualizer.plot_goal_coding(sfigs_row2[1], ylim=(0.0325, 0.18), tags='fig2_CA1_position')
        sfigs_row2[2] = hpc_visualizer.plot_goal_coding_stats(sfigs_row2[2], ylim=(0.05, 0.1625), tags='fig2_CA1_position')
        sfigs_row3[0] = self.plot_placeholder(sfigs_row3[0], text='PFC schematic')
        sfigs_row3[1] = pfc_visualizer.plot_goal_coding(sfigs_row3[1], ylim=(0.0325, 0.18), update_type=['switch'],
                                                        tags='fig2_PFC_position')  # pulled upper lim from HPC plot
        sfigs_row3[2] = pfc_visualizer.plot_goal_coding_stats(sfigs_row3[2], ylim=(0.05, 0.1625), tags='fig2_PFC_position')

        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_2', tight_layout=False, results_type='manuscript')

        if with_supplement:
            ##### supp figure - local and initial cue quantification
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 6))
            sfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[1, 1])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=3, width_ratios=[1, 1, 1])
            sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

            sfigs_row0[0] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row0[0], update_type='switch')
            sfigs_row0[1] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row0[1], update_type='non_update')
            sfigs_row1[0] = hpc_visualizer.plot_goal_coding(sfigs_row1[0], tags='fig2_CA1_position',
                                                            other_zones=['local', 'central', 'original'])
            sfigs_row1[1] = hpc_visualizer.plot_goal_coding_stats(sfigs_row1[1], tags='fig2_CA1_position',
                                                                  other_zones=['local', 'central', 'original'])

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'figure_local_initial', tight_layout=False, results_type='manuscript')

            ##### supp figure 2
            n_sfig = 2
            hpc_visualizer_long_window = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                                    analysis_kwargs=dict(features=['y_position'],
                                                                                         params=dict(region=['CA1'])),
                                                                    visualization_kwargs=dict(window=10),
                                                                    overwrite=self.overwrite)

            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=3, ncols=1, height_ratios=[1, 2, 2])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
            sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])

            # plot data
            sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='probe tract visualization')
            sfigs_row0[1] = self.plot_placeholder(sfigs_row0[1], text='histological images')
            sfigs_row1[0] = hpc_visualizer.plot_tuning_curves(sfigs_row1[0], title='hippocampal position')
            sfigs_row1[1] = hpc_visualizer.plot_decoding_validation(sfigs_row1[1], tags=f'sfig{n_sfig}_CA1_position')
            sfigs[2] = hpc_visualizer_long_window.plot_decoding_output_heatmap(sfigs[2], update_type='switch')

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False, results_type='manuscript')
            
            ##### supp figure 3
            n_sfig = 3
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[1.5, 1])
            sfigs[0] = hpc_visualizer.plot_goal_coding(sfigs[0], groups='animal',
                                                       tags=f'sfig{n_sfig}_CA1_position_by_animal')
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False,
                                     results_type='manuscript')

            ##### supp figure 4
            n_sfig = 4
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])
            sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 4])
            sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 4])
            sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

            # plot data
            sfigs_row0[0] = pfc_visualizer.plot_tuning_curves(sfigs_row0[0], title='prefrontal  position')
            sfigs_row0[1] = pfc_visualizer.plot_decoding_validation(sfigs_row0[1], tags=f'sfig{n_sfig}_PFC_position')
            sfigs_row1[0] = self.plot_placeholder(sfigs_row1[0], text='PFC schematic')
            sfigs_row1[1] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row1[1])
            sfigs_row2[1] = pfc_visualizer.plot_goal_coding(sfigs_row2[1], ylim=(0.0, 0.18825),
                                                            tags=f'sfig{n_sfig}_PFC_position')  # pulled upper lim from HPC plot
            sfigs_row3[0] = pfc_visualizer.plot_goal_coding_stats(sfigs_row3[0], tags=f'sfig{n_sfig}_PFC_position')

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False, results_type='manuscript')

            ##### supp figure chance (won't save but just use to look at prob over chance stats
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
            sfigs = hpc_visualizer.plot_goal_coding_stats(sfigs, prob_value='prob_over_chance', other_zones=['central'],
                                                          tags='sfigchance_CA1_position')
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_chance_hpc', tight_layout=False,
                                     results_type='manuscript')

    def plot_figure_3(self, with_supplement=True):
        visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                analysis_kwargs=dict(features=['y_position'],
                                                                     params=dict(region=['CA1'],
                                                                                 decoder_bin_size=0.025)),
                                                overwrite=self.overwrite)
        example_visualizer = self.run_analysis_pipeline(analysis_to_run='Examples', overwrite=self.overwrite,
                                                        session_names=[('S', 29, 211118)])

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=2, ncols=1, width_ratios=[1], height_ratios=[1.5, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 5])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=4, width_ratios=[1, 1, 1, 2])

        sfigs_row0[0] = visualizer.plot_phase_reference(sfigs_row0[0])
        sfigs_row0[1] = example_visualizer.plot_theta_trace(sfigs_row0[1])
        sfigs_row1[0] = visualizer.plot_theta_phase_modulation(sfigs_row1[0], update_type=['switch'],
                                                               time=['pre', 'post'])
        sfigs_row1[1] = visualizer.plot_theta_phase_modulation(sfigs_row1[1], update_type=['switch'],
                                                               time=['pre', 'post'])
        sfigs_row1[2] = visualizer.plot_theta_phase_stats(sfigs_row1[2], type='pre_vs_post', update_type=['switch'],
                                                          tags='fig3_CA1_theta_pre_vs_post', divider='half')
        sfigs_row1[2] = visualizer.plot_theta_phase_stats(sfigs_row1[2], type='modulation', update_type=['switch'],
                                                          tags='fig3_CA1_theta_trial_types', divider='half')
        sfigs_row1[3] = visualizer.plot_theta_phase_stats(sfigs_row1[3], type='phase_modulation',
                                                          update_type=['switch'],
                                                          tags='fig3_CA1_theta_phase_mod', divider='half')
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_3', tight_layout=False, results_type='manuscript')

        if with_supplement:
            n_sfig = 5
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 4))
            sfigs = fig.subfigures(nrows=1, ncols=4, width_ratios=[1, 1, 1, 2])

            sfigs[0] = visualizer.plot_theta_phase_modulation(sfigs[0], update_type=['non_update'],
                                                              time=['pre', 'post'])
            sfigs[1] = visualizer.plot_theta_phase_modulation(sfigs[1], update_type=['non_update'],
                                                              time=['pre', 'post'])
            sfigs[2] = visualizer.plot_theta_phase_stats(sfigs[2], type='modulation', update_type=['non_update'],
                                                              tags=f'sfig{n_sfig}_CA1_theta', divider='half')
            sfigs[3] = visualizer.plot_theta_phase_stats(sfigs[3], type='phase_modulation',
                                                              update_type=['non_update'], ylim=(0.11, 0.15),
                                                              tags=f'sfig{n_sfig}_CA1_theta_phase_mod', divider='half')

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False,
                                     results_type='manuscript')

    def plot_figure_4(self, with_supplement=True):
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                analysis_kwargs=dict(features=['choice'],
                                                                     params=dict(region=['PFC'])),
                                                overwrite=self.overwrite)
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['choice'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)
        choice_visualizer = self.run_analysis_pipeline(analysis_to_run='Choice', overwrite=False)  # don't overwrite
        example_visualizer = self.run_analysis_pipeline(analysis_to_run='Examples', overwrite=False,
                                                        session_names=[('S', 29, 211118)],
                                                        analysis_kwargs=dict(feature='choice'))

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=5, ncols=1, width_ratios=[1], height_ratios=[1, 2, 1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2)
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 4])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=4, width_ratios=[1, 2, 2, 2])
        sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=3, width_ratios=[1, 4, 2])
        sfigs_row4 = sfigs[4].subfigures(nrows=1, ncols=4, width_ratios=[1, 2, 2, 2])

        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='LSTM network schematic')
        sfigs_row0[1] = choice_visualizer.plot_choice_commitment(sfigs_row0[1])  # TODO - check which trial types included
        sfigs_row1[0] = self.plot_placeholder(sfigs_row1[0], text='Recording setup + choice schematic')
        sfigs_row1[1] = example_visualizer.plot_decoding_trial(sfigs_row1[1], region='PFC')
        sfigs_row2[0] = self.plot_placeholder(sfigs_row2[0], text='Choice schematic')
        sfigs_row2[1] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row2[1], feat='choice')
        sfigs_row2[2] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row2[2], feat='choice', update_type='non_update')
        sfigs_row2[3] = self.plot_placeholder(sfigs_row2[3], text='')
        sfigs_row3[0] = self.plot_placeholder(sfigs_row3[0], text='Trial schematic')
        sfigs_row3[1] = pfc_visualizer.plot_goal_coding(sfigs_row3[1], ylim=(0.06, 0.25), with_velocity=True, tags='fig4_PFC_choice',)
        sfigs_row3[2] = pfc_visualizer.plot_goal_coding_stats(sfigs_row3[2], tags='fig4_PFC_choice',)

        sfigs_row4[0] = self.plot_placeholder(sfigs_row4[0], text='CA1 schematic')
        sfigs_row4[1] = hpc_visualizer.plot_goal_coding(sfigs_row4[1], ylim=(0.06, 0.25), tags='fig4_CA1_choice',
                                                        update_type=['switch'])
        sfigs_row4[2] = hpc_visualizer.plot_goal_coding_stats(sfigs_row4[2], tags='fig4_CA1_choice',
                                                              time_window=(0, 2))
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_4', tight_layout=False, results_type='manuscript')

        if with_supplement:
            ##### supp figure 6
            n_sfig = 6
            pfc_visualizer_long_window = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                                    analysis_kwargs=dict(features=['choice'],
                                                                                         params=dict(region=['PFC'])),
                                                                    visualization_kwargs=dict(window=10),
                                                                    overwrite=self.overwrite)

            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=3, ncols=1, height_ratios=[1, 1, 2])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])
            sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 1.5])

            # plot data
            sfigs_row0[0] = pfc_visualizer.plot_tuning_curves(sfigs_row0[0], title='prefrontal cortex choice')
            sfigs_row0[1] = pfc_visualizer.plot_decoding_validation(sfigs_row0[1], tags=f'sfig{n_sfig}_PFC_choice')
            sfigs[1] = pfc_visualizer_long_window.plot_decoding_output_heatmap(sfigs[1], feat='choice',
                                                                               update_type='switch')
            sfigs_row2[0] = pfc_visualizer.plot_motor_timing(sfigs_row2[0], tags=f'sfig{n_sfig}_pfc_motor_timing')

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False, results_type='manuscript')

            ##### supp figure 7
            n_sfig = 7
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=1, ncols=2, width_ratios=[1.5, 1])
            sfigs[0] = pfc_visualizer.plot_goal_coding(sfigs[0], groups='animal', tags=f'sfig{n_sfig}_pfc_by_animal')
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False,
                                     results_type='manuscript')

            ##### supp figure 8
            n_sfig = 8
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=3, ncols=1, height_ratios=[1, 1, 1])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])
            sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
            sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

            # plot data
            sfigs_row0[0] = hpc_visualizer.plot_tuning_curves(sfigs_row0[0], title='hippocampal choice')
            sfigs_row0[1] = hpc_visualizer.plot_decoding_validation(sfigs_row0[1], tags=f'sfig{n_sfig}_CA1_choice')
            sfigs_row1[0] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row1[0])
            sfigs_row1[1] = hpc_visualizer.plot_goal_coding(sfigs_row1[1], tags=f'sfig{n_sfig}_CA1_choice',
                                                            update_type=['switch', 'non_update'])
            sfigs_row2[0] = hpc_visualizer.plot_goal_coding_stats(sfigs_row2[0], tags=f'sfig{n_sfig}_CA1_choice',
                                                                  time_window=(0, 2))

            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False, results_type='manuscript')


            #### supp figure  chance (won't save but just use to look at prob over chance stats
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=1, ncols=1, width_ratios=[1], height_ratios=[1])
            sfigs = pfc_visualizer.plot_goal_coding_stats(sfigs, prob_value='prob_over_chance',
                                                          tags='sfig_chance_PFC_choice')
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_chance_pfc', tight_layout=False,
                                     results_type='manuscript')

    def plot_figure_5_and_6_and_7(self, with_supplement=True):
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['choice'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['y_position'],
                                                                         params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)
        behavior_visualizer = self.run_analysis_pipeline(analysis_to_run='Behavior', overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=4, width_ratios=[1, 1, 1, 1])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])

        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='Task schematic with example stay trial')
        sfigs_row0[1] = behavior_visualizer.plot_performance(sfigs_row0[1], tags='fig5_performance',
                                                             update_type=['stay_update'])
        sfigs_row0[2] = behavior_visualizer.plot_trajectories_by_position(sfigs_row0[2], update_type=['stay'],
                                                                          example_animal=29,
                                                                          example_session='S29_211118')
        sfigs_row0[3] = behavior_visualizer.plot_trajectories_by_event(sfigs_row0[3], update_type=['stay_update'])
        sfigs_row1[0] = hpc_visualizer.plot_goal_coding(sfigs_row1[0], tags='fig5_HPC_position', update_type=['stay'],)
        sfigs_row1[1] = hpc_visualizer.plot_goal_coding_stats(sfigs_row1[1], tags='fig5_HPC_position_stats',
                                                              update_type=['stay', 'switch', 'non_update'])
        sfigs_row2[0] = pfc_visualizer.plot_goal_coding(sfigs_row2[0], tags='PFC_choice_fig_5', update_type=['stay'])
        sfigs_row2[1] = pfc_visualizer.plot_goal_coding_stats(sfigs_row2[1], tags='fig5_PFC_choice_stats',
                                                              update_type=['stay', 'switch', 'non_update'])

        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_5', tight_layout=False, results_type='manuscript')

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
        sfigs = fig.subfigures(nrows=3, ncols=2, width_ratios=[1.25, 1], height_ratios=[1, 1, 1])

        sfigs[0][0] = hpc_visualizer.plot_goal_coding(sfigs[0][0], comparison='correct', use_residuals=True,
                                                      tags='fig6_HPC_choice_accuracy')
        sfigs[0][1] = hpc_visualizer.plot_goal_coding_stats(sfigs[0][1], comparison='correct', title='hippocampal',
                                                            tags='fig6_HPC_position_accuracy', use_residuals=True,)
        sfigs[1][0] = pfc_visualizer.plot_goal_coding(sfigs[1][0], comparison='correct', use_residuals=True,
                                                      tags='fig6_PFC_position_accuracy')
        sfigs[1][1] = pfc_visualizer.plot_goal_coding_stats(sfigs[1][1], comparison='correct', title='prefrontal',
                                                            tags='fig6_PFC_choice_accuracy', use_residuals=True)
        sfigs[2][0] = hpc_visualizer.plot_goal_coding_prediction(sfigs[2][0], comparison='all_update',
                                                                 tags="('CA1',)_y_position_50_0.2")
        sfigs[2][1] = pfc_visualizer.plot_goal_coding_prediction(sfigs[2][1], comparison='all_update',
                                                                 tags="('PFC',)_choice_50_0.2")

        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_6', tight_layout=False, results_type='manuscript')

        ##### figure 7
        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 6))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])

        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='Choice commitment schematic')
        sfigs_row0[1] = hpc_visualizer.plot_goal_coding_by_commitment(sfigs_row0[1], update_type=['switch'],
                                                                      behavior_only=True)

        sfigs_row1[0] = hpc_visualizer.plot_goal_coding_by_commitment(sfigs_row1[0], update_type=['switch'])
        sfigs_row1[1] = pfc_visualizer.plot_goal_coding_by_commitment(sfigs_row1[1], update_type=['switch'])

        sfigs_row2[0] = hpc_visualizer.plot_goal_coding_by_commitment_single_trials(sfigs_row2[0],
                                                                                    update_type=['switch'],
                                                                                    tags='fig7_CA1_commitment')
        sfigs_row2[1] = pfc_visualizer.plot_goal_coding_by_commitment_single_trials(sfigs_row2[1],
                                                                                    update_type=['switch'],
                                                                                    tags='fig7_PFC_commitment')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'figure_7', tight_layout=False, results_type='manuscript')

        if with_supplement:
            # figures structure 9
            n_sfig = 9
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 6.5))
            sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
            sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
            sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])

            sfigs_row0[0] = hpc_visualizer.plot_goal_coding(sfigs_row0[0], comparison='correct',
                                                            tags=f'sfig{n_sfig}_HPC_choice_accuracy')
            sfigs_row0[1] = pfc_visualizer.plot_goal_coding(sfigs_row0[1], comparison='correct',
                                                          tags=f'sfig{n_sfig}_PFC_position_accuracy')

            sfigs_row1[0] = hpc_visualizer.plot_goal_coding(sfigs_row1[0], comparison='correct', use_residuals=True,
                                                            tags=f'sfig{n_sfig}_HPC_choice_accuracy')
            sfigs_row1[1] = pfc_visualizer.plot_goal_coding(sfigs_row1[1], comparison='correct', use_residuals=True,
                                                            tags=f'sfig{n_sfig}_PFC_position_accuracy')

            sfigs_row2[0] = hpc_visualizer.plot_residual_r_squared(sfigs_row2[0])
            sfigs_row2[1] = pfc_visualizer.plot_residual_r_squared(sfigs_row2[1])

            # figure saving
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False, results_type='manuscript')

            # figure structure - supp fig 11
            n_sfig = 10
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
            sfigs = fig.subfigures(nrows=2, ncols=1, width_ratios=[1], height_ratios=[1, 4])
            sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])

            sfigs_row0[0] = hpc_visualizer.plot_goal_coding_by_commitment(sfigs_row0[0], update_type=['stay'],)
            sfigs_row0[1] = pfc_visualizer.plot_goal_coding_by_commitment(sfigs_row0[1], update_type=['stay'],)
            sfigs[1] = hpc_visualizer.plot_goal_coding_by_commitment_single_trials(sfigs[1], plot_full_breakdown=True,
                                                                                   tags=f'sfig{n_sfig}')

            # figure saving
            self.add_panel_labels(sfigs)
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_{n_sfig}', tight_layout=False, results_type='manuscript')

            # figure structure - pre update onset
            fig = plt.figure(constrained_layout=True, figsize=(6.5, 6))
            sfigs = fig.subfigures(nrows=2, ncols=1, width_ratios=[1], height_ratios=[1, 1])

            sfigs[0] = pfc_visualizer.plot_goal_coding_stats(sfigs[0], comparison='correct', title='pre - prefrontal',
                                                             tags=f'sfig_PFC_choice_pre', time_window=(-1.5, 0),
                                                             use_residuals=True)
            sfigs[1] = hpc_visualizer.plot_goal_coding_stats(sfigs[1], comparison='correct', title='pre - hippocampal',
                                                             tags=f'sfig_HPC_choice_pre', time_window=(-1.5, 0),
                                                             use_residuals=True)

            # figure saving
            self.results_io.save_fig(fig=fig, filename=f'supp_figure_pre_correct', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_all_trials(self):
        encoder_trials_all = dict(update_type=[1, 2, 3], correct=[0, 1], maze_id=[3, 4])
        hpc_visualizer_encode_all = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                               analysis_kwargs=dict(features=['y_position'],
                                                                                    params=dict(region=['CA1'],
                                                                                                encoder_trial_types=encoder_trials_all,
                                                                                                decoder_test_size=0.8)),
                                                               overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=3, ncols=1, width_ratios=[1], height_ratios=[1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

        # plot data
        sfigs_row0[0] = self.plot_placeholder(sfigs_row0[0], text='schematic of using all trials for decoder')
        sfigs_row0[1] = hpc_visualizer_encode_all.plot_decoding_output_heatmap(sfigs_row0[1])
        sfigs[1] = hpc_visualizer_encode_all.plot_goal_coding(sfigs[1], tags='sfig4_CA1_position_all')
        sfigs_row2[0] = hpc_visualizer_encode_all.plot_goal_coding_stats(sfigs_row2[0], tags='sfig4_CA1_position_stats',
                                                                       time_window=(0, 1.5))

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_4', tight_layout=False, results_type='manuscript')

    def plot_supp_figure_lateral_position(self):
        hpc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['x_position'],
                                                                        params=dict(region=['CA1'])),
                                                    overwrite=self.overwrite)
        pfc_visualizer = self.run_analysis_pipeline(analysis_to_run='Decoder',
                                                    analysis_kwargs=dict(features=['x_position'],
                                                                         params=dict(region=['PFC'])),
                                                    overwrite=self.overwrite)

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

        # plot data
        sfigs_row0[0] = hpc_visualizer.plot_tuning_curves(sfigs_row0[0], title='hippocampal lateral position')
        sfigs_row0[1] = hpc_visualizer.plot_decoding_validation(sfigs_row0[1], tags='sfig5_HPC_lateral_position')
        sfigs_row1[0] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row1[0])
        sfigs_row1[1] = hpc_visualizer.plot_decoding_output_heatmap(sfigs_row1[1], update_type='non_update')
        sfigs[2] = hpc_visualizer.plot_goal_coding(sfigs[2], tags='sfig5_HPC_lateral_position',
                                                        update_type=['switch', 'non_update'])
        sfigs_row3[0] = hpc_visualizer.plot_goal_coding_stats(sfigs_row3[0], tags='sfig5_HPC_lateral_position')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_5_hpc', tight_layout=False, results_type='manuscript')

        # figure structure
        fig = plt.figure(constrained_layout=True, figsize=(6.5, 9))
        sfigs = fig.subfigures(nrows=4, ncols=1, height_ratios=[1, 1, 1, 1])
        sfigs_row0 = sfigs[0].subfigures(nrows=1, ncols=2, width_ratios=[1, 2])
        sfigs_row1 = sfigs[1].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        sfigs_row2 = sfigs[2].subfigures(nrows=1, ncols=2, width_ratios=[1, 1])
        sfigs_row3 = sfigs[3].subfigures(nrows=1, ncols=2, width_ratios=[2, 1])

        sfigs_row0[0] = pfc_visualizer.plot_tuning_curves(sfigs_row0[0], title='prefrontal lateral position')
        sfigs_row0[1] = pfc_visualizer.plot_decoding_validation(sfigs_row0[1], tags='sfig5_PFC_lateral_position')
        sfigs_row1[0] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row1[0])
        sfigs_row1[1] = pfc_visualizer.plot_decoding_output_heatmap(sfigs_row1[1], update_type='non_update')
        sfigs_row2[0] = pfc_visualizer.plot_goal_coding(sfigs_row2[0], tags='sfig5_PFC_lateral_position',
                                                        update_type=['switch'])
        sfigs_row2[1] = pfc_visualizer.plot_goal_coding(sfigs_row2[1], tags='sfig5_PFC_lateral_position',
                                                        update_type=['non_update'])
        sfigs_row3[0] = pfc_visualizer.plot_goal_coding_stats(sfigs_row3[0], tags='sfig5_PFC_lateral_position')

        # figure saving
        self.add_panel_labels(sfigs)
        self.results_io.save_fig(fig=fig, filename=f'supp_figure_5_pfc', tight_layout=False, results_type='manuscript')

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
