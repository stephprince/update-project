from update_project.general.session_loader import SessionLoader
from update_task_figure_generator import UpdateTaskFigureGenerator


def run_all_figures(animals, dates_included=None, overwrite=False):
    # load relevant sessions
    session_db = SessionLoader(animals=animals, dates_included=dates_included)

    # initialize general analysis class and run for each of the specified steps
    analyzer = UpdateTaskFigureGenerator(sessions=session_db, overwrite=overwrite)

    # generate all manuscript figures
    #analyzer.plot_main_figures()
    #analyzer.plot_supplemental_figures()
    #analyzer.plot_figure_a()#hpc goal coding stay vs delay done
    #analyzer.plot_figure_e()#position stats PC switch vs delay done
    ###analyzer.plot_supp_figure_all_trials()#all trials used for encoding. (change name back in analyzer when done and set subset)
    analyzer.plot_figure_k()#b4 cue onset comparisons for choice#
    #analyzer.plot_figure_l()
    #analyzer.plot_figure_1(with_supplement=False)

    


if __name__ == '__main__':
    animals = [17, 20, 25, 28, 29, 33, 34]
    #animals = [25,33]
    run_all_figures(animals)
