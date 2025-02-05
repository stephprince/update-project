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
    #analyzer.plot_figure_4(with_supplement=False)
    #analyzer.plot_figure_b()#long windowed heatmap with iti nan#DC currently looking at
    analyzer.plot_figure_c()#stay vs delay only represenations with smaller timewindow done#looks fine, now has all trials
    #analyzer.plot_figure_d()#correct incorrect on delay only trials done
    #analyzer.plot_figure_e()#position stats PC switch vs delay done
    analyzer.plot_figure_f()#examples for ind. trials(F)
    #analyzer.plot_figure_g()#accounting for view angle, done
    #analyzer.plot_figure_h()#nonlocal representations#DC  currently looking at
    #analyzer.plot_figure_i()#comparing across regions
    ###analyzer.plot_supp_figure_all_trials()#all trials used for encoding. (change name back in analyzer when done and set subset)
    #analyzer.plot_figure_j()#all remaining heat maps that need flipped#good with all trials now
    #analyzer.plot_figure_k()#b4 cue onset comparisons for choice#stats and figure will spit out with initial and new flipped
    #analyzer.plot_figure_l()
    #analyzer.plot_figure_1(with_supplement=False)

    


if __name__ == '__main__':
    animals = [17, 20, 25, 28, 29, 33, 34]
    #animals = [25,33]
    run_all_figures(animals)
