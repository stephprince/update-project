from pynwb import NWBHDF5IO

from update_project.session_loader import SessionLoader
from bayesian_decoder import BayesianDecoder

# setup
animals = [25]  # 17, 20, 25, 28, 29
dates_included = [210913]  #210913
dates_excluded = []
overwrite = True

session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = session_db.load_sessions()

# run decoder for individual sessions
for name, session in unique_sessions:

    # load nwb file
    session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
    io = NWBHDF5IO(str(session_db.get_base_path() / f'{session_id}.nwb'), 'r')
    nwbfile = io.read()

    # run decoder
    params = dict(units_threshold=20,
                  speed_threshold=1,
                  firing_threshold=0,
                  encoder_trial_types=dict(update_type=[1]),
                  encoder_bin_num=30,
                  decoder_trial_types=dict(update_type=[1, 2, 3]),
                  decoder_bin_size=0.25,
                  decoder_bin_type='time',)
    features = ['x_position', 'y_position', 'view_angle', 'choice']
    for feat in features:
        # build decoding model
        view_angle_decoder = BayesianDecoder(nwbfile=nwbfile, params=params, session_id=session_id, feature=feat,
                                             overwrite=overwrite)
        view_angle_decoder.run_decoding()

        # compile and save data structures
        view_angle_decoder.summarize()  # general summary of decoding accuracy
        view_angle_decoder.aggregate(trial_types=['switch', 'stay'], times='t_update', nbins=50, window=5, flip=True)  # around update times
        view_angle_decoder.export_data()

        # save intermediate data
        view_angle_decoder.plot()


# get decoder group summary data
BayesianDecoderSummarizer




    # decode view angle



    # get plotting args
    filename = 'bayesian_decoding_summary'
    fig_args = fig_generator.get_figure_args(filename=filename, session_id=session_id)





