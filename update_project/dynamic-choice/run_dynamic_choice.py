import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from pynwb import NWBHDF5IO
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import normalize

from update_project.session_loader import SessionLoader
from update_project.results_io import ResultsIO
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts

plt.style.use(Path().absolute().parent / 'prince-paper.mplstyle')


def load_dynamic_choice_data(nwbfile):
    # get non-update ymaze delay trials
    trials_df = nwbfile.trials.to_dataframe()
    mask = pd.concat([trials_df['update_type'] == 1,  # non-update trials only to train
                      trials_df['maze_id'] == 4,  # delay trials only not visual warmup
                      trials_df['duration'] <= trials_df['duration'].mean()*2]  # trials shorter than 2*mean only
                     , axis=1).all(axis=1)
    trial_inds = np.array(trials_df.index[mask])

    # get position/velocity data
    pos_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['position']['position'],
                                                         nwbfile.intervals['trials'][trial_inds],)
    trans_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['translational_velocity'],
                                               nwbfile.intervals['trials'][trial_inds], )
    rot_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['rotational_velocity'],
                                               nwbfile.intervals['trials'][trial_inds], )
    view_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['view_angle']['view_angle'],
                                                           nwbfile.intervals['trials'][trial_inds])

    # resample data so all constant sampling rate
    data = [np.vstack([p[:, 0], p[:, 1], t, r, v]).T for p, t, r, v in zip(pos_data, trans_data, rot_data, view_data)]
    input_data = sorted(data, key=lambda k: np.shape(k)[0])
    sort_index = [i[0] for i in sorted(enumerate(data), key=lambda k: np.shape(k[1])[0])]

    # get target sequence (binary, reported choice OR initial cue)
    longest_length = np.max([np.shape(k)[0] for k in input_data])
    choice_values = trials_df[mask]['choice'].to_numpy() - 1  # 0 for left and 1 for right
    choice_values_sorted = choice_values[sort_index]
    target_data = [np.tile(choice, longest_length) for choice in choice_values_sorted]

    return input_data, target_data, sort_index


def run_dynamic_choice():
    # setup sessions
    animals = [17, 20, 25, 28, 29]  # 17, 20, 25, 28, 29
    dates_included = [210913]  # 210913
    dates_excluded = []
    session_db = SessionLoader(animals=animals, dates_included=dates_included, dates_excluded=dates_excluded)
    session_names = session_db.load_session_names()

    for name in session_names:
        # load nwb file
        print(f'Getting dynamic-choice data for {session_db.get_session_id(name)}')
        io = NWBHDF5IO(session_db.get_session_path(name), 'r')
        nwbfile = io.read()
        results_io = ResultsIO(creator_file=__file__, session_id=session_db.get_session_id(name),
                               folder_name='dynamic-choice', )
        # get data
        input_data, target_data, sort_index = load_dynamic_choice_data(nwbfile)

        # k-fold cross validatione
        cv = KFold(n_splits=6, random_state=21)
        all_scores, all_accuracies, all_outputs = [], [], []
        all_test_trials = []
        for train_index, test_index in cv.split(input_data):
            # get input and target data
            i_train_fold = [d for ind, d in enumerate(input_data) if ind in train_index]
            i_test_fold = [d for ind, d in enumerate(input_data) if ind in test_index]
            t_train_fold, t_test_fold = np.array(target_data)[train_index], np.array(target_data)[test_index]

            # mask data
            mask_value = -10000
            input_train_fold = tf.keras.preprocessing.sequence.pad_sequences(i_train_fold, dtype='float64', padding='post', value=-10000)
            input_test_fold = tf.keras.preprocessing.sequence.pad_sequences(i_test_fold, dtype='float64', padding='post', value=-10000)
            target_train_fold = tf.keras.preprocessing.sequence.pad_sequences(t_train_fold, dtype='float64', truncating='post', maxlen=np.shape(input_train_fold)[1])
            target_test_fold = tf.keras.preprocessing.sequence.pad_sequences(t_test_fold, dtype='float64', truncating='post', maxlen=np.shape(input_train_fold)[1])

            input_train_fold, input_test_fold = input_data_pad[train_index, :, :], input_data_pad[test_index, :, :]
            target_train_fold, target_test_fold = target_data_pad[train_index], target_data_pad[test_index]

            # balance data (can't use stratifiedKFold to balance bc classes of target and input not the same)
            # TODO - resample L/R trial types if needed so equal amounts of each
            # TODO - sort input data

            # normalize, sort, and pad data
            layer_norm = tf.keras.layers.Normalization(axis=1)
            layer_norm.adapt(np.vstack(input_data))  # put in all batch data that is not padded

            # build model
            print('Creating model...')
            model = tf.keras.models.Sequential()
            model.add(tf.keras.Input(shape=np.shape(input_train_fold)[1:]))
            model.add(tf.keras.layers.Masking(mask_value=mask_value, input_shape=np.shape(input_train_fold)[1:]))
            model.add(layer_norm)
            model.add(tf.keras.layers.LSTM(units=10, input_shape=np.shape(input_train_fold), return_sequences=True))
            model.add(tf.keras.layers.Dense(units=1,
                                            activation='sigmoid',
                                            kernel_regularizer=tf.keras.regularizers.L2(l2=0.1)))  # is this where it goes?

            print('Compiling model...')
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                          metrics=tf.keras.metrics.BinaryAccuracy())  # what should my loss be
            #model.summary()

            # fit model
            print('Fitting model...')
            model.fit(input_train_fold, target_train_fold, batch_size=100, epochs=3)  # number of epochs?

            # assess model accuracy
            score, acc = model.evaluate(input_test_fold, target_test_fold)

            # use model to generate dynamic choice variable
            output = model.predict(input_test_fold)

            # concatenate data
            all_scores.append(score)
            all_accuracies.append(acc)
            all_outputs.append(output)
            all_test_trials.append(test_index)

        # The hyperparameters of the network architecture and training procedure
        # were selected using a grid search in a small number of pilot sessions.
        params = dict(batch_size=[100, 20, 50],
                      epochs=[3, 5, 10, 20],
                      )  # TODO - maybe something about k-fold numbers, unit numbers, l2 regularization etc.
        estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(model)  # want this to be model only up to compile
        gs = GridSearchCV(estimator=estimator, param_grid=params, cv=cv)
        gs = gs.fit(input_data, target_data)
        gs.cv_results
        #cross_val_score(model, input_data, test_data, scoring='accuracy', cv=cv)

        # get decoder accuracy

        #  The decoder performance, or the decodability of reported choice or cue identity, was quantified as model log
        #  likelihood, equivalent to the negatively signed binary cross-entropy loss. where
        #  is the true binary value (reported choice or cue), and
        #  is the prediction (dynamic choice or cue-biased running). Log base 2 was used so that the log likelihood
        #  equals -1 for chance-level predictions and 0 for perfect prediction.



if __name__ == '__main__':
    run_dynamic_choice()
