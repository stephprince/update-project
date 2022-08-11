import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from pynwb import NWBHDF5IO
from sklearn.model_selection import RepeatedKFold, GridSearchCV
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
                      trials_df['duration'] <= trials_df['duration'].mean()*2],  # trials shorter than 2*mean only
                     axis=1).all(axis=1)
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
    input_data = [np.vstack([p[:, 1], t, r, v]).T for p, t, r, v in zip(pos_data, trans_data, rot_data, view_data)]

    # get target sequence (binary, reported choice OR initial cue)
    longest_length = np.max([np.shape(k)[0] for k in input_data])
    choice_values = trials_df[mask]['choice'].to_numpy() - 1  # 0 for left and 1 for right
    target_data = [np.tile(choice, longest_length) for choice in choice_values]

    return input_data, target_data

def get_repeated_fold_average(output_df, label=None, stack='v'):
    all_trial_inds = np.hstack(output_df['test_index'].to_numpy())
    if stack == 'v':
        all_trial_outputs = np.vstack(output_df[label].to_numpy())
    elif stack == 'h':
        all_trial_outputs = np.hstack(output_df[label].to_numpy())

    mean_data = []
    for trial_ind in range(np.shape(np.unique(all_trial_inds))[0]):
        if np.ndim(all_trial_outputs):
            output_data = all_trial_outputs[all_trial_inds == trial_ind]
        else:
            output_data = all_trial_outputs[all_trial_inds == trial_ind, :]
        mean_data.append(np.mean(output_data, axis=0).squeeze())

    return np.array(mean_data)


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
        input_data, target_data = load_dynamic_choice_data(nwbfile)

        # k-fold cross validation
        output_data = []
        cv = RepeatedKFold(n_splits=6, n_repeats=5, random_state=21)
        for train_index, test_index in cv.split(input_data, target_data):
            # pad data
            mask_value = -9999
            input_data_pad = tf.keras.preprocessing.sequence.pad_sequences(input_data, dtype='float64', padding='post',
                                                                           value=mask_value)
            target_data_pad = tf.keras.preprocessing.sequence.pad_sequences(target_data, dtype='float64',
                                                                            truncating='post',
                                                                            maxlen=np.shape(input_data_pad)[1])

            # get input and target data
            input_train_fold, input_test_fold = input_data_pad[train_index, :, :], input_data_pad[test_index, :, :]
            target_train_fold, target_test_fold = target_data_pad[train_index], target_data_pad[test_index]

            # balance data (can't use stratifiedKFold to balance bc classes of target and input not the same)
            # TODO - resample L/R trial types if needed so equal amounts of each

            # normalize, sort, and pad data
            input_train_no_pad = [d for ind, d in enumerate(input_data) if ind in train_index]
            layer_norm = tf.keras.layers.Normalization(axis=1)
            layer_norm.adapt(np.vstack(input_train_no_pad))  # put in all batch data that is not padded

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
            # TODO - might want to add kernel_regularizer to LSTM too?

            print('Compiling model...')
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                          metrics=tf.keras.metrics.BinaryAccuracy())  # what should my loss be
            model.summary()

            # fit model
            print('Fitting model...')
            history = model.fit(input_train_fold, target_train_fold, batch_size=100, epochs=5)  # number of epochs?

            # assess model accuracy
            score, acc = model.evaluate(input_test_fold, target_test_fold, verbose=0)
            print(f'Evaluating model... score: {score}, binary_accuracy: {acc}')

            # use model to generate dynamic choice variable
            prediction = model.predict(input_test_fold)

            # concatenate data
            output_data.append(dict(history=history.history, score=score, accuracy=acc, prediction=prediction,
                                    input_test_fold=input_test_fold,
                                    target_test_fold=target_test_fold[:, 0].astype(int), test_index=test_index))

        # get average for each trial across all cross-validation folds
        output_df = pd.DataFrame(output_data)
        predict_data = get_repeated_fold_average(output_df, label='prediction', stack='v')
        target_data = get_repeated_fold_average(output_df, label='target_test_fold', stack='h').astype(int)
        y_position = get_repeated_fold_average(output_df, label='input_test_fold', stack='v')[:, :, 0]
        y_position[y_position < -9000] = np.nan

        # plot the results for the session
        fig, axes = plt.subplots()
        axes.plot(y_position.T[:, target_data == 1], predict_data.T[:, target_data == 1], color='b',
                  label='left choice')
        axes.plot(y_position.T[:, target_data == 0], predict_data.T[:, target_data == 0], color='r',
                  label='right choice')
        axes.set(xlabel='position in track', ylabel='p(left)', title='LSTM prediction - test trials')
        plt.show()

        # get log likelihood
        log_likelihood = target_data * log2(predict_data) + (1 - target_data)*log2(1 - predict_data)
        # grid search for hyperparameters
        params = dict(batch_size=[100, 20, 50],
                      epochs=[3, 5, 10, 20],
                      )  # TODO - maybe something about k-fold numbers, unit numbers, l2 regularization etc. loss fxn
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
