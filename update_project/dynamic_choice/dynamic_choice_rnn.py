import itertools
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import warnings

from sklearn.model_selection import RepeatedStratifiedKFold

from update_project.results_io import ResultsIO
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts


class DynamicChoiceRNN:

    def __init__(self, nwbfile, session_id):
        self.trials_df = nwbfile.trials.to_dataframe()
        self.input_data, self.target_data = self._setup_data(nwbfile)

        self.mask_value = -9999
        self.grid_search_params = dict(batch_size=[20, 50, 100],
                                       epochs=[10, 20, 30],
                                       regularizer=[None, tf.keras.regularizers.l2(0.01),
                                                    tf.keras.regularizers.l2(0.1)],
                                       learning_rate=[0.01, 0.1])
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name='dynamic-choice', )
        self.data_files = dict(behavior_output=dict(vars=['output_data', 'agg_data', 'trajectories'],
                                                    format='pkl'))

    def run(self, overwrite=False, grid_search=False):
        if overwrite:
            if grid_search:
                self.data_files = dict(behavior_output=dict(vars=['grid_search_data'], format='pkl'))
                self._grid_search()
            else:
                self._get_dynamic_choice()
                self._aggregate_data()
            self._export_data()
        else:
            if grid_search:
                self.data_files = dict(behavior_output=dict(vars=['grid_search_data'], format='pkl'))

            if self.results_io.data_exists(self.data_files):
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run(overwrite=True)

    def _grid_search(self):
        grid_search_data = []
        for batch_size, epochs, regularizer, learning_rate in itertools.product(*list(self.grid_search_params.values())):
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=21)
            for train_index, test_index in cv.split(self.input_data, np.array(self.target_data)[:, 0]):
                preprocessed_data = self._preprocess_data(train_index, test_index)
                build_data = dict(norm_data=preprocessed_data['input_train_no_pad'], regularizer=regularizer,
                                  shape=np.shape(preprocessed_data['input_train']), mask_value=self.mask_value,
                                  learning_rate=learning_rate)
                model = self._get_classifier(build_data)
                history = model.fit(preprocessed_data['input_train'], preprocessed_data['target_train'],
                                    validation_data=(preprocessed_data['input_test'], preprocessed_data['target_test']),
                                    batch_size=batch_size, epochs=epochs)
                score, acc = model.evaluate(preprocessed_data['input_test'], preprocessed_data['target_test'],
                                            verbose=0)

                grid_search_data.append(dict(score=score, accuracy=acc, history=history.history, batch_size=batch_size,
                                             epochs=epochs, regularizer=regularizer, learning_rate=learning_rate))

        self.grid_search_data = pd.DataFrame(grid_search_data)

    def _get_dynamic_choice(self):
        # k-fold cross validation
        output_data = []
        cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=2, random_state=21)
        for train_index, test_index in cv.split(self.input_data, np.array(self.target_data)[:, 0]):
            preprocessed_data = self._preprocess_data(train_index, test_index)
            build_data = dict(norm_data=preprocessed_data['input_train_no_pad'], regularizer=None,
                              shape=np.shape(preprocessed_data['input_train']), mask_value=self.mask_value)
            model = self._get_classifier(build_data)

            print('Fitting model...')
            history = model.fit(preprocessed_data['input_train'], preprocessed_data['target_train'],
                                batch_size=100, epochs=5)

            score, acc = model.evaluate(preprocessed_data['input_test'], preprocessed_data['target_test'], verbose=0)
            prediction = model.predict(preprocessed_data['input_test'])
            print(f'Evaluating model... score: {score}, binary_accuracy: {acc}')

            # concatenate data
            output_data.append(dict(history=history.history, score=score, accuracy=acc, prediction=prediction,
                                    input_test_fold=preprocessed_data['input_test'],
                                    target_test_fold=preprocessed_data['target_test'], test_index=test_index))

        self.output_data = pd.DataFrame(output_data)

    @staticmethod
    def _get_classifier(build_data=None):
        # setup normalization layer
        layer_norm = tf.keras.layers.Normalization(axis=1)
        layer_norm.adapt(np.vstack(build_data['norm_data']))  # put in all batch data that is not padded

        # build model
        print('Creating model...')
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=build_data['shape'][1:]))
        model.add(tf.keras.layers.Masking(mask_value=build_data['mask_value'],
                                          input_shape=build_data['shape'][1:]))
        model.add(layer_norm)
        model.add(tf.keras.layers.LSTM(units=10, input_shape=build_data['shape'], return_sequences=True,
                                       kernel_regularizer=build_data['regularizer']))
        model.add(tf.keras.layers.Dense(units=1,
                                        activation='sigmoid',
                                        kernel_regularizer=build_data['regularizer']))

        # compile model
        print('Compiling model...')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=build_data['learning_rate']),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=tf.keras.metrics.BinaryAccuracy())

        return model

    def _preprocess_data(self, train_index, test_index):
        # pad data
        input_data_pad = tf.keras.preprocessing.sequence.pad_sequences(self.input_data, dtype='float64', padding='post',
                                                                       value=self.mask_value)
        target_data_pad = tf.keras.preprocessing.sequence.pad_sequences(self.target_data, dtype='float64',
                                                                        truncating='post',
                                                                        maxlen=np.shape(input_data_pad)[1])

        # get input and target data
        input_train, input_test = input_data_pad[train_index, :, :], input_data_pad[test_index, :, :]
        target_train, target_test = target_data_pad[train_index], target_data_pad[test_index]

        # normalize, sort, and pad data
        input_train_no_pad = [d for ind, d in enumerate(self.input_data) if ind in train_index]

        return dict(input_train=input_train, input_test=input_test, target_train=target_train, target_test=target_test,
                    input_train_no_pad=input_train_no_pad)

    def _get_trial_inds(self):
        # get non-update ymaze delay trials
        mask = pd.concat([self.trials_df['update_type'] == 1,  # non-update trials only to train
                          self.trials_df['maze_id'] == 4,  # delay trials only not visual warmup
                          self.trials_df['duration'] <= self.trials_df['duration'].mean() * 2],  # trials shorter than 2*mean only
                          axis=1).all(axis=1)

        return np.array(self.trials_df.index[mask])

    def _setup_data(self, nwbfile):
        # get trial data
        trial_inds = self._get_trial_inds()

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
        pos_data = [p/np.nanmax(p) for p in pos_data]  # max of y_position
        input_data = [np.vstack([p[:, 1], t, r, v]).T for p, t, r, v in zip(pos_data, trans_data, rot_data, view_data)]

        # get target sequence (binary, reported choice OR initial cue)
        longest_length = np.max([np.shape(k)[0] for k in input_data])
        choice_values = self.trials_df.iloc[trial_inds]['choice'].to_numpy() - 1  # 0 for left and 1 for right
        target_data = [np.tile(choice, longest_length) for choice in choice_values]

        return input_data, target_data

    def _aggregate_data(self):  # TODO - decide if I should leave here or move to visualizer
        # get average for each trial across all cross-validation folds
        predict_data = self._get_repeated_fold_average(self.output_data, label='prediction')
        target_data = self._get_repeated_fold_average(self.output_data, label='target_test_fold')[:, 0].astype(int)
        y_position = self._get_repeated_fold_average(self.output_data, label='input_test_fold')[:, :, 0]
        y_position[y_position < -9000] = np.nan  # remove mask values
        log_likelihood = np.array([self._log2_likelihood(np.tile(t, len(p)), p) for t, p in zip(target_data, predict_data)])

        self.agg_data = dict(predict=predict_data, target=target_data, y_position=y_position,
                             log_likelihood=log_likelihood)

    @staticmethod
    def _log2_likelihood(y_true, y_pred, eps=1e-15):
        # adjust small values so work with log correctly
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # calculate log_likelihood of elements
        log_likelihood_elem = y_true * np.log2(y_pred) + (1 - y_true) * np.log2(1 - y_pred)

        return log_likelihood_elem

    @staticmethod
    def _get_repeated_fold_average(output_df, label=None):
        all_trial_inds = np.hstack(output_df['test_index'].to_numpy())
        all_trial_outputs = np.vstack(output_df[label].to_numpy())

        mean_data = []
        for trial_ind in range(np.shape(np.unique(all_trial_inds))[0]):
            output_data = all_trial_outputs[all_trial_inds == trial_ind, :]
            mean_data.append(np.mean(output_data, axis=0).squeeze())

        return np.array(mean_data)

    def _load_data(self):
        print(f'Loading existing data for session {self.results_io.session_id}...')

        # load npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            import_data = self.results_io.load_pickled_data(fname)
            for v, data in zip(file_info['vars'], import_data):
                setattr(self, v, data)

        return self

    def _export_data(self):
        print(f'Exporting data for session {self.results_io.session_id}...')

        # save npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            with open(fname, 'wb') as f:
                [pickle.dump(getattr(self, v), f) for v in file_info['vars']]