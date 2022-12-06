import itertools
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import warnings

from bisect import bisect, bisect_left
from sklearn.model_selection import RepeatedStratifiedKFold

from update_project.general.results_io import ResultsIO
from update_project.general.timeseries import align_by_time_intervals as align_by_time_intervals_ts


class DynamicChoiceRNN:

    def __init__(self, nwbfile, session_id, target_var='choice', velocity_only=False):
        # based on grid search of subset of sessions (210407, 210415, 210509, 210520, 210909, 210913, 211113, 211115),
        # best parameters were regularizer = None, learning_rate = 0.1, batch_size = 20-50, epochs = 20-30
        # extra 10 epochs buys 1-2% improvement in accuracy so probably will stick with 20
        # batch size will go with 32 between the two good options since it also seems recommended to keep at power of 2

        # setup params
        self.velocity_only = velocity_only
        self.mask_value = -9999
        self.params = dict(batch_size=32, epochs=20, regularizer=None, learning_rate=0.1, predict_update=True)
        self.grid_search_params = dict(batch_size=[20, 50, 100],
                                       epochs=[10, 20, 30],
                                       regularizer=[None, tf.keras.regularizers.l2(0.01),
                                                    tf.keras.regularizers.l2(0.1)],
                                       learning_rate=[0.01, 0.1],
                                       predict_update=[False])

        # setup data
        self.trials_df = nwbfile.trials.to_dataframe()
        self.session_ts = nwbfile.processing['behavior']['view_angle']['view_angle']
        self.input_data, self.target_data, self.timestamp_data = self._setup_data(nwbfile, target_var)
        _, self.non_update_index = self._get_trial_inds(with_update=False, ret_index=True)
        if self.params.get('predict_update', False):
            input, target, timestamp = self._setup_data(nwbfile, target_var, with_update=True)
            self.update_input_data = input
            self.update_target_data = target
            self.update_timestamp_data = timestamp
            _, self.update_index = self._get_trial_inds(with_update=True, ret_index=True)
            self.max_pad_length = np.max([np.max([len(i) for i in self.input_data]),
                                          np.max([len(u) for u in self.update_input_data])])
        else:
            self.update_input_data = []
            self.update_target_data = []
            self.update_index = []
            self.max_pad_length = np.max([len(i) for i in self.input_data])

        # get results to save
        add_tags = '_velocity_only' if self.velocity_only else ''
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name='dynamic_choice',
                                    tags=f'{target_var}{add_tags}')
        self.data_files = dict(dynamic_choice_output=dict(vars=['output_data', 'agg_data', 'decoder_data', 'params',
                                                                ],
                                                          format='pkl'))

    def run_analysis(self, overwrite=False, grid_search=False):
        if grid_search:
            self.data_files = dict(grid_search=dict(vars=['grid_search_data', 'grid_search_params'], format='pkl'))

        if overwrite:
            if grid_search:
                self._grid_search()
            else:
                self._get_dynamic_choice()
                self._aggregate_data()
                self._get_decoder_data()  # get data in format for bayesian decoder
            self._export_data()
        else:
            if self.results_io.data_exists(self.data_files):
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run_analysis(overwrite=True)

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
        cv = RepeatedStratifiedKFold(n_splits=6, n_repeats=3, random_state=21)
        for train_index, test_index in cv.split(self.input_data, np.array(self.target_data)[:, 0]):
            preprocessed_data = self._preprocess_data(train_index, test_index)
            build_data = dict(norm_data=preprocessed_data['input_train_no_pad'], regularizer=self.params['regularizer'],
                              shape=np.shape(preprocessed_data['input_train']), mask_value=self.mask_value,
                              learning_rate=self.params['learning_rate'])
            model = self._get_classifier(build_data)

            print('Fitting model...')
            history = model.fit(preprocessed_data['input_train'], preprocessed_data['target_train'],
                                batch_size=self.params['batch_size'], epochs=self.params['epochs'])

            score, acc = model.evaluate(preprocessed_data['input_test'], preprocessed_data['target_test'], verbose=0)
            print(f'Evaluating model... score: {score}, binary_accuracy: {acc}')

            print(f'Predicting with model...')
            prediction = model.predict(preprocessed_data['input_test'])
            if self.params.get('predict_update', False):
                update_input_data, update_target_data, update_timestamp_data = self._pad_data(self.update_input_data,
                                                                                              self.update_target_data,
                                                                                              self.update_timestamp_data)
                update_prediction = model.predict(update_input_data)
            else:
                update_prediction = []

            # concatenate data
            output_data.append(dict(score=score, accuracy=acc, history=history.history, prediction=prediction,
                                    input_test_fold=preprocessed_data['input_test'],
                                    target_test_fold=preprocessed_data['target_test'], test_index=test_index,
                                    update_prediction=update_prediction, update_test_index=self.update_index,
                                    timestamp_train=preprocessed_data['timestamp_train'],
                                    timestamp_test=preprocessed_data['timestamp_test'],
                                    timestamp_update=update_timestamp_data))

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

    def _setup_data(self, nwbfile, target_var, with_update=False):
        # get trial data
        trial_inds = self._get_trial_inds(with_update=with_update)

        # get position/velocity data
        pos_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['position']['position'],
                                              nwbfile.intervals['trials'][trial_inds],)
        trans_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['translational_velocity'],
                                                nwbfile.intervals['trials'][trial_inds], )
        rot_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['rotational_velocity'],
                                              nwbfile.intervals['trials'][trial_inds], )
        view_data = align_by_time_intervals_ts(nwbfile.processing['behavior']['view_angle']['view_angle'],
                                               nwbfile.intervals['trials'][trial_inds])
        _, timestamps = align_by_time_intervals_ts(nwbfile.processing['behavior']['position']['position'],
                                                   nwbfile.intervals['trials'][trial_inds], return_timestamps=True)

        # compile data into input matrix
        pos_data = [p/np.nanmax(p) for p in pos_data]  # max of y_position
        if self.velocity_only:
            input_data = [np.vstack([t, r]).T for t, r in zip(trans_data, rot_data)]
        else:
            input_data = [np.vstack([p[:, 1], t, r, v]).T for p, t, r, v in zip(pos_data, trans_data, rot_data, view_data)]

        # get target sequence (binary, reported choice OR initial cue)
        longest_length = np.max([np.shape(k)[0] for k in input_data])
        choice_values = self.trials_df.iloc[trial_inds][target_var].to_numpy() - 1  # 0 for left and 1 for right
        target_data = [np.tile(choice, longest_length) for choice in choice_values]

        return input_data, target_data, timestamps

    def _pad_data(self, input_data, target_data, timestamp_data):
        input_data_pad = tf.keras.preprocessing.sequence.pad_sequences(input_data, dtype='float64', padding='post',
                                                                       value=self.mask_value,
                                                                       maxlen=self.max_pad_length)
        target_data_pad = tf.keras.preprocessing.sequence.pad_sequences(target_data, dtype='float64',
                                                                        truncating='post',
                                                                        maxlen=np.shape(input_data_pad)[1])
        timestamp_data_pad = tf.keras.preprocessing.sequence.pad_sequences(timestamp_data, dtype='float64',
                                                                           padding='post', value=self.mask_value,
                                                                           maxlen=np.shape(input_data_pad)[1])

        return input_data_pad, target_data_pad, timestamp_data_pad

    def _preprocess_data(self, train_index, test_index):
        input_data_pad, target_data_pad, timestamp_data_pad = self._pad_data(self.input_data,
                                                                             self.target_data,
                                                                             self.timestamp_data)

        # get input and target data
        input_train, input_test = input_data_pad[train_index, :, :], input_data_pad[test_index, :, :]
        timestamp_train, timestamp_test = timestamp_data_pad[train_index, :], timestamp_data_pad[test_index, :]
        target_train, target_test = target_data_pad[train_index], target_data_pad[test_index]

        # normalize, sort, and pad data
        input_train_no_pad = [d for ind, d in enumerate(self.input_data) if ind in train_index]

        return dict(input_train=input_train, input_test=input_test, target_train=target_train, target_test=target_test,
                    timestamp_train=timestamp_train, timestamp_test=timestamp_test,
                    input_train_no_pad=input_train_no_pad)

    def _get_trial_inds(self, with_update, ret_index=False):
        if with_update:
            update_types = [2, 3]
        else:
            update_types = [1]

        # get non-update ymaze delay trials
        mask = pd.concat([self.trials_df['update_type'].isin(update_types),  # non-update trials only to train
                          self.trials_df['maze_id'] == 4,  # delay trials only not visual warmup
                          self.trials_df['duration'] <= self.trials_df['duration'].mean() * 2],  # trials shorter than 2*mean only
                          axis=1).all(axis=1)

        if ret_index:
            return np.array(self.trials_df.index[mask]), self.trials_df.index[mask]
        else:
            return np.array(self.trials_df.index[mask])

    def _aggregate_data(self):
        # get average for each trial across all cross-validation folds
        predict_data = self._get_repeated_fold_average(self.output_data, label='prediction')
        update_predict_data = self._get_repeated_fold_average(self.output_data, label='update_prediction',
                                                              trial_index='update_test_index')
        target_data = self._get_repeated_fold_average(self.output_data, label='target_test_fold')[:, -1]
        log_likelihood = np.array([self._log2_likelihood(np.tile(t, len(p)), p) for t, p in zip(target_data, predict_data)])

        timestamps = self._get_repeated_fold_average(self.output_data, label='timestamp_test')
        timestamps[timestamps < -9000] = np.nan  # remove mask values
        update_timestamps = self._get_repeated_fold_average(self.output_data, label='timestamp_update',
                                                            trial_index='update_test_index')
        update_timestamps[update_timestamps < -9000] = np.nan  # remove mask values
        # TODO - right now update data is linked incorrectly to the rest of the data, need to fix

        if self.velocity_only:
            y_position = np.empty(np.shape(timestamps))
            y_position[:] = np.nan
        else:
            y_position = self._get_repeated_fold_average(self.output_data, label='input_test_fold')[:, :, 0]
            y_position[y_position < -9000] = np.nan  # remove mask values
            # update_y_position = self._get_repeated_fold_average(self.output_data, label='input_test_fold')[:, :, 0]
            # update_y_position[update_y_position < -9000] = np.nan  # remove mask values

        self.agg_data = dict(predict=predict_data, update_predict_data=update_predict_data, target=target_data,
                             y_position=y_position, timestamps=timestamps, update_timestamps=update_timestamps,
                             log_likelihood=log_likelihood, )

        # vis_df = pd.json_normalize(dict(predict=predict_data, target=target_data, y_position=y_position,
        #                                 timestamps=timestamps, log_likelihood=log_likelihood, ))
        # self.vis_data = vis_df.explode(list(vis_df.columns))

        # update_df = pd.json_normalize(dict(update_predict_data=update_predict_data, update_timestamps=update_timestamps,
        #                                    update_y_position=update_y_position),)
        # self.update_data = update_df.explode(list(update_df.columns))

    def _get_decoder_data(self):
        # input to decoder is just a long series of
        data = np.empty(np.shape(self.session_ts.data))
        data[:] = np.nan
        session_timestamps = np.round(self.session_ts.timestamps[:], 6)

        for key, value in dict(update_timestamps='update_predict_data', timestamps='predict').items():
            for timestamps, prediction in zip(self.agg_data[key], self.agg_data[value]):
                trial_timestamps = timestamps[~np.isnan(timestamps)]
                trial_prediction = prediction[~np.isnan(timestamps)]
                idx_start = bisect_left(session_timestamps, np.round(trial_timestamps[0], 6))
                idx_stop = bisect(session_timestamps, np.round(trial_timestamps[-1], 6), lo=idx_start)
                data[idx_start:idx_stop] = trial_prediction  # fill in values with the choice value

        self.decoder_data = data

    @staticmethod
    def _log2_likelihood(y_true, y_pred, eps=1e-15):
        # adjust small values so work with log correctly
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # calculate log_likelihood of elements
        log_likelihood_elem = y_true * np.log2(y_pred) + (1 - y_true) * np.log2(1 - y_pred)

        return log_likelihood_elem

    @staticmethod
    def _get_repeated_fold_average(output_df, label=None, trial_index='test_index'):
        all_trial_inds = np.hstack(output_df[trial_index].to_numpy())
        all_trial_outputs = np.vstack(output_df[label].to_numpy())

        mean_data = []
        for trial_ind in np.unique(all_trial_inds):
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