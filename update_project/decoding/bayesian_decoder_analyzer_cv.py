import numpy as np
import pandas as pd
import pynapple as nap
import warnings

from bisect import bisect, bisect_left
from pathlib import Path
from pynwb import NWBFile
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from update_project.general.results_io import ResultsIO
from update_project.general.virtual_track import UpdateTrack
from update_project.general.lfp import get_theta
from update_project.general.acquisition import get_velocity
from update_project.general.trials import get_trials_dataframe
from update_project.general.preprocessing import get_commitment_data
from update_project.base_analysis_class import BaseAnalysisClass


class BayesianDecoderAnalyzerCV(BaseAnalysisClass):
    def __init__(self, nwbfile: NWBFile, session_id: str, features: list, params=dict()):#turn off after this
        # setup parameters
        self.region = params.get('region', ['CA1', 'PFC'])
        self.subset_reg = False
        self.units_types = params.get('units_types',
                                      dict(region=self.region,  # dict of filters to apply to units table
                                           cell_type=['Pyramidal Cell', 'Narrow Interneuron', 'Wide Interneuron']))
        self.speed_threshold = params.get('speed_threshold', 1000)  # minimum virtual speed to subselect epochs
        self.firing_threshold = params.get('firing_threshold', 0)  # Hz, minimum peak firing rate of place cells to use
        self.decoder_test_size = params.get('decoder_test_size', 0.2)  # prop of trials for testing on train/test split
        self.encoder_trial_types = params.get('encoder_trial_types', dict(update_type=[1],  # TODO - test and then replace
                                                                          correct=[0, 1],
                                                                          maze_id=[3, 4]))  # trial filters
        self.encoder_bin_num = params.get('encoder_bin_num', 50)  # number of bins to build encoder
        self.decoder_trial_types = params.get('decoder_trial_types', dict(update_type=[1, 2, 3],  # TODO - test and replace
                                                                          correct=[0, 1],
                                                                          maze_id=[4]))  # trial filters
        self.decoder_trial_types_delay_only = dict(update_type=[1], correct=[0, 1], maze_id=[4])
        self.decoder_trial_types_update_only = dict(update_type=[2, 3], correct=[0, 1], maze_id=[4])
        self.decoder_bin_type = params.get('decoder_bin_type', 'time')  # time or theta phase to use for decoder
        self.decoder_bin_size = params.get('decoder_bin_size', 0.2)  # time to use for decoder
        self.linearized_features = params.get('linearized_features', ['y_position'])  # which features to linearize
        self.prior = params.get('prior', 'uniform')  # whether to use uniform or history-dependent prior
        self.virtual_track = UpdateTrack(linearization=bool(self.linearized_features))

        # setup decoding/encoding functions based on dimensions
        self.dim_num = params.get('dim_num', 1)  # 1D decoding default
        self.encoder, self.decoder = self._setup_decoding_functions()

        # setup file paths for io
        trial_types = [str(t) for t in self.encoder_trial_types['correct']]
        self.results_tags = f"{'_'.join(features)}_{'_'.join(self.units_types['region'])}_" \
                            f"enc_binscvtest{self.encoder_bin_num}_dec_bins{self.decoder_bin_size}_speed_thresh" \
                            f"{self.speed_threshold}_trial_types{'_'.join(trial_types)}"\
                            f"{'_subset' if self.subset_reg == True else ''}"
        self.results_io = ResultsIO(creator_file=__file__, session_id=session_id, folder_name=Path(__file__).parent.stem,
                                    tags=self.results_tags)
        self.data_files = dict(bayesian_decoder_output=dict(vars=['encoder_times', 'decoder_times',
                                                                  'decoder_times_delay_only', 'decoder_times_update_only',
                                                                  'spikes', 'features_test', 'features_train',
                                                                  'features_test_delay_only', 'features_test_update_only',
                                                                  'train_df', 'test_df', 'test_delay_only_df', 'test_update_only_df',
                                                                  'model', 'model_test', 'model_delay_only', 'model_update_only',
                                                                  'bins', 'decoded_values', 'decoded_probs',
                                                                  'theta', 'velocity', 'commitment'],
                                                            format='pkl'),
                               params=dict(vars=['speed_threshold', 'firing_threshold', 'units_types',
                                                 'encoder_trial_types', 'encoder_bin_num', 'decoder_trial_types',
                                                 'decoder_bin_type', 'decoder_bin_size', 'decoder_test_size', 'dim_num',
                                                 'feature_names', 'linearized_features',],
                                           format='npz'))#make sure these are labelled all or converted to all as needed

        # setup data
        self.feature_names = features
        self.trials = get_trials_dataframe(nwbfile, with_pseudoupdate=True)
        self.units = nwbfile.units.to_dataframe()
        self.data = self._setup_data(nwbfile)
        self.velocity = get_velocity(nwbfile)
        self.commitment = get_commitment_data(nwbfile, self.results_io)
        self.theta = get_theta(nwbfile, adjust_reference=True, session_id=session_id)
        self.limits = {feat: self.virtual_track.get_limits(feat) for feat in self.feature_names}

        # setup feature specific settings
        self.convert_to_binary = params.get('convert_to_binary', False)  # convert decoded outputs to binary (e.g., L/R)
        if self.feature_names[0] in ['choice_binarized', 'turn_type']:
            self.convert_to_binary = True  # always convert choice to binary
            self.encoder_bin_num = 2

    def run_analysis(self, overwrite=False, export_data=True):
        print(f'Decoding data for session {self.results_io.session_id}...')
 
        if overwrite:
            self._preprocess()._encode()._decode()  # build model. will need to put a for loop in all of these processes
            if export_data:
                self._export_data()  # save output data
        else:
            if self._data_exists() and self._params_match():
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data with those input parameters does not exist, setting overwrite to True')
                self.run_analysis(overwrite=True, export_data=export_data)

        return self

    def _setup_data(self, nwbfile):
        data_dict = dict()
        for feat in self.feature_names:
            if feat in ['x_position', 'y_position']:
                time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
                if feat in self.linearized_features:
                    raw_data = time_series.data[:]
                    data = self.virtual_track.linearize_track_position(raw_data)
                else:
                    column = [ind for ind, val in enumerate(['x_position', 'y_position']) if val == feat][0]
                    data = time_series.data[:, column]
            elif feat in ['view_angle']:
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')
                data = time_series.data[:]
            elif feat in ['choice_binarized', 'turn_type']:
                choice_mapping = {'0': np.nan, '1': -1, '2': 1}  # convert to negative/non-negative for flipping
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

                data = np.empty(np.shape(time_series.data))
                data[:] = np.nan
                for n, trial in self.trials.iterrows():
                    idx_start = bisect_left(time_series.timestamps, trial['start_time'])
                    idx_stop = bisect(time_series.timestamps, trial['stop_time'], lo=idx_start)
                    data[idx_start:idx_stop] = choice_mapping[
                        str(int(trial[feat]))]  # fill in values with the choice value

                    # choice is the animals final decision but turn_type indicates what side is correct/incorrect
                    # turn type will flip around the update
                    if feat in ['turn_type'] and ~np.isnan(trial['t_update']) and trial['update_type'] == 2:
                        idx_switch = bisect_left(time_series.timestamps, trial['t_update'])
                        data[idx_switch:idx_stop] = -1 * choice_mapping[str(int(trial[feat]))]
            elif feat in ['choice', 'cue_bias']:
                time_series = nwbfile.processing['behavior']['view_angle'].get_spatial_series('view_angle')

                # load dynamic choice from saved output
                data_mapping = dict(dynamic_choice='choice', choice='choice',
                                    cue_bias='turn_type')
                fname_tag = data_mapping[self.feature_names[0]]
                choice_path = Path(__file__).parent.parent.parent / 'results' / 'choice'
                fname = self.results_io.get_data_filename(filename=f'dynamic_choice_output_{fname_tag}',
                                                          results_type='session', format='pkl',
                                                          diff_base_path=choice_path)
                import_data = self.results_io.load_pickled_data(fname)
                for v, i_data in zip(['output_data', 'agg_data', 'decoder_data', 'params'], import_data):
                    if v == 'decoder_data':
                        choice_data = i_data
                data = choice_data - 0.5  # convert to -0.5 and +0.5 for flipping between trials
            else:
                raise RuntimeError(f'{feat} feature is not currently supported')

            data_dict[feat] = pd.Series(index=time_series.timestamps[:], data=data)

        return pd.DataFrame.from_dict(data_dict)

    def _get_time_intervals(self, trial_starts, trial_stops, align_times=None):
        movement = self.velocity['combined'] > self.speed_threshold

        if align_times is not None:
            iterator = zip(trial_starts, trial_stops, align_times)
        else:
            align_times = trial_stops.copy(deep=True)
            align_times[:] = np.nan
            iterator = zip(trial_starts, trial_stops, align_times)  # nan values if none indicated

        new_starts = []
        new_stops = []
        for start, stop, align in iterator:
            # pull out trial specific time periods
            start_ind = bisect(movement.index, start)
            stop_ind = bisect_left(movement.index, stop)
            movement_by_trial = movement.iloc[start_ind:stop_ind]

            # get intervals where movement crosses speed threshold
            breaks = np.where(np.hstack((True, np.diff(movement_by_trial), True)))[0]
            breaks[-1] = breaks[-1] - 1  # subtract one index to get closest to final trial stop time
            index = pd.IntervalIndex.from_breaks(movement_by_trial.index[breaks])
            movements_df = pd.DataFrame(movement_by_trial[index.left].to_numpy(), index=index, columns=['moving'])

            # append new start/stop times to data struct
            new_times = movements_df[movements_df['moving'] == True].index
            if np.isnan(align):
                new_left = np.array(new_times.left.values[0])
                new_right = np.array(new_times.right.values[-1])
                new_starts.append(new_left.item())
                new_stops.append(new_right.item())
            else:
                n_steps_back = np.floor((align - new_times.left.values) / self.decoder_bin_size)
                n_steps_forward = np.floor((new_times.right.values - align) / self.decoder_bin_size)
                new_left = align - (self.decoder_bin_size * n_steps_back)
                new_right = align + (self.decoder_bin_size * n_steps_forward)
                assert np.max((new_times.right.values - new_times.left.values) -
                              (new_right - new_left)) < self.decoder_bin_size * 2, \
                    'Durations differences after adjustment should be no more than 2x bin size (1 bin for forward and' \
                    ' back'
                new_starts.append(new_left[-1].item())
                new_stops.append(new_right[-1].item())

            #new_starts.append(new_left)
            #new_stops.append(new_right[-1].item())

        if np.size(new_starts):
            start = np.hstack(new_starts)
            end = np.hstack(new_stops)
        else:
            start, end = [], []
        times = nap.IntervalSet(start=start, end=end, time_units='s')

        return times

    def _setup_decoding_functions(self):
        if self.dim_num == 1:
            encoder = nap.compute_1d_tuning_curves
            decoder = nap.decode_1d
        elif self.dim_num == 2:  # TODO - test 2D data
            encoder = nap.compute_2d_tuning_curves
            decoder = nap.decode_2d
        elif self.dim_num not in [1, 2]:
            raise RuntimeError('Invalid decoding dimension')

        return encoder, decoder

    def _params_match(self):
        params_path = self.results_io.get_data_filename(f'params', results_type='session', format='npz')
        params_cached = np.load(params_path, allow_pickle=True)
        params_matched = []
        for k, v in params_cached.items():
            params_matched.append(getattr(self, k) == v)

        return all(params_matched)

    def _data_exists(self):
        files_exist = []
        for name, file_info in self.data_files.items():
            path = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])
            files_exist.append(path.is_file())

        return all(files_exist)

    def _train_test_split_ok(self, train_data, test_data):
        train_values = np.sort(train_data[self.feature_names[0]].unique())
        test_values = np.sort(test_data[self.feature_names[0]].unique())

        train_values = np.delete(train_values, train_values == 0)  # remove failed trials bc n/a to train/test check
        test_values = np.delete(test_values, test_values == 0)

        return all(train_values == test_values)

    def _train_test_split(self, random_state=21):
        # test data includes update/non-update trials, would focus only on non-update trials for error summary so the
        # test size refers only to the non-update trials in the decoding vs encoding phase

        # get encoder/training data times
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.encoder_trial_types.items()], axis=1).all(axis=1)
        encoder_trials = self.trials[mask]

        # get decoder/testing data times (decoder conditions + remove any training data)
        mask = pd.concat([self.trials[k].isin(v) for k, v in self.decoder_trial_types.items()], axis=1).all(axis=1)
        decoder_trials = self.trials[mask]

        mask = pd.concat([self.trials[k].isin(v) for k, v in self.decoder_trial_types_delay_only.items()], axis=1).all(axis=1)
        decoder_trials_delay_only = self.trials[mask]

        mask = pd.concat([self.trials[k].isin(v) for k, v in self.decoder_trial_types_update_only.items()], axis=1).all(axis=1)
        decoder_trials_update_only = self.trials[mask]

        # split if needed
        if self.decoder_test_size == 1:
            train_df = encoder_trials
            test_df = decoder_trials
            test_delay_only_df = decoder_trials_delay_only
            test_update_only_df = decoder_trials_update_only
        else:
            train_data, test_data = train_test_split(encoder_trials, test_size=self.decoder_test_size,
                                                     random_state=random_state)
            train_df = train_data.sort_index()
            test_df = decoder_trials[~decoder_trials.index.isin(train_df.index)]  # remove any training data
            test_delay_only_df = decoder_trials_delay_only[~decoder_trials_delay_only.index.isin(train_df.index)]
            test_update_only_df = decoder_trials_update_only[~decoder_trials_update_only.index.isin(train_df.index)]

        # check that split is ok for binarized data, run different random states until it is
        if self.convert_to_binary and not self._train_test_split_ok(train_df, test_df):
            train_df, test_df = self._train_test_split(random_state=random_state + 1)  # use new random state

        return train_df, test_df, test_delay_only_df, test_update_only_df

    def _preprocess(self):
        # get spikes
        units_mask = pd.concat([self.units[k].isin(v) for k, v in self.units_types.items()], axis=1).all(axis=1)
        units_subset = self.units[units_mask]
        if self.subset_reg:
            path = self.results_io.get_results_path(results_type='response')
            file_path = path / 'unit_counts_per_region.xlsx'
            df = pd.read_excel(file_path)#loading in the excel spreadsheet with all of the unit numbers for each region from each nwb session
            df['file_without_ext'] = df['File'].str.replace('.nwb', '', regex=False)#removing .nwb from each session name
            # Find the row where the 'file' column matches the session_id
            matching_row = df[df['file_without_ext'] == self.results_io.session_id]
            # Extract the number of units for CA1 and PFC (assuming the columns are named 'CA1_units' and 'PFC_units')
            if not matching_row.empty:
                ca1_units = matching_row['CA1'].values[0]
                pfc_units = matching_row['PFC'].values[0]
                print(f"matching session ID found for {self.results_io.session_id} CA1 units: {ca1_units}, PFC units: {pfc_units}")
                if ca1_units > pfc_units and self.region == "CA1":
                    ca1_mask = np.random.choice(np.arange(units_subset),size=pfc_units, replace=False)
                    data = {
                        'session_id': [session_id] * len(units_subset),
                        'ca1_mask': [True if i in ca1_mask else False for i in range(len(ca1_mask))],  # Example CA1 mask
                    }
                    # Create a pandas DataFrame
                    df2 = pd.DataFrame(data)
                    output_file_path = self.results_io.get_data_filename(filename=f'unit_masks_{self.results.session_id}',
                                                                         results_type='response', 
                                                                         format='csv')
                    df2.to_csv(output_file_path, index=False)
                    print(f"Mask data for session {self.results.session_id} saved to {output_file_path}")
                    units_subset=units_subset[ca1_mask]
                    
                elif pfc_units > ca1_units and self.region == "PFC":
                    pfc_mask = np.random.choice(np.arange(units_subset),size=ca1_units, replace=False)
                    data = {
                        'session_id': [session_id] * len(units_subset),
                        'pfc_mask': [True if i in pfc_mask else False for i in range(len(pfc_mask))],  # Example CA1 mask
                    }
                    # Create a pandas DataFrame
                    df2 = pd.DataFrame(data)
                    output_file_path = self.results_io.get_data_filename(filename=f'unit_masks_{self.results.session_id}',
                                                                         results_type='response', 
                                                                         format='csv')
                    df2.to_csv(output_file_path, index=False)
                    print(f"Mask data for session {self.results.session_id} saved to {output_file_path}")
                    units_subset=units_subset[pfc_mask]

        spikes_dict = {n: nap.Ts(t=units_subset.loc[n, 'spike_times'], time_units='s') for n in units_subset.index}
        if spikes_dict:
            self.spikes = nap.TsGroup(spikes_dict, time_units='s')
        else:
            self.spikes = []  # if no units match criteria, leave spikes empty

        # # split data into training/encoding and testing/decoding trials
        
        #categorical label for stratification (check this, not 100% sure about this)
        y = self.data['trial_type']
        num_splits = 5
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
        # Create an empty list to store all splits
        self.splits_data = []
        
        # Loop over the splits
        for fold, (train_index, test_index) in enumerate(skf.split(self.data, y)):
            print(f"Running fold {fold + 1}")
            # Split the data into training/encoding and testing/decoding
            train_df = self.data.iloc[train_index]
            test_df = self.data.iloc[test_index]
            # Get time intervals of train and test trials
            encoder_times = self._get_time_intervals(train_df['start_time'], train_df['stop_time'])
            decoder_times = self._get_time_intervals(test_df['start_time'], test_df['stop_time'], align_times=test_df['t_update'])
            # Optionally handle specific subsets (delay-only, update-only). leaving in for now to match Steph's set up, but not sure it's needed
            test_delay_only_df = test_df[test_df['trial_type'] == 'delay']
            test_update_only_df = test_df[test_df['trial_type'] == 'update']
            
            decoder_times_delay_only = self._get_time_intervals(test_delay_only_df['start_time'],
                                                                test_delay_only_df['stop_time'],
                                                                align_times=test_delay_only_df['t_update'])
            decoder_times_update_only = self._get_time_intervals(test_update_only_df['start_time'],
                                                                test_update_only_df['stop_time'],
                                                                align_times=test_update_only_df['t_update'])
            # Store everything in self.splits_data for future use
            split_data = {
                'fold': fold,
                'train_df': train_df,
                'test_df': test_df,
                'encoder_times': encoder_times,
                'decoder_times': decoder_times,
                'decoder_times_delay_only': decoder_times_delay_only,
                'decoder_times_update_only': decoder_times_update_only
            }
            self.splits_data.append(split_data)
        return self

    def _encode(self):
        # if there are no units/spikes to use for encoding, create empty dataframe as default
        
        # Create empty lists to store models for each split
        self.models = []
        self.models_test = []
        self.models_delay_only = []
        self.models_update_only = []
        self.bins_list = []
        
        # Iterate through each split from the stratified data
        for split_data in self.splits_data:
            # Default to empty DataFrames for each split
            model = pd.DataFrame()
            model_test = pd.DataFrame()
            model_delay_only = pd.DataFrame()
            model_update_only = pd.DataFrame()
            bins = []
            if self.spikes:  # If there were units and spikes to use for encoding
                if self.dim_num == 1:
                    # Train encoding model
                    feat_input = split_data['train_df'][self.feature_names[0]]
                    bins = np.linspace(*self.limits[self.feature_names[0]], self.encoder_bin_num + 1)
                    model = self.encoder(group=self.spikes, feature=feat_input, nb_bins=self.encoder_bin_num,
                                        ep=split_data['encoder_times'], minmax=self.limits[self.feature_names[0]])
                    # Test model on all test trials
                    model_test = self.encoder(group=self.spikes, feature=split_data['test_df'][self.feature_names[0]],
                                            nb_bins=self.encoder_bin_num, ep=split_data['decoder_times'],
                                            minmax=self.limits[self.feature_names[0]])
                    # Test model on delay-only trials
                    model_delay_only = self.encoder(group=self.spikes,
                                                    feature=split_data['test_delay_only_df'][self.feature_names[0]],
                                                    nb_bins=self.encoder_bin_num, ep=split_data['decoder_times_delay_only'],
                                                    minmax=self.limits[self.feature_names[0]])
                    # Test model on update-only trials if they exist
                    if np.size(split_data['test_update_only_df']):
                        model_update_only = self.encoder(group=self.spikes,
                                                        feature=split_data['test_update_only_df'][self.feature_names[0]],
                                                        nb_bins=self.encoder_bin_num, ep=split_data['decoder_times_update_only'],
                                                        minmax=self.limits[self.feature_names[0]])
                elif self.dim_num == 2:
                # If using 2-dimensional features: not planning on using but including to match Steph set up
                    model, bins = self.encoder(group=self.spikes, feature=split_data['train_df'],
                                            nb_bins=self.encoder_bin_num, ep=split_data['encoder_times'])
                    
            # Store models and bins for each split
            self.models.append(model)
            self.models_test.append(model_test)
            self.models_delay_only.append(model_delay_only)
            self.models_update_only.append(model_update_only)
            self.bins_list.append(bins)

        return self

    def _decode(self):      
        # Create lists to store decoded values and probabilities for each split
        self.decoded_values = pd.DataFrame()
        self.decoded_probs = pd.DataFrame()
        # Iterate through each split from the stratified data
        for split_data, model in zip(self.splits_data, self.models):
            # Get arguments for the decoder
            kwargs = dict(tuning_curves=model, group=self.spikes, ep=split_data['decoder_times'],
                        bin_size=self.decoder_bin_size)
            
            if self.dim_num == 2:
                kwargs.update(binsxy=self.bins_list[self.splits_data.index(split_data)])
                
            # Apply history-dependent prior if applicable
            if self.prior == 'history-dependent':
                if self.dim_num == 1:
                    kwargs.update(feature=split_data['train_df'])
                elif self.dim_num == 2:
                    kwargs.update(features=split_data['train_df'])
                    
            # Run decoding for this split
            if model.any().any():
                decoded_values, decoded_probs = self.decoder(**kwargs)  
                # Handle edge cases where there are empty/zero bins in the model
                edge_spaces = np.squeeze(np.argwhere(np.sum(model, axis=1).to_numpy() == 0))
                decoded_probs.iloc[:, edge_spaces] = np.nan
                # Concatenate the results for each split into the final DataFrames
                self.decoded_values = pd.concat([self.decoded_values, decoded_values], axis=0)
                self.decoded_probs = pd.concat([self.decoded_probs, decoded_probs], axis=0)
            else:
                # If no valid model, append empty DataFrames
                self.decoded_values = pd.concat([self.decoded_values, pd.DataFrame()], axis=0)
                self.decoded_probs = pd.concat([self.decoded_probs, pd.DataFrame()], axis=0)
        # Optionally convert to binary
        if self.convert_to_binary:
            self.decoded_values.values[self.decoded_values > 0] = int(1)
            self.decoded_values.values[self.decoded_values < 0] = int(-1)   

        return self

