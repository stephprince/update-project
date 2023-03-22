import numpy as np
import pandas as pd
import pynapple as nap
import warnings

from bisect import bisect, bisect_left
from pathlib import Path
from pynwb import NWBFile
from sklearn.model_selection import train_test_split

from update_project.general.results_io import ResultsIO
from update_project.general.virtual_track import UpdateTrack
from update_project.general.lfp import get_theta
from update_project.general.acquisition import get_velocity
from update_project.general.trials import get_trials_dataframe
from update_project.base_analysis_class import BaseAnalysisClass


class ExperimentMetadataAnalyzer(BaseAnalysisClass):
    def __init__(self, nwbfile: NWBFile, session_id: str):
        # setup parameters
        self.results_io = ResultsIO(creator_file=__file__,
                                    session_id=session_id,
                                    folder_name=Path(__file__).parent.stem)
        self.data_files = dict(bayesian_decoder_output=dict(vars=['metadata_df'], format='pkl'), )
        self.coordinate_file_path = Path(__file__).parent.parent.parent / 'docs' / 'metadata-summaries' / \
                                    'UpdateTaskProbeCoordinatesSummary.csv'
        self._setup_data(nwbfile)

    def run_analysis(self, overwrite=False, export_data=True):
        print(f'Getting metadata for session {self.results_io.session_id}...')

        if overwrite:
            self._get_metadata_df()  # build model
            if export_data:
                self._export_data()  # save output data
        else:
            if self._data_exists():
                self._load_data()  # load data structure if it exists and matches the params
            else:
                warnings.warn('Data does not exist, setting overwrite to True')
                self.run_analysis(overwrite=True, export_data=export_data)

        return self

    def _setup_data(self, nwbfile):
        self.trials = get_trials_dataframe(nwbfile)
        self.units = nwbfile.units.to_dataframe()
        self.coordinates = pd.read_csv(self.coordinate_file_path,
                                       encoding='cp1252')  # TODO - at some point add this to the NWB files so can pull straight from there

        return

    def _get_metadata_df(self):
        # get trial counts
        trial_counts = (self.trials
                        .assign(update_type=lambda x: x['update_type'].map({1: 'delay only', 2: 'switch', 3: 'stay'}))
                        .groupby('update_type')['start_time']
                        .count()
                        .rename(lambda x: f'{x} trials'))
        trial_counts['total trials'] = np.shape(self.trials)[0]

        # get unit counts
        unit_counts = (self.units
                       .groupby('region')['ContamPct']
                       .count()
                       .rename(lambda x: f'{x} units'))
        unit_counts['total units'] = np.shape(self.units)[0]

        # get locations
        coordinate_locations = (self.coordinates
                                .query(f'animal == "{self.results_io.animal}"')
                                .groupby('region')
                                .apply(lambda grp: {f'{x["name"]}_{i}': {'A/P': x['AP_location'],
                                                                         'D/V': x['DV_location'],
                                                                         'M/L': x['ML_location']}
                                                    for i, x in grp.iterrows()})
                                .rename(lambda x: f'{x} location'))
        if np.size(coordinate_locations):
            pass
        else:
            coordinate_locations = pd.Series({'CA1 location': 'Slices damaged', 'PFC location': 'Slices damaged'})

        # concatenate data
        self.metadata_df = pd.concat([trial_counts, unit_counts, coordinate_locations], axis=0)
        self.metadata_df['session_id'] = self.results_io.session_id
        self.metadata_df['animal'] = self.results_io.animal
        self.metadata_df = self.metadata_df.iloc[::-1]  # reverse order

        return self

    def _data_exists(self):
        files_exist = []
        for name, file_info in self.data_files.items():
            path = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])
            files_exist.append(path.is_file())

        return all(files_exist)
