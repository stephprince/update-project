import numpy as np
import pickle

from abc import abstractmethod, ABC
from pynwb import NWBFile


class BaseAnalysisClass(ABC):
    def _load_data(self):
        print(f'Loading existing data for session {self.results_io.session_id}...')

        # load npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            if file_info['format'] == 'npz':
                import_data = np.load(fname, allow_pickle=True)
                for v in file_info['vars']:
                    setattr(self, v, import_data[v])
            elif file_info['format'] == 'pkl':
                import_data = self.results_io.load_pickled_data(fname)
                for v, data in zip(file_info['vars'], import_data):
                    setattr(self, v, data)
            else:
                raise RuntimeError(f'{file_info["format"]} format is not currently supported for loading data')

        return self

    def _export_data(self):
        print(f'Exporting data for session {self.results_io.session_id}...')

        # save npz files
        for name, file_info in self.data_files.items():
            fname = self.results_io.get_data_filename(filename=name, results_type='session', format=file_info['format'])

            if file_info['format'] == 'npz':
                kwargs = {v: getattr(self, v) for v in file_info['vars']}
                np.savez(fname, **kwargs)
            elif file_info['format'] == 'pkl':
                with open(fname, 'wb') as f:
                    [pickle.dump(getattr(self, v), f) for v in file_info['vars']]
            else:
                raise RuntimeError(f'{file_info["format"]} format is not currently supported')

    @abstractmethod
    def _setup_data(self, nwbfile: NWBFile):
        """Child AnalysisInterface classes should override this to pull their specific data"""
        pass

    @abstractmethod
    def run_analysis(self, overwrite: bool = False):
        """Child AnalysisInterface classes should override this to run their specific analysis"""
        raise NotImplementedError('The run_analysis method for this AnalysisInterface has not been defined!')




