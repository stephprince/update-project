import pickle

from git import Repo
from pathlib import Path


class ResultsIO:
    git_hash = Repo(search_parent_directories=True).head.object.hexsha[:10]
    base_path = Path().absolute().parent.parent / 'results'

    def __init__(self, creator_file, git_hash=git_hash, base_path=base_path, session_id='', folder_name='', tags=''):
        self.creator_file = creator_file  # to add to metadata to know which script generated the figure
        self.git_hash = git_hash  # to add to filename to know which git commit generated the figure
        self.base_path = base_path / folder_name
        self.session_id = session_id
        self.animal = session_id.split('_')[0]
        self.tags = tags

    @staticmethod
    def load_pickled_data(fname):
        with open(fname, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def get_data_filename(self, filename, results_type='group', results_name='', format='npy'):
        data_path = self.get_data_path(results_type, results_name)
        fname = data_path / f'{filename}_{self.tags}.{format}'  # only use git hash for figs, not for intermediate data

        return fname

    def get_data_path(self, results_type='group', results_name=''):
        data_path = self.get_results_path(results_type, results_name=results_name) / 'intermediate_data'
        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path

    def get_results_path(self, results_type='group', results_name=''):
        if results_type == 'group':
            results_path = self.base_path / 'group_summary' / results_name
        elif results_type == 'session':
            assert self.session_id is not None, "No session id provided so cannot create session folder"
            results_path = self.base_path / self.session_id / results_name

        Path(results_path).mkdir(parents=True, exist_ok=True)

        return results_path

    def get_figure_args(self, filename, results_type='group', additional_tags='', results_name='', format='pdf'):
        results_path = self.get_results_path(results_type, results_name)
        extra_tags = f'_{additional_tags}'
        if self.tags == '' and additional_tags == '':
            fname = results_path / f'{filename}_git{self.git_hash}.{format}'
        else:
            fname = results_path / f'{filename}_{self.tags}{extra_tags}_git{self.git_hash}.{format}'

        kwargs = dict(fname=fname,
                      dpi=300,
                      format=format,
                      metadata={'Creator': self.creator_file},
                      )

        return kwargs

    def load_data(self, filename, results_type='group', format='npy'):
        fname = self.get_data_filename(filename, results_type=results_type, format=format)
        data = self.load_pickled_data(fname)

        return data

    def export_data(self, data, filename, results_type='group', format='npy'):
        fname = self.get_data_filename(filename, results_type=results_type, format=format)
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    def data_exists(self, data_files):
        files_exist = []
        for name, file_info in data_files.items():
            path = self.get_data_filename(filename=name, results_type='session', format=file_info['format'])
            files_exist.append(path.is_file())

        return all(files_exist)
