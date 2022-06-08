from git import Repo
from pathlib import Path


class ResultsIO:
    git_hash = Repo(search_parent_directories=True).head.object.hexsha[:10]
    base_path = Path().absolute().parent.parent / 'results'

    def __init__(self, creator_file, git_hash=git_hash, base_path=base_path, session_id=None, folder_name='', tags=''):
        self.creator_file = creator_file  # to add to metadata to know which script generated the figure
        self.git_hash = git_hash  # to add to filename to know which git commit generated the figure
        self.base_path = base_path / folder_name
        self.session_id = session_id
        self.animal = session_id.split('_')[0]
        self.tags = tags

    def get_results_path(self, results_type='group', results_name=''):
        if results_type == 'group':
            results_path = self.base_path / 'group_summary' / results_name
        elif results_type == 'session':
            assert self.session_id is not None, "No session id provided so cannot create session folder"
            results_path = self.base_path / self.session_id / results_name

        Path(results_path).mkdir(parents=True, exist_ok=True)

        return results_path

    def get_data_path(self, results_type='group', results_name=''):
        data_path = self.get_results_path(results_type, results_name=results_name) / 'intermediate_data'
        Path(data_path).mkdir(parents=True, exist_ok=True)

        return data_path

    def get_figure_args(self, filename, results_type='group', results_name='', format='pdf'):
        results_path = self.get_results_path(results_type, results_name)
        fname = results_path / f'{filename}_{self.tags}_git{self.git_hash}.{format}'

        kwargs = dict(fname=fname,
                      dpi=300,
                      format=format,
                      metadata={'Creator': self.creator_file},
                      )

        return kwargs

    def get_data_filename(self, filename, results_type='group', results_name='', format='npy'):
        data_path = self.get_data_path(results_type, results_name)
        fname = data_path / f'{filename}_{self.tags}_git{self.git_hash}.{format}'

        return fname