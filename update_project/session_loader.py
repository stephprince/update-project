import pandas as pd

from pathlib import Path


class SessionLoader:
    def __init__(self, animals, base_path=None, csv_filename=None, dates_included=None, dates_excluded=None,
                 behavior_only=False):
        self.base_path = base_path or Path('Y:/singer/NWBData/UpdateTask/')  # if no value given, use default path
        self.csv_filename = csv_filename or 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'

        self.behavior_only = behavior_only
        if self.behavior_only:
            behavior_spreadsheet_filename = 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskBehaviorSummary.csv'
            self.behavior_csv_filename = behavior_spreadsheet_filename

        self.animals = animals
        self.dates_included = dates_included
        self.dates_excluded = dates_excluded

    @staticmethod
    def get_animal_id(session_name):
        return session_name[1]

    @staticmethod
    def get_session_id(session_name):
        session_id = f"{session_name[0]}{session_name[1]}_{session_name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913

        return session_id

    def get_session_path(self, session_name):
        session_path = f'{self.base_path / self.get_session_id(session_name)}.nwb'

        return session_path

    def get_session_info(self):
        # import all session info
        df_ephys = pd.read_csv(self.csv_filename, skiprows=[1],
                             encoding='cp1252')  # skip the first row that's just the detailed header info

        if self.behavior_only:
            df_behavior = pd.read_csv(self.behavior_csv_filename)
            df_behavior['Animal'] = df_behavior['Animal'].str.replace('[Ss]', '').astype(
                int)  # convert animal ids into int
            df_behavior['ID'] = 'S'  # add separate ID column with the S tag

            dob_mapping = df_ephys.groupby(['Animal', 'DOB']).size().reset_index()
            df_all = df_behavior.merge(dob_mapping[['Animal', 'DOB']], on='Animal', how='outer')
        else:
            df_all = df_ephys[df_ephys['Include'] == 1]  # only use included ephys sessions

        # if None values, deal with appropriately so it doesn't negatively affect the filtering
        dates_incl = self.dates_included or df_all['Date']  # if no value given, include all dates
        dates_excl = self.dates_excluded or [None]  # if no value given, exclude no dates

        # filter session info depending on cases
        session_info = df_all[(df_all['Animal'].isin(self.animals)) &  # IS IN the animals list
                              (df_all['Date'].isin(dates_incl)) &  # IS IN the included dates list
                              ~(df_all['Date'].isin(dates_excl))  # NOT IN the excluded dates list
                              ]

        return session_info

    def load_sessions(self):
        all_session_info = self.get_session_info()
        unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

        return unique_sessions

    def load_session_names(self):
        unique_sessions = self.load_sessions()
        names = list(unique_sessions.groups.keys())

        return names
