import pandas as pd


class SessionLoader:
    def __init__(self, animals, base_path=None, csv_filename=None, dates_included=None, dates_excluded=None,
                 behavior=None):
        self.base_path = base_path or Path('Y:/singer/NWBData/UpdateTask/')  # if no value given, use default path
        self.csv_filename = csv_filename or 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'

        self.animals = animals
        self.dates_included = dates_included
        self.dates_excluded = dates_excluded
        self.behavior = behavior

    def get_session_info(self):
        # import all session info
        df_all = pd.read_csv(self.csv_filename, skiprows=[1],
                             encoding='cp1252')  # skip the first row that's just the detailed header info

        # if None values, deal with appropriately so it doesn't negatively affect the filtering
        dates_incl = self.dates_included or df_all['Date']  # if no value given, include all dates
        dates_excl = self.dates_excluded or [None]  # if no value given, exclude no dates
        behavior = self.behavior or df_all['Behavior'].unique()  # if no value given, include all behavior types

        # filter session info depending on cases
        session_info = df_all[(df_all['Include'] == 1) &  # DOES HAVE an include value in the column
                              (df_all['Animal'].isin(self.animals)) &  # IS IN the animals list
                              (df_all['Date'].isin(dates_incl)) &  # IS IN the included dates list
                              ~(df_all['Date'].isin(dates_excl)) &  # NOT IN the excluded dates list
                              (df_all['Behavior'].isin(behavior))  # IS IN the behavior type list
                              ]

        return session_info

    def load_sessions(self):
        all_session_info = self.get_session_info()
        unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

        return unique_sessions

    def get_base_path(self):
        return self.base_path