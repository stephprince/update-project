import pandas as pd

def get_session_info(filename, animals, dates_included=None, dates_excluded=None, behavior=None):
    # import all session info
    df_all = pd.read_csv(filename, skiprows=[1], encoding='cp1252')  # skip the first row that's just the detailed header info

    # if None values, deal with appropriately so it doesn't negatively affect the filtering
    dates_incl = dates_included or df_all['Date']                   # if no value given, include all dates
    dates_excl = dates_excluded or [None]                           # if no value given, exclude no dates
    behavior = behavior or df_all['Behavior'].unique()              # if no value given, include all behavior types

    # filter session info depending on cases
    session_info = df_all[(df_all['Include'] == 1) &                # DOES HAVE an include value in the column
                          (df_all['Animal'].isin(animals)) &        # IS IN the animals list
                          (df_all['Date'].isin(dates_incl)) &       # IS IN the included dates list
                          ~(df_all['Date'].isin(dates_excl)) &  # NOT IN the excluded dates list
                          (df_all['Behavior'].isin(behavior))       # IS IN the behavior type list
                          ]

    return session_info