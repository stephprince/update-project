import more_itertools as mit
import numpy as np

def get_largest_field_loc(place_field_bool):
    result = list(mit.run_length.encode(place_field_bool))
    biggest_field = np.nanmax([count if f else np.nan for f, count in result])

    field_ind = 0
    largest_field_ind = np.nan
    for f, count in result:
        if f and (count == int(biggest_field)):
            largest_field_ind = field_ind + 1
        else:
            field_ind = field_ind + count

    return largest_field_ind


def get_place_fields(tuning_curves):
    # get goal selective cells (cells with a place field in at least one of the choice locations)
    place_field_thresholds = tuning_curves.apply(lambda x: x > (np.mean(x) + np.std(x)))
    place_fields_2_bins = place_field_thresholds.rolling(window=2).mean() > 0.5
    bins_shifted = place_fields_2_bins.shift(periods=-1, axis=0, )
    bins_shifted.iloc[-1, :] = place_fields_2_bins.iloc[-1, :]
    place_fields = np.logical_or(place_fields_2_bins, bins_shifted).astype(bool)

    return place_fields