import numpy as np
import pandas as pd
from update_project.general.virtual_track import UpdateTrack


def get_velocity(nwbfile):
    rotational_velocity = nwbfile.acquisition['rotational_velocity'].data
    translational_velocity = nwbfile.acquisition['translational_velocity'].data
    velocity = np.abs(rotational_velocity[:]) + np.abs(translational_velocity[:])
    rate = nwbfile.acquisition['translational_velocity'].rate
    timestamps = np.arange(0, len(velocity) / rate, 1 / rate)

    velocity = pd.DataFrame(dict(combined=pd.Series(index=timestamps[:], data=velocity),
                                 rotational=pd.Series(index=timestamps[:], data=rotational_velocity),
                                 translational=pd.Series(index=timestamps[:], data=translational_velocity)))

    return velocity


def get_licks(nwbfile):
    lick_voltage = nwbfile.acquisition['licks'].data
    rate = nwbfile.acquisition['licks'].rate
    timestamps = np.arange(0, len(lick_voltage) / rate, 1 / rate)

    licks = pd.DataFrame(dict(licks=pd.Series(index=timestamps[:], data=lick_voltage),))

    return licks

def get_location(nwbfile):
    virtual_track = UpdateTrack(linearization=bool(['y_position']))
    time_series = nwbfile.processing['behavior']['position'].get_spatial_series('position')
    raw_data = time_series.data[:]
    data = virtual_track.linearize_track_position(raw_data)
    
    location = pd.DataFrame(dict(location=pd.Series(index=time_series.timestamps[:], data=data),))
    return location