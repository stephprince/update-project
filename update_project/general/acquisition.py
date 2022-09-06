import numpy as np
import pandas as pd


def get_velocity(nwbfile):
    rotational_velocity = nwbfile.acquisition['rotational_velocity'].data
    translational_velocity = nwbfile.acquisition['translational_velocity'].data
    velocity = np.abs(rotational_velocity[:]) + np.abs(translational_velocity[:])
    rate = nwbfile.acquisition['translational_velocity'].rate
    timestamps = np.arange(0, len(velocity) / rate, 1 / rate)
    velocity = pd.Series(index=timestamps[:], data=velocity)

    return velocity