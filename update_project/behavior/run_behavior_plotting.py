import matplotlib.pyplot as plt

from pathlib import Path
from pynwb import NWBHDF5IO

#%%
# load the nwb files
nwb_filename = Path('../../data//test_behavior.nwb')

io = NWBHDF5IO(filename,'r')
nwbfile = io.read()

### compile behavioral data



### plot behavioral performance

# make violin plot of performance across trial types


### plot trajectories

# make line plot of individual and average trajectories around update period

