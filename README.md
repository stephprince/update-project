# update-project

## overview

The goal of this project is to look at neural activity from hippocampal CA1 and medial prefrontal cortex when mice
 have to update their trajectories in response to new information. This project is currently in development.

## installation

To get started using this project, follow the instructions below. First, clone or download this repository. 

```bash
git clone "https://github.com/stephprince/update-project.git"
```

Then, create a new conda environment to work with the data. 
Within this environment, you can then install the package in development mode 
(package dependencies will be automatically installed).
```bash
conda create --name update-project python==3.9
conda activate update-project

cd /path/to/files/update-project/
pip install -e .
```

To install R and the relevant r-python interface for the statistical analyses, run the commands below.

```bash
conda install -c conda-forge r r-base r-lmertest r-emmeans rpy2
```
After running these commands, edit your local environment variables so `R_HOME = /path/to/Anaconda3/envs/update-project/lib/R`
(Note: if you are using another package manager like pip/poetry, you will have to setup a local R installation and edit 
your local environment variables to point towards that path before installing rpy2). 

## how to use
To generate figures for the manuscript, run the main script `run_all_figures.py` in the terminal or your IDE.

```bash
python run_all_figures.py
```

All figures will save in the local `update-project/results/manuscript-figures` folder
(Note: the scripts assume all raw data is stored on the internal server at `/singer/NWBData/UpdateTask`,
future examples will include instructions for data deposited online).

If you would like to run any of the modules individually, 
or generate figures not included in the final version of the paper, 
each analysis stream has its own submodule.
(e.g. `run_decoding.py` found in the update_project/decoding folder)

## project details
These paths describe the location of code, data, and materials within the Singer lab and is mainly relevant for
internal use.
Future examples will include instructions for data deposited online.

- **analysis code and results**:  `/singer/Steph/Code/update-project` (this repository)
    - **update-project/** -  contains python code
    - **pyproject.toml** -  contains package info and project requirements
    - **tests/** - contains tests for code
    - **results/** -  contains relevant output figures and intermediary data structures, not stored on github
    - **docs/** - contains relevant documentation and metadata files
      - ephys recording summary: `/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv`
      - behavior summary: `/docs/metadata-summaries/VRUpdateTaskBehaviorSummary.csv`
      - this folder also contains notes on things to consider when analyzing the data
    - **scripts/** - contains early test scripts for establishing some parameters and verifying results
- **NWB data**: `/singer/NWBData/UpdateTask`, `.nwb` main data files compiling several of the data streams below into the NeurodataWithoutBorders format
- **raw data**: `/singer/RawData/UpdateTask`, `.rec` files acquired from SpikeGadgets ephys acquisition system
- **processed data**: `/singer/ProcessedData/UpdateTask`, `.mat` files from custom preprocessing/spike sorting pipeline
- **imaging data**: `/singer/HistologyImages/Steph/UpdateTask`, `.czi` files acquired from Zeiss microscope
- **behavioral data**: `/singer/VirmenLogs/UpdateTask`, `.mat` files acquired from ViRMEn behavioral software
- **camera data**: `/singer/CameraData/UpdateTask`, `.mp4` files acquired from Basler camera
- **brain tissue slides**: -20 fridge in box labelled "Steph Prince - update project"

## related papers

preprint coming soon...
