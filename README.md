# update-project

## overview

The goal of this project is to look at neural activity from hippocampal CA1 and medial prefrontal cortex in mice during 
the update virtual-reality task. This project is currently in development.

## getting started

To get started using this project, follow the instructions below. First, clone this repository. 

```bash
git clone "https://github.com/stephprince/update-project.git"
```

Then, create a new conda environment to work with the data. Within this environment, you can then install the package in
development mode. This command will also install all dependencies.
```bash
conda create --name update-project
conda activate update-project

cd /path/to/files/update-project/
pip install -e .
```

To generate figures for the manuscript, run the main figure-generating script.
All figures will save in a local update-project/results/manuscript-figures folder.
(Note: these commands assume data is stored in the [project structures and locations](#project-structure-and-locations) 
format described below. Future examples will include instructions for data deposited online.)

```bash
python run_all_figures.py
```

If you would like to run any of the modules individually, 
or generate figures not included in the final version of the paper, 
each analysis stream has its own submodule.
(e.g. `run_decoding.py` found in the update_project/decoding folder)

## project structure and locations

- **raw data**
    - location: `/ad.gatech.edu/bme/labs/singer/RawData/UpdateTask`
    - format: acquired from SpikeGadgets, stored as `.rec` 
- **processed data**
    - location: `/ad.gatech.edu/bme/labs/singer/ProcessedData/UpdateTask`
    - format: stored in the Singer Lab data format as `.mat` 
- **imaging data**
    - location: `/ad.gatech.edu/bme/labs/singer/HistologyImages/Steph/UpdateTask`
    - format: stored in Zeiss data format as  `.czi` 
- **behavioral data**
    - location: `/ad.gatech.edu/bme/labs/singer/VirmenLogs/UpdateTask`
    - format: stored in custom virmen output format as  `.mat` 
- **camera data**
    - location: `/ad.gatech.edu/bme/labs/singer/CameraData/UpdateTask`
    - format: stored in custom virmen output format as  `.mp4` 
- **NWB data**
    - location: `/ad.gatech.edu/bme/labs/singer/NWBData/UpdateTask`
    - format: stored in neurodata without borders format as `.nwb`
- **analysis code and results**: 
    - location: this folder, `/ad.gatech.edu/bme/labs/singer/Steph/Code/update-project`
    - sub folders:
        - **doc/** - contains relevant documentation files (behavior csvs, methodology, manuscript drafts, etc.)
                - ephys recording summary: `/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv`
                - behavior summary: `/docs/metadata-summaries/VRUpdateTaskBehaviorSummary.csv`
                - this folder also contains several important notes about the processing pipelines and things to consider when analyzing the data
        - **results/** -  contains relevant output figures and intermediary data structures, not stored on github
        - **requirements.txt** -  contains the minimal package requirements for the project.
        - **update-project/** -  contains relevant matlab and python code
- **brain tissue**:
    - **brains and slices**: -20 fridge in clear plastic box with mouse labels starting with S
    - **slides**: -20 fridge in box labelled Steph Prince - update project

## related papers

preprint coming soon...
