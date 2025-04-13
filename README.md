# update-project

## overview

### summary
This repository contains the custom code and analysis pipelines to generate the results in the manuscript Prince et al., "New information triggers prospective codes to adapt for flexible navigation."

### abstract
Navigating a dynamic world requires rapidly updating choices by integrating past experiences with new information. In hippocampus and prefrontal cortex, neural activity representing future goals is theorized to support planning. However, it remains unknown how prospective goal representations incorporate new, pivotal information. Accordingly, we designed a novel task that precisely introduces new information using virtual reality, and we recorded neural activity as mice flexibly adapted their planned destinations. We found that new information triggered increased hippocampal prospective representations of both possible goals; while in prefrontal cortex, new information caused prospective representations of choices to rapidly shift to the new choice. When mice did not flexibly adapt, prefrontal choice codes failed to switch, despite relatively intact hippocampal goal representations. Prospective code updating depended on the commitment to the initial choice and degree of adaptation needed. Thus, we show how prospective codes update with new information to flexibly adapt ongoing navigational plans.

## system requirements

### hardware requirements
This package requires only a standard computer with enough RAM to support the in-memory operations. These analyses were primarily performed on a computer with the following specs:

- RAM: 256 GB
- CPU: 18 cores @ 3.00 GHz

### software requirements

The code was developed and tested on a Windows 10 system. Dependencies are detailed in the package file. Analyses were performed with Python >= 3.9 and R >= 4.1.3.

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
your local environment variables to point towards that path before installing rpy2). The installation steps typically take less than 20 min.

## how to use
To generate figures for the manuscript, run the main script `run_all_figures.py` in the terminal or your IDE.

```bash
python run_all_figures.py
```

All figures will save in the local `update-project/results/manuscript-figures` folder
(Note: the scripts assume all raw data is stored on an internal server at `/singer/NWBData/UpdateTask`,
future examples will include instructions for data deposited online).

