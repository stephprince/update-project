%% monitorUpdateTaskBehavior
% use this script to monitor behavioral performance on the update task
% during training

%% sets up animals, dates, directories, flags
indices.animals = [20];
indices.excldates = []; indices.incldates = [];

%gets data directories and indices
dirs = getDefaultUpdateTaskDirectories;
indices = getdefaultindiceshpcpfcinteractions(indices,dirs);

%set flags for what to look at
makenewfiles = 1

%% perform behavioral analyses
getUpdateTaskBehavior(dirs, indices, debugmode, specificdataset, makenewfiles); %note, behavior uses different indices then the rest of the analyses  because there are more dates outside of ephys days

