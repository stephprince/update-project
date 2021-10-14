
%% script_hpcpfcinteractions
% SP 190813
% this script will generate all the analyses for the hpc-pfc diamond maze analyses

%% sets up animals, dates, directories
indices.animals = [17,20,25];
indices.excldates = []; indices.incldates = [];

%set flags for what you want to do
debugmode = 0; %loads up a smaller sample dataset to work through new ideas
specificdataset = 0; %loads a specific date iteration
overwrite = 0; % this flag generates new data structures, when set to 0 it uses preexisting data structures
behavior = 1;
ephys = 1; %run only sessions with corresponding ephys data

%gets data directories and indices
dirs = getdefaultdirectorieshpcpfcinteractions;
indices = getdefaultindiceshpcpfcinteractions(indices, dirs, behavior, ephys);

%% perform behavioral analyses
if behavior
    getUpdateTaskBehavior(dirs, indices, debugmode, specificdataset, ephys, overwrite); %note, behavior uses different indices then the rest of the analyses  because there are more dates outside of ephys days
end

%% rest of analyses here
