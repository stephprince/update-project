%% script_hpcpfcinteractions
% SP 190813
% this script will generate all the analyses for the hpc-pfc diamond maze analyses

%% sets up animals, dates, directories
indices.animals = [13];
indices.excldates = []; indices.incldates = [];

%gets data directories and indices
dirs = getdefaultdirectorieshpcpfcinteractions;
indices = getdefaultindiceshpcpfcinteractions(indices,dirs);

%set flags for what you want to do
makenewfiles = 0; % this flag generates new data structures, when set to 0 it uses preexisting data structures
behavior = 1;

%% perform behavioral analyses
if behavior
    getUpdateTaskBehavior(dirs, indices, makenewfiles); %note, behavior uses different indices then the rest of the analyses  because there are more dates outside of ephys days
end

%% rest of analyses here
