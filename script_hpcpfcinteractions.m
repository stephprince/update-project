%% script_hpcpfcinteractions
% SP 190813
% this script will generate all the analyses for the hpc-pfc diamond maze analyses

%% sets up animals, dates, directories
animals = [7,8,9];
excldates = []; incldates = [];

%gets data directories and indices
dirs = getdefaultdirectorieshpcpfcinteractions;
indices = getdefaultindiceshpcpfcinteractions(animals,dirs,excldates,incldates); 

%set flags for what you want to do
makenewfiles = 1; % this flag generates new data structures, when set to 0 it uses preexisting data structures
behavior = 1;  

%% perform behavioral analyses
if behavior
    getDiamondTrackBehavior(dirs, indices, makenewfiles); %note, behavior uses different indices then the rest of the analyses  because there are more dates outside of ephys days
end

%% rest of analyses here