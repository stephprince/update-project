function indices = getdefaultindiceshpcpfcinteractions(indices, dirs, behavior, ephys)
% this function gets the index matrices for behavior, ephys recordings,
% etc. from the spreadsheets defined in
% 'getdefaultdirectorieshpcpfcinteractions function

% inputs:
%       animals - animals to get data for
%       excldates - dates to not grab, used to exclude specific files
%       incldates - dates to grab, used to include only specific files
% outputs:
%       indices - structure with different index matrices for behavior,
%       ephys, etc.

indices.animalID = 'S'; %puts S identifier in front of number for filenames

%% behavior
behaviorindex = getUpdateTaskIndex(indices.animals, dirs.behaviorspreadsheetfname, indices.animalID);

if ~isempty(indices.excldates)
    behaviorindex = behaviorindex(~ismember(behaviorindex{:,2},indices.excldates),:);
end
if ~isempty(indices.incldates)
    behaviorindex = behaviorindex(ismember(behaviorindex{:,2},indices.incldates),:);
end

indices.behaviorindex = behaviorindex; %[animal# date session# ephys/noephys]

%% ephys
%get the animal info based on the inputs
ephysindexT = selectindextable(dirs.ephysspreadsheetfname, 'animal', indices.animals,...
    'datesincluded', indices.incldates, 'datesexcluded', indices.excldates,...
    'behavior', behavior);
allindex = ephysindexT{:,{'Animal', 'Date','Recording'}};

%% fix the behavior index if wanting to look at only ephys data
if ephys
    behaviorWithEphys = indices.behaviorindex(ismember(indices.behaviorindex{:,1:3}, ephysindexT{:,2:4}, 'rows'),:);
    indices.behaviorindex = behaviorWithEphys;
end

end
