function indices = getdefaultindiceshpcpfcinteractions(animals,dirs,excldates,incldates)
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
behaviorindex = getDiamondMazeIndex(animals, dirs.behaviorspreadsheetdir);
if ~isempty(excldates)
    behaviorindex = behaviorindex(~ismember(behaviorindex(:,2),excldates),:);
end
if ~isempty(incldates)
    behaviorindex = behaviorindex(ismember(behaviorindex(:,2),incldates),:);
end
indices.behaviorindex = behaviorindex; %[animal# date session# genotype]
