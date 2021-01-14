function behaviorData = loadRawDiamondVirmenFile(dirs, index, animalID,makenewfiles)
%this function converts raw virmen data from the Diamond Maze track into a matlab structure 
%SP 190813

%inputs: 
%   dirs - directory structure with all the file path info
%   index - single animal index in format [animal# date session# genotype]
%outputs:
%   dataStruct - contains all the info from the virmen file organized into
%   a structure

%% load mat file
try
    sessionfile = load([dirs.virmendatadir, 'Data\', animalID, num2str(index(1)), ...
        '_', num2str(index(2)), '_', num2str(index(3)),'\dataWithLickometer.mat']);
catch ME
    if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
        fprintf(['File could not be found for ' animalID, num2str(index(1)), ...
        '_', num2str(index(2)), '_', num2str(index(3)) ' . Skipping analysis... \n']);
        dataStruct = nan;
        return
    end
end
    
%% make data structure
rawdata = sessionfile.saveDatCopy.data;
behaviorDataRaw = struct('trackname',sessionfile.saveDatCopy.trackname,'params',sessionfile.saveDatCopy.params,...
 'sessioninfo',sessionfile.saveDatCopy.sessioninfo,'time',(rawdata(:,1) - rawdata(1,1))*24*3600,...
 'positionX',rawdata(:,2),'positionY',rawdata(:,3),...
 'velocTrans',rawdata(:,4),'velocRot',rawdata(:,5),'numRewards',rawdata(:,6),...
 'numLicks',rawdata(:,7),'currentZone',rawdata(:,8),'correctZone',rawdata(:,9),...
 'currentWorld',rawdata(:,10),'currentPhase',rawdata(:,11),'numTrials',rawdata(:,12));
%currentZone/correctZone: 0 = nonRewardZone 1 = East, 2 = West, indicates which reward zone you're in 
%current World: 1 = diamond maze, 2 = delay box, 3 = intertrial box
%currentPhase: 0 = encoding, 1 = delay, 2 = choice, 3 = intertrial interval
%with reward, 4 = intertrial interval with punishment
%params - note at this time max trial duration was calculated as a difference in seconds so
%anything over 60 meant there was no max trial duration

if index(2) < 190812 %note, after this time period I changed the data structure so it has view angle as well
    behaviorDataRaw.viewAngle = nan;
else
    behaviorDataRaw.viewAngle  = rawdata(:,13);
end

%% save structure for single animal
savedatadir = [dirs.behaviorfigdir 'data\'];
if ~exist(savedatadir); mkdir(savedatadir); end;
filename = [savedatadir 'behaviorDataRaw_' animalID num2str(index(1)) '_' num2str(index(2)) '_' num2str(index(3))];
save(filename,'behaviorDataRaw');

end