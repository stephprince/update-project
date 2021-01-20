function [resampledData, trialTable] = calcUpdateTaskTrialData(dataTable, params, dirs, index, animalID, makenewfiles);
%this function splits up diamond track behavioral data into individual trials

%input:
%       dataTable - raw behavior table, output of loadRawDiamondVirmenFile
%       dirs - directory structure with all the file path info
%       index - single animal info
%       animalID - animal identifier, ie 'S','F'

filename1 = [dirs.savedatadir 'behaviorDataTableTrial_' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)];
filename2 = [dirs.savedatadir 'behaviorDataTableResampled_' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)];
disp(['Calculating trial metrics for ' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)]);

%catch for any cases that occurred before 10/5
if index.Date < 201005
    params.taskStatesMap = containers.Map({'startOfTrial','withinTrial','choiceMade','duringReward','endOfTrial','interTrial'},[1:6]);
end

if ~exist(filename1) || ~exist(filename2) %|| makenewfiles    
    %% resample data for whole session and then split into trials
    %resample the raw data
    rawData = dataTable.RawData{1}; %get the matrix from the virmen file
    resampledData = resampleUpdateTask(rawData, params);
    
    % get trial starts and ends
    patternToFind = [num2str(params.taskStatesMap('interTrial')) num2str(params.taskStatesMap('startOfTrial'))]; 
    trialEnds = [regexp(sprintf('%i', resampledData.taskState), patternToFind)'; numel(resampledData.taskState)]; %this end metric includes the intertrialinterval   
    trialStarts = [1; trialEnds(1:end-1)+1]; %do not include the last incomplete trial at the end
    trialNum = 1:numel(trialStarts);
    
    %add column for each trial start/end as we define it
    trialIntervals = arrayfun(@(s,f) s:f,trialStarts,trialEnds,'UniformOutput',false);
    trialGroup = []; counter = 1;
    for i = 1:numel(trialIntervals)
        trialGroup(trialIntervals{i}) = counter;
        counter = counter+1;
    end
    resampledData = [resampledData, table(trialGroup','VariableNames',{'trialGroup'})];
    
    %split up the resampled data into individual trials
    for trialIdx = 1:numel(trialStarts)
        resampledTrialData{trialIdx,:} = resampledData(trialStarts(trialIdx):trialEnds(trialIdx),:);
    end
    resampledTrialData = cell2table(resampledTrialData);
    
    %% get trial duration 
    trialDurs = resampledData.time(trialEnds)-resampledData.time(trialStarts); %might need to correct this time
    
    %% get trial outcomes
    patternToFind = [num2str(params.taskStatesMap('endOfTrial')) '[' num2str(params.taskStatesMap('interTrial')) ']+' num2str(params.taskStatesMap('startOfTrial'))]; 
    trialEndsBeforeIntertrial = [regexp(sprintf('%i', resampledData.taskState), patternToFind)'; numel(resampledData.taskState)];
    trialOutcomes = resampledData.choice(trialEndsBeforeIntertrial); %what the outcome was at the end of the trial
    
    %% get trial types
    trialTypesLeftRight = resampledData.trialType(trialStarts); %what the type is (changes at the end of the trial)
    trialTypesUpdate = resampledData.trialTypeUpdate(trialStarts); %what the type is in terms of update cue or not
    %going to want to add something for delay when I come across that data
    trialWorld = resampledData.currentWorld(trialEndsBeforeIntertrial);
    
    %% get trial delay location and duration(if any)
    if index.Date >= 201005
        whenDelay = find((diff(resampledData.delayUpdateOccurred) == 1)) + 1; %this value is more accurate than task state bc closer to when cues actually change (1 ind closer)
    else
        whenDelay = find((diff(resampledData.updateOccurred) == 1)) + 1;
    end
    trialDelayLocation = nan(size(trialOutcomes,1),1);
    whichTrialsWithUpdate = resampledData.trialGroup(whenDelay);
    trialDelayLocation(whichTrialsWithUpdate) = resampledData.yPos(whenDelay);
    
    %get trial delay duration in time
    trialDelayUpdateTime = nan(size(trialOutcomes,1),1);
    trialDelayUpdateTime(whichTrialsWithUpdate) = resampledData.time(whenDelay);
    
    trialChoiceTime = nan(size(trialOutcomes,1),1);
    whenChoiceMade = find(resampledData.taskState == params.taskStatesMap('choiceMade')); %find when animal makes choice
    whichTrialsWithChoice = resampledData.trialGroup(whenChoiceMade);
    trialChoiceTime(whichTrialsWithChoice) = resampledData.time(whenChoiceMade);
    
    trialDelayDuration = trialChoiceTime - trialDelayUpdateTime;
    
    %% get trial update location (if any)
    if index.Date >= 201005
    	whenUpdate = find((diff(resampledData.updateOccurred) == 1)) + 1; 
    else
        whenUpdate = [];
    end
    trialUpdateLocation = nan(size(trialOutcomes,1),1);
    whichTrialsWithUpdate = resampledData.trialGroup(whenUpdate);
    trialUpdateLocation(whichTrialsWithUpdate) = resampledData.yPos(whenUpdate);
    
    %% concatenate data into table
    trialArray = [trialNum' trialStarts, trialEnds, trialDurs, trialOutcomes, trialTypesLeftRight, trialDelayLocation, trialUpdateLocation, trialDelayDuration, trialTypesUpdate, trialWorld];
    trialArrayHeaders = {'trialNum','trialStart','trialEnd', 'trialDur', 'trialOutcomes', 'trialTypesLeftRight', 'trialDelayLocation','trialUpdateLocation','trialDelayDuration', 'trialTypesUpdate', 'trialWorld'};
    trialTableTemp = array2table(trialArray,'VariableNames',trialArrayHeaders);
    trialTable = [trialTableTemp, resampledTrialData];
    
    %% save file
    save(filename1, 'trialTable', '-v7.3');
    save(filename2, 'resampledData', '-v7.3');

else
    load(filename1);
    load(filename2);
end
