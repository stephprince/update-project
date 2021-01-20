function [positionData, binsTable] = getPositionInfoForTrajectories(trialdata)

if ~isempty(trialdata)
    %extract trial periods of interest
    positionData = [];
    for trialIdx = 1:size(trialdata,1)
        virmenData = trialdata(trialIdx,:).resampledTrialData{1};
        patternToFind = ['[^5]5']; %look for any number of intertrial periods and the start of a new trial
        teleportToInterTrial = regexp(sprintf('%i', virmenData.currentWorld), patternToFind)'; %teleport happens at first intertrial phase

        if ~isempty(teleportToInterTrial)
            temp.trajectoryData = virmenData(1:teleportToInterTrial,2:6); %all the position related variables
            temp.minVals = varfun(@(x) min(x), temp.trajectoryData);
            temp.maxVals = varfun(@(x) max(x), temp.trajectoryData);
        else
            temp.trajectoryData = nan; %use the old min/max vals
        end
        positionData = [positionData; struct2table(temp,'AsArray',1)];
    end

    %get position bins
    minVals = varfun(@(x) min(x), positionData.minVals);
    maxVals = varfun(@(x) max(x), positionData.maxVals);
    newNames = regexprep(minVals.Properties.VariableNames, 'Fun_Fun_', '');
    for valIdx = 1:size(minVals,2)
        binsTemp.(newNames{valIdx}) = linspace(table2array(minVals(1,valIdx)),table2array(maxVals(1,valIdx)),50)';
    end
    binsTable = struct2table(binsTemp);

    %concatenate data across all the trials
    [histXTable, histYTable] = calcPositionHists(positionData, binsTable);
    positionData = [positionData, histXTable, histYTable];

    %add the outcomes and left/right trial type to the position data table
    positionData = [positionData, trialdata(:,{'trialOutcomes', 'trialTypesLeftRight', 'trialDelayLocation', 'trialUpdateLocation'})];
else
    positionData = [];
    binsTable = [];
end