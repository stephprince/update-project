function behaviorDataDiamondByTrial = calcDiamondMetricsByTrial(sessdata, dirs, index, animalID)
%this function splits up diamond track behavioral data into individual
%trials

%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

%% get trial times
trialStarts = [1; find(diff(sessdata.numTrials)) + 1];  
trialEnds = [find(diff(sessdata.numTrials)); length(sessdata.numTrials)];

%% make new trial data structure
fnames = fieldnames(sessdata);
fields = fnames(4:end); %get rid of the trackname, params, session info fields that weren't matrices
for trialIdx = 1:size(trialStarts,1)
    for fieldIdx = 1:size(fields,1)
        trialInds = [trialStarts(trialIdx) trialEnds(trialIdx)];
        if ~isnan(sessdata.(fields{fieldIdx}))
            behaviorDataDiamondByTrial{trialIdx}.(fields{fieldIdx}) = sessdata.(fields{fieldIdx})(trialInds(1):trialInds(2));
        else
            behaviorDataDiamondByTrial{trialIdx}.(fields{fieldIdx}) = nan;
        end
    end
end

%% get switch times for incremental vectors (ex. rewards, licks, phases)
for trialIdx = 1:size(trialStarts,1)
    %get lick times
    behaviorDataDiamondByTrial{trialIdx}.lickInds = find(diff(behaviorDataDiamondByTrial{trialIdx}.numLicks))+1;

    %get reward times
    behaviorDataDiamondByTrial{trialIdx}.rewardInds = find(diff(behaviorDataDiamondByTrial{trialIdx}.numRewards))+1;
    
    %get phase switch times
    phaseStarts = [1; find(diff(behaviorDataDiamondByTrial{trialIdx}.currentPhase))+1];
    phaseEnds = [find(diff(behaviorDataDiamondByTrial{trialIdx}.currentPhase)); length(behaviorDataDiamondByTrial{trialIdx}.currentPhase)];
    behaviorDataDiamondByTrial{trialIdx}.phaseInds = [phaseStarts phaseEnds];
end

%% get trial duration
for trialIdx = 1:size(trialStarts,1)
    timeVect = behaviorDataDiamondByTrial{trialIdx}.time;
    behaviorDataDiamondByTrial{trialIdx}.trialdur = timeVect(end)-timeVect(1); %duration in sec
end

%% get north or south trial
for trialIdx = 1:size(trialStarts,1)
    startPos = behaviorDataDiamondByTrial{trialIdx}.positionY(2); %have to start at 2 bc north trials have start at 0,0
    if startPos > 600
        behaviorDataDiamondByTrial{trialIdx}.startPos = 'north';
    elseif startPos < 200
        behaviorDataDiamondByTrial{trialIdx}.startPos = 'south';
    end
end

%% get incorrect/correct trial %0 = incorrect, 1 = correct, -1 = failed                                                           
for trialIdx = 1:size(trialStarts,1)
    if strcmp(sessdata.params.trainingtype,'linear') %linear track does not have correct/incorrect zone and only has failed/correct trials
        behaviorDataDiamondByTrial{trialIdx}.outcome = 1; %correct trial
        behaviorDataDiamondByTrial{trialIdx}.outcome = -1; %failed trial 
    else
        %check if correct zone (1/2 depending on track)
        rightZoneVect = behaviorDataDiamondByTrial{trialIdx}.correctZone - behaviorDataDiamondByTrial{trialIdx}.currentZone;
        isCorrect = find(rightZoneVect == 0); %at one point the correct zone was equal to the current zone
        isIncorrect = find(behaviorDataDiamondByTrial{trialIdx}.currentZone ~= 0); %at one point the animal entered a reward zone
        if isempty(isIncorrect)
            behaviorDataDiamondByTrial{trialIdx}.outcome = -1; %failed trial occurred if the animal never reached a reward zone
        elseif ~isempty(isIncorrect) && isempty(isCorrect)
            behaviorDataDiamondByTrial{trialIdx}.outcome = 0; %incorrect trial if animal reached a reward zone but was not the correct one
        elseif ~isempty(isCorrect)
            behaviorDataDiamondByTrial{trialIdx}.outcome = 1;
        end
    end
end

