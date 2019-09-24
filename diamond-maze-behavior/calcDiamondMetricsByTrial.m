function behaviorDataDiamondByTrial = calcDiamondMetricsByTrial(sessdata, params, dirs, index, animalID, makenewfiles)
%this function splits up diamond track behavioral data into individual
%trials

%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'


filename = [dirs.savedatadir 'behaviorDataDiamondTrial_' animalID num2str(index(1)) '_' num2str(index(2)) '_' num2str(index(3))];
disp(['Calculating trial metrics for ' animalID num2str(index(1)) ' ' num2str(index(2)) ' ' num2str(index(3))]);

if ~exist(filename) || makenewfiles
    %% get trial times
    trialStarts = [1; find(diff(ismember(sessdata.currentPhase,[3 4])) == -1) + 1]; %looks for when reward or punishment phase happened and finds the end
    trialEnds = [find(diff(ismember(sessdata.currentPhase,[3 4])) == -1); length(sessdata.currentPhase)];
    
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
        
        %get phase times
        [phaseInds worldByPhase phaseType] = getPhaseInds(behaviorDataDiamondByTrial{trialIdx});
        behaviorDataDiamondByTrial{trialIdx}.phaseInds = phaseInds;
        behaviorDataDiamondByTrial{trialIdx}.worldByPhase = worldByPhase;
        behaviorDataDiamondByTrial{trialIdx}.phaseType = phaseType
    end
    
    %% get trial duration
    for trialIdx = 1:size(trialStarts,1)
        timeVect = behaviorDataDiamondByTrial{trialIdx}.time;
        behaviorDataDiamondByTrial{trialIdx}.trialdur = timeVect(end)-timeVect(1); %duration in sec
    end
    
    %% get incorrect/correct trial %0 = incorrect, 1 = correct, -1 = failed
    for trialIdx = 1:size(trialStarts,1)
        behaviorDataDiamondByTrial{trialIdx}.outcome = getTrialOutcomes(sessdata, behaviorDataDiamondByTrial{trialIdx});
    end
    
    %% get north or south trial
    for trialIdx = 1:size(trialStarts,1)
        [startLoc choiceLoc] = getTrialStartLoc(behaviorDataDiamondByTrial{trialIdx});
        behaviorDataDiamondByTrial{trialIdx}.trialStartLoc = startLoc;
        behaviorDataDiamondByTrial{trialIdx}.trialChoiceLoc = choiceLoc;
    end
    
    %% get right or left trial
    for trialIdx = 1:size(trialStarts,1)
        %note the first correct zone value will be correct side for encoding only (would switch for choice)
        if behaviorDataDiamondByTrial{trialIdx}.correctZone(1) == 2
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectEncLoc = 'east';
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectChoiceLoc = 'west';
        elseif behaviorDataDiamondByTrial{trialIdx}.correctZone(1) == 1
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectEncLoc = 'west';
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectChoiceLoc = 'east';
        end
    end
    
    %% save file
    save(filename, 'behaviorDataDiamondByTrial');
    
else
    load(filename);
end
