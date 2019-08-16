function  behaviorDataDiamondBySess = calcDiamondMetricsBySess(sessdata, trialdata, dirs, index, animalID);
%this function calculates session metrics using the raw and trial data
%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       trialdata - behavior broken up into trials, output of calcDiamondMetricsByTrial
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

newSampSize = 10000;

%% resample vectors so can combine across trials
for trialIdx = 1:size(trialdata,2)
    numSamples = length(trialdata{trialIdx}.time); %initial sample number
    
    %resample velocity, time, etc.
    behaviorDataDiamondBySess.timeNorm(trialIdx,:) = resample(trialdata{trialIdx}.time, newSampSize, numSamples);
    behaviorDataDiamondBySess.velocTransNorm(trialIdx,:) = resample(trialdata{trialIdx}.velocTrans, newSampSize, numSamples);
    behaviorDataDiamondBySess.velocRotNorm(trialIdx,:) = resample(trialdata{trialIdx}.velocRot, newSampSize, numSamples);
    if ~isnan(trialdata{trialIdx}.viewAngle)
        behaviorDataDiamondBySess.viewAngle(trialIdx,:) = resample(trialdata{trialIdx}.viewAngle, newSampSize, numSamples);
    end
    
    %adjust phase/licks/reward times for resampled data
    %(get fraction of initial trial time that event occurred and multiply by new sample size)
    behaviorDataDiamondBySess.phaseInds{trialIdx} = trialdata{trialIdx}.phaseInds*newSampSize/numSamples;
    behaviorDataDiamondBySess.rewardInds{trialIdx} = trialdata{trialIdx}.rewardInds*newSampSize/numSamples;
    behaviorDataDiamondBySess.lickInds{trialIdx} = trialdata{trialIdx}.lickInds*newSampSize/numSamples;

    %resample position (need to separate north and south trials since different starting positions)
    %position is also special since it jumps, have to resample by phases so
    %that big jumps don't get averaged over
    counter1 = 1; counter2 = 1;
    for phaseIdx = 1:size(trialdata{trialIdx}.phaseInds,1)
        phaseInds = trialdata{trialIdx}.phaseInds;
        numSamplesPhase = length(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
        if strcmp(trialdata{trialIdx}.startPos,'north')
            behaviorDataDiamondBySess.phase(phaseIdx).posXNormNorth(counter1,:) = resample(trialdata{trialIdx}.positionX(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2)), newSampSize, numSamplesPhase);
            behaviorDataDiamondBySess.phase(phaseIdx).posYNormNorth(counter1,:) = resample(trialdata{trialIdx}.positionY(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2)), newSampSize, numSamplesPhase);
            counter1 = counter1 + 1;
        elseif strcmp(trialdata{trialIdx}.startPos,'south')
            behaviorDataDiamondBySess.phase(phaseIdx).posXNormSouth(counter2,:) = resample(trialdata{trialIdx}.positionX(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2)), newSampSize, numSamplesPhase);
            behaviorDataDiamondBySess.phase(phaseIdx).posYNormSouth(counter2,:) = resample(trialdata{trialIdx}.positionY(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2)), newSampSize, numSamplesPhase);
            counter2 = counter2 + 1;
        end
        
    end
end

%% get percent correct
temp = []; temp = cellfun(@(x) [temp; x.outcome], trialdata, 'UniformOutput', 0);
behaviorDataDiamondBySess.sessOutcomes = cell2mat(temp);
behaviorDataDiamondBySess.numCorrect = length(find(behaviorDataDiamondBySess.sessOutcomes == 1));
behaviorDataDiamondBySess.numFailed = length(find(behaviorDataDiamondBySess.sessOutcomes == -1));
behaviorDataDiamondBySess.numIncorrect = length(find(behaviorDataDiamondBySess.sessOutcomes == 0));
behaviorDataDiamondBySess.numTrials = size(trialdata,2);
behaviorDataDiamondBySess.trainingtype = sessdata.params.trainingtype;
 
%% get average trial duration
trialdur = [];
for trialIdx = 1:size(trialdata,2)
    trialdur = [trialdur; trialdata{trialIdx}.trialdur];
end
behaviorDataDiamondBySess.trialdur = trialdur;
behaviorDataDiamondBySess.trialdurAvg = mean(trialdur);
behaviorDataDiamondBySess.trialdurSem = std(trialdur)/sqrt(length(trialdur));
