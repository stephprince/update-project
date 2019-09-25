function [output eventsOutput] = resampleDiamondMazeTrials(trialdata,params);
% SP 190924

numSamples = length(trialdata.time); %initial sample number

%% resample velocity, time, etc.
output.timeNorm = resample(trialdata.time, params.newSampSize, numSamples);
output.velocTransNorm = resample(trialdata.velocTrans, params.newSampSize, numSamples);
output.velocRotNorm = resample(trialdata.velocRot, params.newSampSize, numSamples);
if ~isnan(trialdata.viewAngle)
    output.viewAngle = resample(trialdata.viewAngle, params.newSampSize, numSamples);
end

%% adjust phase/licks/reward times for resampled data
%(get fraction of initial trial time that event occurred and multiply by new sample size)
eventsOutput.phaseInds = trialdata.phaseInds*params.newSampSize/numSamples;
eventsOutput.rewardInds = trialdata.rewardInds*params.newSampSize/numSamples;
eventsOutput.lickInds = trialdata.lickInds*params.newSampSize/numSamples;

%% adjust position
%caveat here, need to separate phases well bc otherwise you will smooth over the really big jumps in position
%so for the adjusted position vectors, each phase will be numSamples x 1
possiblePhaseTypes = [0 1 2 3 4]; %encoding (0), delay (1), choice (2), reward (3) punish (4) usually
incompletePhases = possiblePhaseTypes(~ismember(possiblePhaseTypes, trialdata.phaseType));
completePhases = trialdata.phaseType;
phaseInds = trialdata.phaseInds;
eventsOutput.completePhases = completePhases+1;

%fill in incomplete phases with nans
for phaseIdx = 1:length(incompletePhases)
    phaseType = incompletePhases(phaseIdx);
    eventsOutput.phase(phaseType+1).posXNorm = nan(1, params.newSampSize); %since encoding is 0, have to shift phases by 1
    eventsOutput.phase(phaseType+1).posYNorm = nan(1, params.newSampSize); %so now enc = 1, del = 2, choice = 3, reward = 4, punish = 5;
end

%fill in complete phases with data (have to pad on either ends to get rid of weird sampling issues)
for phaseIdx = 1:length(completePhases)
    phaseType = completePhases(phaseIdx);
    posX = trialdata.positionX(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
    posY = trialdata.positionY(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
    
    %get ratio of downsampling and multiply by padFactor to add pad on either side
    padFactor = 100;
    padLength = ceil(length(posX)/params.newSampSize)*padFactor; 
    posXpad = [repmat(posX(1),padLength,1); posX; repmat(posX(end),padLength,1)];
    posYpad = [repmat(posY(1),padLength,1); posY; repmat(posY(end),padLength,1)];
    numSamplesPhase = length(posXpad);
    
    posXtemp = resample(posXpad, params.newSampSize+padFactor*2, numSamplesPhase);
    posYtemp = resample(posYpad, params.newSampSize+padFactor*2, numSamplesPhase);
    if length(posY) == 1
        posYtemp = repmat(posY,params.newSampSize+padFactor*2,1);
        posXtemp = repmat(posX,params.newSampSize+padFactor*2,1);
    end
    eventsOutput.phase(phaseType+1).posXNorm = posXtemp(padFactor:end-padFactor-1);
    eventsOutput.phase(phaseType+1).posYNorm = posYtemp(padFactor:end-padFactor-1);
end

end