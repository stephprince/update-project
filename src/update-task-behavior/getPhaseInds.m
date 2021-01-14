function [phaseInds worldByPhase phaseType] = getPhaseInds(trialdata)
%SP 190924
%this function gets the start and end times of different task phases for
%each trial

windowLength = 20; %default is to look at first 20 samples to confirm phases separated correctly

%% get phase switch times
%690 and 160 are the north and south start points for the long track
phaseStartsTemp = [1; find(diff(trialdata.currentPhase))+1];
phaseEndsTemp = [find(diff(trialdata.currentPhase)); length(trialdata.currentPhase)];
phaseIndsTemp = [phaseStartsTemp phaseEndsTemp];

%% get new phase times for resampled track
phaseInds = []; worldByPhase = []; phaseType = [];
for phaseIdx = 1:size(phaseIndsTemp,1)
    %get window of time to look at for incorrectly separated phase times
    if (phaseIndsTemp(phaseIdx,2) - phaseIndsTemp(phaseIdx,1)) < windowLength
        windowLength = (phaseIndsTemp(phaseIdx,2) - phaseIndsTemp(phaseIdx,1)); %if less than 20 samples, just take the max
        if windowLength == 0; windowLength = 1; end; %corrects if phase is only one sample
    end
    
    %get new start times
    startPosRaw = trialdata.positionY(phaseIndsTemp(phaseIdx,1):phaseIndsTemp(phaseIdx,1)+windowLength-1); %look at first 20 samples
    if length(startPosRaw) == 1; startPosRaw = [startPosRaw; startPosRaw]; end %control so if only one data point the code below works
    startpoints = [690 160 0]; %north, south, delay/intertrial interval
    [minval, closestpos] = min(abs(startPosRaw - repmat(startpoints,length(startPosRaw),1))); %find which of first 20 samples is the closest to one of the startpoints
    [minval, northorsouth] = min(minval); %find which of the startpoints is closest: north is 1, south is 2, delay box is 3
    phaseStartNewRelative = closestpos(northorsouth); %phase start relative to original
    phaseStartsNew = phaseIndsTemp(phaseIdx,1) + phaseStartNewRelative - 1;
    
    %get new end times
    if phaseIndsTemp(phaseIdx,2) - windowLength < 1
        endPosRaw = 1;
    else
        endPosRaw = trialdata.positionY(phaseIndsTemp(phaseIdx,2)-windowLength:phaseIndsTemp(phaseIdx,2));
    end
    phaseInds = [phaseInds; phaseStartsNew phaseIndsTemp(phaseIdx,2)];
    
    %get current worlds for each phase
    %1 = main track, 2 = delay box, 3 = intertrial interval box
    worldByPhase = [worldByPhase; trialdata.currentWorld(phaseStartsNew)];
    
    %get phase types for each phase
    % 0 = end, 1 = delay, 2 = choice, 3 = reward, 4 = punish
    phaseType = [phaseType; trialdata.currentPhase(phaseStartsNew)];
end

end