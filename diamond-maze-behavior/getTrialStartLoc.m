function [startLoc choiceLoc] = getTrialStartLoc(trialdata)
%SP 190924 this function gets the starting location for the encoding and
%choice phases of the task

%arbitrary designations, the starting north position is above this point
%for both short and long trials
northThreshold = 400;
southThreshold = 200;

%% initial encoding location
startPos = trialdata.positionY(trialdata.phaseStartsEnds(1,1));
if startPos > northThreshold 
    startLoc = 'north';
elseif startPos < southThreshold
    startLoc = 'south';
end

%% choice location if that phase occurred
%phases are encoding (1), delay (2), choice (3), intertrial interval (4)
if ismember(2,trialdata.phaseType) %if there's a choice phase
    choicePhase = find(trialdata.phaseType == 2);
    choicePos = trialdata.positionY(trialdata.phaseInds(choicePhase,1));
    if choicePos > northThreshold %arbitrary designations, the starting north position is above this point
        choiceLoc = 'north';
    elseif choicePos < southThreshold
        choiceLoc = 'south';
    end
else
    choiceLoc = 'nan';
end

end