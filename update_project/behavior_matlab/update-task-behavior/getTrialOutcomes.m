function outcome = getTrialOutcomes(sessdata, trialdata)
% this function gets correct or incorrect trials based on if the reward
% zone was entered and it was correct

%% get failed, incorrect, and correct trials
%[0 1 2 3] is the pattern of phases for a correct trial
%[0 1 2 4] is the pattern of phases for an incorrect or failed trial
%[0 4] is the pattern of phases for a failed trial
%[0], [0 1], [0 1 2] are the pattern of phases for early termination trial (for example the end of the session)
if ~ismember(2,trialdata.phaseType) %if animal never reached choice phase then it was failed
    outcome = -1;
elseif ismember(3,trialdata.phaseType) %if animal entered post reward phase then it was correct
    outcome = 1;
elseif ismember(2,trialdata.phaseType) && ismember(4,trialdata.phaseType) %if the animal get to choice but goes to punishment, need to determine if fail or incorrect
    choicePhase = find(trialdata.phaseType == 2);
    choicePhaseInds = trialdata.phaseStartsEnds(choicePhase,1):trialdata.phaseStartsEnds(choicePhase,2);
    rightZoneVect = trialdata.correctZone(choicePhaseInds) - trialdata.currentZone(choicePhaseInds);
    isCorrect = find(rightZoneVect == 0); %at one point the correct zone was equal to the current zone
    gotReward = find(trialdata.currentZone ~= 0); %at one point the animal entered a reward zone
    if isempty(gotReward)
        outcome = -1; %failed trial occurred if the animal never reached a reward zone
    else
        outcome = 0; %incorrect trial if animal reached a reward zone but still entered punishment phase
    end
else %only other option is an early end of virmen ([0 1 2]) which counts as a fail
    outcome = -1;
end

%% separate linear track trials and continuous alt trials
if strcmp(sessdata.params.trainingtype,'linear') %linear track does not have correct/incorrect zone and only has failed trials
    outcome = -1; %failed trial
    return
end

%[0 3], [0 4] are the pattern of phases for continuous alt trials
if strcmp(sessdata.params.trainingtype,'continuousalt')
    if ismember(3,trialdata.phaseType) %if animal entered post reward phase then it was correct
        outcome = 1;
    elseif ismember(4,trialdata.phaseType) %if the animal get to choice but goes to punishment
        outcome = 0;
    end
end
end
