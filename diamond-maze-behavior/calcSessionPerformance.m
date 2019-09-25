function output = calcSessionPerformance(sessdata,trialdata)
%SP 190924

trialTypes = {'All','Left','Right','Same','Alt'};

%% get outcomes for all trials
temp = []; temp = cellfun(@(x) [temp; x.outcome], trialdata, 'UniformOutput', 0);
output.sessOutcomesAll = cell2mat(temp);
output.sessOutcomesLeft = output.sessOutcomesAll(sessdata.turnDirChoiceAll == 1); %left turns (right or left turns were correct during choice phase)
output.sessOutcomesRight = output.sessOutcomesAll(sessdata.turnDirChoiceAll == 2); %right turns
output.sessOutcomesSame = output.sessOutcomesAll(sessdata.sameTurnAll == 1); %same turn (encoding and choice phases had the same turn to be correct or not)
output.sessOutcomesAlt = output.sessOutcomesAll(sessdata.sameTurnAll == 0); %alternate turns

%% calculate number correct for each trial type
for typeIdx = 1:length(trialTypes)
    output.(['numCorrect' trialTypes{typeIdx}]) = length(find(output.(['sessOutcomes' trialTypes{typeIdx}]) == 1));
    output.(['numIncorrect' trialTypes{typeIdx}]) = length(find(output.(['sessOutcomes' trialTypes{typeIdx}]) == 0));
    output.(['numFailed' trialTypes{typeIdx}]) = length(find(output.(['sessOutcomes' trialTypes{typeIdx}]) == -1));
    output.(['numTrials' trialTypes{typeIdx}]) = length(output.(['sessOutcomes' trialTypes{typeIdx}]));
end

end