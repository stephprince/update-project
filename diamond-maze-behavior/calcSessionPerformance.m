function output = calcSessionPerformance(sessdata,trialdata)
%SP 190924

trialTypes = {'All','Left','Right','Same','Alt'};

%% get outcomes for all trials
temp = []; temp = cellfun(@(x) [temp; x.outcome], trialdata, 'UniformOutput', 0);
output.sessOutcomesAll = cell2mat(temp)';
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

%% calculate percent correct for each trial type
for typeIdx = 1:length(trialTypes)
    output.(['perCorrect' trialTypes{typeIdx}]) =  output.(['numCorrect' trialTypes{typeIdx}])/output.(['numTrials' trialTypes{typeIdx}]);
    output.(['perIncorrect' trialTypes{typeIdx}]) = output.(['numIncorrect' trialTypes{typeIdx}])/output.(['numTrials' trialTypes{typeIdx}]);
    output.(['perFailed' trialTypes{typeIdx}]) = output.(['numFailed' trialTypes{typeIdx}])/output.(['numTrials' trialTypes{typeIdx}]);
end

%% get trial duration for each trial type
for typeIdx = 1:length(trialTypes)
    output.(['durCorrect' trialTypes{typeIdx}]) = sessdata.durAll(output.(['sessOutcomes' trialTypes{typeIdx}]) == 1);
    output.(['durIncorrect' trialTypes{typeIdx}]) = sessdata.durAll(output.(['sessOutcomes' trialTypes{typeIdx}]) == 0);
    output.(['durFailed' trialTypes{typeIdx}]) = sessdata.durAll(output.(['sessOutcomes' trialTypes{typeIdx}]) == -1);
end

%% calculate time since last correct trial (ie btwn correct trials on continuous alternation)
for typeIdx = 1:length(trialTypes)
    trialsSinceCorrect = [];
    correctTrials = find(output.(['sessOutcomes' trialTypes{typeIdx}]) == 1);
    incorrectIntervals = [correctTrials, [correctTrials(2:end)-1; length(output.(['sessOutcomes' trialTypes{typeIdx}]))]];
    if ~isempty(correctTrials)
        for i = 1:size(incorrectIntervals,1)
            startToEnd = incorrectIntervals(i,:);
            countsSinceLast = [startToEnd(1):startToEnd(2)]-startToEnd(1);
            trialsSinceLast = max(countsSinceLast);
            trialsSinceCorrect = [trialsSinceCorrect, countsSinceLast];
        end
    else
        trialsSinceCorrect = 0;
        trialsSinceLast = 0;
    end
    output.(['trialsSinceCorrect' trialTypes{typeIdx}]) =  trialsSinceCorrect';
    output.(['trialsSinceLast' trialTypes{typeIdx}]) =  trialsSinceLast';
end

end