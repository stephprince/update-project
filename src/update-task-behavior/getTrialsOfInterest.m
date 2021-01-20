function trialdata = getTrialsOfInterest(animaldata, params, paramIdx, howMuchToRound)

%% get trials from this world across all sessions
trialRows = cellfun(@(x) find(x.trialWorld == params.plotCategories(paramIdx,1)), animaldata.trialTable,'UniformOutput',0);
trialdata = [];
for trialIdx = 1:numel(trialRows)
    trialdata = [trialdata; animaldata.trialTable{trialIdx,:}(trialRows{trialIdx},:)];
end

%% get different types of delay/update trials
trialsFromDelayTypeTemp1 = find(round(trialdata.trialDelayLocation) <= params.plotCategories(paramIdx,2));
trialsFromDelayTypeTemp2 = find(round(trialdata.trialDelayLocation) >= params.plotCategories(paramIdx,2)-howMuchToRound);
trialsFromUpdateType = find(round(trialdata.trialTypesUpdate) == params.plotCategories(paramIdx,3));
trialsFromDelayType = [];
if ~isempty(trialsFromDelayTypeTemp1)
    trialsFromDelayType = intersect(trialsFromDelayTypeTemp1,trialsFromDelayTypeTemp2);
end

%subselect position data trials if necessary
if params.plotCategories(paramIdx,1) == 4 && params.plotCategories(paramIdx,3) ~= 2
    trialTypeInds = intersect(trialsFromDelayType,trialsFromUpdateType);
elseif params.plotCategories(paramIdx,1) == 4 && params.plotCategories(paramIdx,3) == 2  %on update trials, want any delay length not just nan trials
    trialTypeInds = trialsFromUpdateType;
else
    trialTypeInds = 1:size(trialdata,1); %all of them if there aren't any of these caveats
end

%% subselect new trial data structure for specific types of trials
trialdata = trialdata(trialTypeInds,:);
