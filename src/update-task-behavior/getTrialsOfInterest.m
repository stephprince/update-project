function trialTypeInds = getTrialsOfInterest(trialdata, params, paramIdx, plotCategoriesForWorld, howMuchToRound)

%% get different types of delay/update trials
trialsFromDelayTypeTemp1 = find(round(trialdata.trialDelayLocation) <= plotCategoriesForWorld(paramIdx,2));
trialsFromDelayTypeTemp2 = find(round(trialdata.trialDelayLocation) >= plotCategoriesForWorld(paramIdx,2)-howMuchToRound);
trialsFromUpdateType = find(round(trialdata.trialTypesUpdate) == plotCategoriesForWorld(paramIdx,3));
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
