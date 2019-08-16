function behaviorDataDiamond = getDiamondTrackBehaviorMetrics(dirs,indices,params,makenewfiles);
% this function gets behavior metrics for diamond maze tracks

%% load data mat files
for anIdx = 1:size(indices.behaviorindex,1)
    sessindex = indices.behaviorindex(anIdx,:);
    behaviorDataDiamondRaw{sessindex(1)}{sessindex(2)}{sessindex(3)} = loadRawDiamondVirmenFile(dirs, sessindex, indices.animalID);
end

%% get trial metrics
for anIdx = 1:size(indices.behaviorindex,1)
    sessindex = indices.behaviorindex(anIdx,:);
    sessdata = behaviorDataDiamondRaw{sessindex(1)}{sessindex(2)}{sessindex(3)};
    behaviorDataDiamondByTrial{sessindex(1)}{sessindex(2)}{sessindex(3)} = calcDiamondMetricsByTrial(sessdata, dirs, sessindex, indices.animalID);
end

%% get session metrics
for anIdx = 1:size(indices.behaviorindex,1)
    sessindex = indices.behaviorindex(anIdx,:);
    sessdata = behaviorDataDiamondRaw{sessindex(1)}{sessindex(2)}{sessindex(3)};
    trialdata = behaviorDataDiamondByTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
    behaviorDataDiamondBySession{sessindex(1)}{sessindex(2)}{sessindex(3)} = calcDiamondMetricsBySess(sessdata, trialdata, dirs, sessindex, indices.animalID);
end

%% make output structure 
behaviorDataDiamond.raw = behaviorDataDiamondRaw;
behaviorDataDiamond.byTrial = behaviorDataDiamondByTrial;
behaviorDataDiamond.bySession = behaviorDataDiamondBySession;