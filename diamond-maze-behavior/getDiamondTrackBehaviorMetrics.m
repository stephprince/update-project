function behaviorDataDiamond = getDiamondTrackBehaviorMetrics(dirs,indices,params,makenewfiles);
% this function gets behavior metrics for diamond maze tracks

filename = [dirs.savedatadir 'behaviorDataDiamondAll.mat'];

if ~exist(filename) || makenewfiles
    %% load data mat files (this converts the virmen files into a mat file)
    for anIdx = 1:size(indices.behaviorindex,1)
        sessindex = indices.behaviorindex(anIdx,:);
        behaviorDataDiamondRaw{sessindex(1)}{sessindex(2)}{sessindex(3)} = loadRawDiamondVirmenFile(dirs, sessindex, indices.animalID, makenewfiles);
    end
    
    %% get trial metrics (this splits up the data into individual trials)
    for anIdx = 1:size(indices.behaviorindex,1)
        sessindex = indices.behaviorindex(anIdx,:);
        sessdata = behaviorDataDiamondRaw{sessindex(1)}{sessindex(2)}{sessindex(3)};
        behaviorDataDiamondByTrial{sessindex(1)}{sessindex(2)}{sessindex(3)} = calcDiamondMetricsByTrial(sessdata, params, dirs, sessindex, indices.animalID, makenewfiles);
    end
    
    %% get session metrics
    for anIdx = 1:size(indices.behaviorindex,1)
        sessindex = indices.behaviorindex(anIdx,:);
        sessdata = behaviorDataDiamondRaw{sessindex(1)}{sessindex(2)}{sessindex(3)};
        trialdata = behaviorDataDiamondByTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
        behaviorDataDiamondBySession{sessindex(1)}{sessindex(2)}{sessindex(3)} = calcDiamondMetricsBySess(sessdata, trialdata, params, dirs, sessindex, indices.animalID, makenewfiles);
    end
    
    %% make output structure
    %behaviorDataDiamond.raw = behaviorDataDiamondRaw; %takes long to save all the raw data
    behaviorDataDiamond.byTrial = behaviorDataDiamondByTrial;
    behaviorDataDiamond.bySession = behaviorDataDiamondBySession;
    save(filename, 'behaviorDataDiamond', '-v7.3' );
    
else
    load(filename);
end