function behaviorDataTable = getUpdateTaskBehaviorMetrics(dirs, indices, params, ephys, makenewfiles)
% this function gets behavior metrics for diamond maze tracks

filename = [dirs.savedatadir 'updateTaskBehaviorData.mat'];

if ~exist(filename) || makenewfiles
    
    %% loop through each session to get the behavioral metrics from the raw virmen file
    for sessIdx = 1:numel(indices.behaviorindex.Animal)    
        tic
        % load data mat files (this converts the virmen files into a mat file)
        sessInfo = indices.behaviorindex(sessIdx,:);
        rawDataTable = loadRawUpdateTaskVirmenFile(dirs, sessInfo, indices.animalID, ephys, makenewfiles);
        
        % get trial metrics (this splits up the data into individual trials to calculate metrics of interest)
        [~, behaviorDataTemp.trialTable] = calcUpdateTaskTrialData(rawDataTable, params, dirs, sessInfo, indices.animalID, makenewfiles);
        
        %combine all session info into row on behavior table
        behaviorDataTempTable = struct2table(behaviorDataTemp,'AsArray',1);
        behaviorDataTable(sessIdx,:) = [sessInfo(:,1:4), behaviorDataTempTable]; %don't include the raw data, protocol log, or params because too big

        clear rawDataTable behaviorDataTempTable behaviorDataTemp
        toc
    end
            
    %% make output structure
    save(filename, 'behaviorDataTable', '-v7.3' );

else
    load(filename);
end
