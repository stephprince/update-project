function  sessTable = calcUpdateTaskSessData(resampledData, trialData, params, dirs, index, animalID, makenewfiles);
%this function calculates session metrics using the raw and trial data
%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       trialdata - behavior broken up into trials, output of calcDiamondMetricsByTrial
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

filename = [dirs.savedatadir 'behaviorDataTableSess_' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)];
disp(['Calculating session metrics for ' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)]);

if ~exist(filename) || makenewfiles

    %% get percent correct
    sessPerformance = calcSessionPerformance(resampledData,trialData, index);
    fnamesPerformance = fieldnames(sessPerformance);
    for fieldIdx = 1:length(fnamesPerformance)
        behaviorDataDiamondBySess.(fnamesPerformance{fieldIdx}) = sessPerformance.(fnamesPerformance{fieldIdx});
    end

    %% get training session type
    behaviorDataDiamondBySess.trainingtype = sessdata.params.trainingtype;

    %% save filename
    save(filename, 'sessTable');

else
    load(filename);
end
