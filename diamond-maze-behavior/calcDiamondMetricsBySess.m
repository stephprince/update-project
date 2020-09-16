function  behaviorDataDiamondBySess = calcDiamondMetricsBySess(sessdata, trialdata, params, dirs, index, animalID, makenewfiles);
%this function calculates session metrics using the raw and trial data
%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       trialdata - behavior broken up into trials, output of calcDiamondMetricsByTrial
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

filename = [dirs.savedatadir 'behaviorDataDiamondSess_' animalID num2str(index(1)) '_' num2str(index(2)) '_' num2str(index(3))];
disp(['Calculating session metrics for ' animalID num2str(index(1)) ' ' num2str(index(2)) ' ' num2str(index(3))]);

if ~exist(filename) || makenewfiles
    %% concatenate trial position, velocity, and viewAngle data
    concatTrialData = concatTrialBehavior(trialdata,sessdata.params.trainingtype);
    fnamesBehavior = fieldnames(concatTrialData);
    for fieldIdx = 1:length(fnamesBehavior)
        behaviorDataDiamondBySess.(fnamesBehavior{fieldIdx}) = concatTrialData.(fnamesBehavior{fieldIdx});
    end

    %% concatenate trial outputs
    concatTrialResults = concatTrialOutcomes(trialdata,sessdata.params.trainingtype);
    fnamesOutcomes = fieldnames(concatTrialResults);
    for fieldIdx = 1:length(fnamesOutcomes)
        behaviorDataDiamondBySess.(fnamesOutcomes{fieldIdx}) = concatTrialResults.(fnamesOutcomes{fieldIdx});
    end

    %% get percent correct
    sessPerformance = calcSessionPerformance(behaviorDataDiamondBySess,trialdata,sessdata.params.trainingtype, index);
    fnamesPerformance = fieldnames(sessPerformance);
    for fieldIdx = 1:length(fnamesPerformance)
        behaviorDataDiamondBySess.(fnamesPerformance{fieldIdx}) = sessPerformance.(fnamesPerformance{fieldIdx});
    end

    %% get training session type
    behaviorDataDiamondBySess.trainingtype = sessdata.params.trainingtype;

    %% save filename
    save(filename, 'behaviorDataDiamondBySess');

else
    load(filename);
end
