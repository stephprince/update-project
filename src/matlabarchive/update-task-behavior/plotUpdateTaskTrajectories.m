function plotUpdateTaskTrajectories(trackdata, indices, dirs, params)

%% setup info
%get different parameters to calculate percent correct by like delay length
allDelayLocations = []; 
howMuchToRound = 20;
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    delayLocations = cell2mat(cellfun(@(x) round(x.trialDelayLocation/howMuchToRound)*howMuchToRound, animaldata.trialTable,'UniformOutput',0));
    delayLocations(isnan(delayLocations)) = [];
    allDelayLocations = [allDelayLocations; delayLocations];
end
params.delayLocations = sort(unique(allDelayLocations), 'descend');
params.plotCategories = [2 nan 1; 3 nan 1; [repmat(4,numel(params.delayLocations),1) params.delayLocations ones(numel(params.delayLocations),1)]; 4 nan 2];

%% create position histograms for all the animals
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    for paramIdx = 1:size(params.plotCategories,1)
        %% compile all trials for each track across days for each world
        %get trials to pick
        trialdata = getTrialsOfInterest(animaldata, params, paramIdx, howMuchToRound);
        
        %get position histograms and bins for trajectories
        [positionData, binsTable] = getPositionInfoForTrajectories(trialdata);
        
        %% make the plots for each track/trial type
        plotHistByPosition(trialdata, positionData, binsTable, anIdx, paramIdx, indices, dirs, params);
        
        if params.plotCategories(paramIdx,3) == 2 %if we're in the delay part of the task with update trials
            plotPositionAroundUpdate(trialdata, positionData, binsTable, anIdx, paramIdx, indices, dirs, params);
        end
    end
end
