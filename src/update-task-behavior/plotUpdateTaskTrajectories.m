function plotUpdateTaskTrajectories(trackdata, indices, dirs, params)

savedfiguresdir = [dirs.behaviorfigdir 'trajectories\'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;

%% setup info
savedfiguresdir = [dirs.behaviorfigdir 'trajectories\'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;

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
    for worldIdx = 1:4
        %% compile all trials for each track across days for each world
        trialRows = cellfun(@(x) find(x.trialWorld == worldIdx), animaldata.trialTable,'UniformOutput',0);
        trialdata = [];
        for trialIdx = 1:numel(trialRows)
            trialdata = [trialdata; animaldata.trialTable{trialIdx,:}(trialRows{trialIdx},:)];
        end
        
        %get position histograms and bins for trajectories
        [positionData, binsTable] = getPositionInfoForTrajectories(trialdata);
        
        %% make the plots for each track/trial type
        plotCategoriesForWorld = params.plotCategories(params.plotCategories(:,1) == worldIdx,:); %find plots to get from this type of world
        for paramIdx = 1:size(plotCategoriesForWorld,1)
            trialTypeInds = getTrialsOfInterest(trialdata, params, paramIdx, plotCategoriesForWorld,howMuchToRound);

            plotHistByPosition(trialdata, positionData, binsTable, trialTypeInds, anIdx, worldIdx, paramIdx, indices, dirs, params)
            
%             if worldIdx == 4 %if we're in the update/delay part of the task
%                 plotPositionAroundUpdate(positionData, binsTable, worldIdx, dirs, params);
%             end
        end
    end
    test
end