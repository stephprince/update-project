function output = plotUpdateTaskCorrectPerformance(trackdata, indices, dirs, params)

params.trialBlockSize = 40;
savedfiguresdir = [dirs.behaviorfigdir 'percentCorrect\'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;
updateTypeKeySet = params.updateTypeMap.keys; updateTypeValueSet = params.updateTypeMap.values;
trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;

%get different parameters to calculate percent correct by like delay length
allDelayLocations = []; allDelayLocationsActual = [];
howMuchToRound = 0;
for anIdx = 1:numel(indices.animals)
    %get all the world 4 trials where the interesting stuff happens
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    trialRows = cellfun(@(x) find(x.trialWorld == 4), animaldata.trialTable,'UniformOutput',0);
    
    %get delay locations
    delayLocations = cell2mat(cellfun(@(x) round(x.trialDelayLocation/25)*25, animaldata.trialTable,'UniformOutput',0));
    delayLocationsActual = cell2mat(cellfun(@(x) x.trialDelayLocation, animaldata.trialTable,'UniformOutput',0));
    delayLocations(isnan(delayLocations)) = [];
    delayLocationsActual(isnan(delayLocationsActual)) = [];
    allDelayLocations = [allDelayLocations; delayLocations];
    allDelayLocationsActual = [allDelayLocationsActual; delayLocationsActual];
end
params.delayLocations = sort(unique(allDelayLocations), 'descend');
params.plotCategories = [2 nan 1; 3 nan 1; [repmat(4,numel(params.delayLocations),1) params.delayLocations ones(numel(params.delayLocations),1)]; 4 nan 2];

%% plot correct performance across different tracks for animals
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    figure('units','normalized','outerposition',[0 0 0.9 0.9]); hold on;
    for paramIdx = 1:size(params.plotCategories,1)
        %compile all trials for each track across days
        trialsFromWorldType = cellfun(@(x) find(x.trialWorld == params.plotCategories(paramIdx,1)), animaldata.trialTable,'UniformOutput',0);
        trialsFromDelayTypeTemp1 = cellfun(@(x) find(round(x.trialDelayLocation) <= params.plotCategories(paramIdx,2)), animaldata.trialTable,'UniformOutput',0);
        trialsFromDelayTypeTemp2 = cellfun(@(x) find(round(x.trialDelayLocation) >= params.plotCategories(paramIdx,2)-20), animaldata.trialTable,'UniformOutput',0);
        trialsFromUpdateType = cellfun(@(x) find(round(x.trialTypesUpdate) == params.plotCategories(paramIdx,3)), animaldata.trialTable,'UniformOutput',0);
        trialdata = [];
        for trialIdx = 1:numel(trialsFromWorldType)
            if ~isempty(trialsFromDelayTypeTemp1{trialIdx})
                trialsFromDelayType{trialIdx} = intersect(trialsFromDelayTypeTemp1{trialIdx},trialsFromDelayTypeTemp2{trialIdx});
            else
                trialsFromDelayType{trialIdx} = [];
            end
            if params.plotCategories(paramIdx,1) ~= 4
                trialRows = trialsFromWorldType{trialIdx};
            elseif params.plotCategories(paramIdx,1) == 4 && params.plotCategories(paramIdx,3) ~= 2
                trialRowsTemp = intersect(trialsFromWorldType{trialIdx},trialsFromDelayType{trialIdx});
                trialRows = intersect(trialRowsTemp,trialsFromUpdateType{trialIdx});
            else  %on update trials, want any delay length not just nan trials
                trialRows = intersect(trialsFromWorldType{trialIdx},trialsFromUpdateType{trialIdx});
            end
            trialdata = [trialdata; animaldata.trialTable{trialIdx,:}(trialRows,:)];
        end
        
        %loop through bins to calc percent correct
        numTrialsAll{anIdx}{paramIdx} = size(trialdata,1);
        rightTrialOutcomes = trialdata.trialOutcomes; leftTrialOutcomes = trialdata.trialOutcomes;
        rightTrialOutcomes(trialdata.trialTypesLeftRight == params.trialTypeMap('left')) = nan; %replace with nans so lengths of vectors are equal
        leftTrialOutcomes(trialdata.trialTypesLeftRight == params.trialTypeMap('right')) = nan;
        perCorrect{anIdx}{paramIdx} = movmean(trialdata.trialOutcomes,params.trialBlockSize,'omitnan');
        perCorrectRight{anIdx}{paramIdx} = movmean(rightTrialOutcomes,params.trialBlockSize,'omitnan');
        perCorrectLeft{anIdx}{paramIdx} = movmean(leftTrialOutcomes,params.trialBlockSize,'omitnan');
   
        %plot the result for individual animals
        if numTrialsAll{anIdx}{paramIdx}
            subplot(size(params.plotCategories,1),1,paramIdx); hold on;
            plot([1:numTrialsAll{anIdx}{paramIdx}],repmat(0.5,1,numTrialsAll{anIdx}{paramIdx}),'k--');
            plot([1:numTrialsAll{anIdx}{paramIdx}],perCorrectRight{anIdx}{paramIdx}, 'r');
            plot([1:numTrialsAll{anIdx}{paramIdx}],perCorrectLeft{anIdx}{paramIdx}, 'b');  
            plot([1:numTrialsAll{anIdx}{paramIdx}],perCorrect{anIdx}{paramIdx}, 'k','LineWidth',2);
            ylim([0 1.01]); ylabel('Percent Correct'); 
            trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
            updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
            title([trackName ' - delay loc: ' num2str(params.plotCategories(paramIdx,2)) ' - trial type:' updateType]);
        end
        
    end
    xlabel(['Trial Window (' num2str(params.trialBlockSize) ' trial moving average)']);
    sgtitle(['S' num2str(indices.animals(anIdx)) ' performance']);
    filename = [savedfiguresdir 'sessPerformanceMovingAvgAll_' trackName  '_S' num2str(indices.animals(anIdx))];
    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    
end

%% plot the results for all animals 
figure('units','normalized','outerposition',[0 0 1 1]); hold on;
cmap = cbrewer('qual','Set2',numel(indices.animals));
for anIdx = 1:numel(indices.animals)
    for paramIdx = 1:size(params.plotCategories,1)
        %plot the performance over time
        if numTrialsAll{anIdx}{paramIdx}
            subplot(size(params.plotCategories,1),3,(paramIdx)*3-2:(paramIdx)*3-1); hold on;
            plot([1:numTrialsAll{anIdx}{paramIdx}],repmat(0.5,1,numTrialsAll{anIdx}{paramIdx}),'k--');
            plot([1:numTrialsAll{anIdx}{paramIdx}],perCorrect{anIdx}{paramIdx}, 'Color',cmap(anIdx,:),'LineWidth',2);
            xlabel(['Trial Window (' num2str(params.trialBlockSize) ' trial moving average)']);
            ylim([0 1.01]); ylabel('Percent Correct'); 
            trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
            updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
            title([trackName ' - delay location: ' num2str(params.plotCategories(paramIdx,2)) 'ypos - ' updateType]);
        
            %plot the performance distribution for each type of track
            subplot(size(params.plotCategories,1),3,(paramIdx)*3); hold on;
            edges = 0:0.025:1;
            perCorrectHist = histcounts(perCorrect{anIdx}{paramIdx},edges);
            perCorrectHistNorm = perCorrectHist/nansum(perCorrectHist);
            plot([0.5 0.5],[0 1], 'k--'); ylim([0 0.33]);
            ylabel(['Proportion of trial windows']);
            if sum(~isnan(perCorrectHistNorm))
                h(anIdx) = histogram('BinCounts', perCorrectHistNorm, 'BinEdges', edges);
                h(anIdx).FaceAlpha = 0.2; xlim([0 1]);
                h(anIdx).FaceColor = cmap(anIdx,:);
            end
        end
    end
end
legendCell = cellstr(num2str(indices.animals', 'S%-d'));
legend(h,legendCell,'Location','NorthWest')
sgtitle(['All animals performance']);
filename = [savedfiguresdir 'sessPerformanceMovingAvgAll_AllAnimals'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

%% plot correct performance across different tracks for animals - all combined
anIdx = 'All';
figure('units','normalized','outerposition',[0 0 0.9 0.9]); hold on;
for paramIdx = 1:size(params.plotCategories,1)
    %compile all trials for each track across days
    trialsFromWorldType = cellfun(@(x) find(x.trialWorld == params.plotCategories(paramIdx,1)), trackdata.trialTable,'UniformOutput',0);
    trialsFromDelayTypeTemp1 = cellfun(@(x) find(round(x.trialDelayLocation) <= params.plotCategories(paramIdx,2)), trackdata.trialTable,'UniformOutput',0);
    trialsFromDelayTypeTemp2 = cellfun(@(x) find(round(x.trialDelayLocation) >= params.plotCategories(paramIdx,2)-20), trackdata.trialTable,'UniformOutput',0);
    trialsFromUpdateType = cellfun(@(x) find(round(x.trialTypesUpdate) == params.plotCategories(paramIdx,3)), trackdata.trialTable,'UniformOutput',0);
    trialdata = [];
    for trialIdx = 1:numel(trialsFromWorldType)
        if ~isempty(trialsFromDelayTypeTemp1{trialIdx})
            trialsFromDelayType{trialIdx} = intersect(trialsFromDelayTypeTemp1{trialIdx},trialsFromDelayTypeTemp2{trialIdx});
        else
            trialsFromDelayType{trialIdx} = [];
        end
        if params.plotCategories(paramIdx,1) ~= 4
            trialRows = trialsFromWorldType{trialIdx};
        elseif params.plotCategories(paramIdx,1) == 4 && params.plotCategories(paramIdx,3) ~= 2
            trialRowsTemp = intersect(trialsFromWorldType{trialIdx},trialsFromDelayType{trialIdx});
            trialRows = intersect(trialRowsTemp,trialsFromUpdateType{trialIdx});
        else  %on update trials, want any delay length not just nan trials
            trialRows = intersect(trialsFromWorldType{trialIdx},trialsFromUpdateType{trialIdx});
        end
        trialdata = [trialdata; trackdata.trialTable{trialIdx,:}(trialRows,:)];
    end
    
    %loop through bins to calc percent correct
    numTrialsAllCombined{paramIdx} = size(trialdata,1);
    rightTrialOutcomes = trialdata.trialOutcomes; leftTrialOutcomes = trialdata.trialOutcomes;
    rightTrialOutcomes(trialdata.trialTypesLeftRight == params.trialTypeMap('left')) = nan; %replace with nans so lengths of vectors are equal
    leftTrialOutcomes(trialdata.trialTypesLeftRight == params.trialTypeMap('right')) = nan;
    perCorrectAllCombined{paramIdx} = movmean(trialdata.trialOutcomes,params.trialBlockSize,'omitnan');
    perCorrectRightCombined{paramIdx} = movmean(rightTrialOutcomes,params.trialBlockSize,'omitnan');
    perCorrectLeftCombined{paramIdx} = movmean(leftTrialOutcomes,params.trialBlockSize,'omitnan');
    
    %plot the result for individual animals
    if numTrialsAllCombined{paramIdx}
        subplot(size(params.plotCategories,1),1,paramIdx); hold on;
        plot([1:numTrialsAllCombined{paramIdx}],repmat(0.5,1,numTrialsAllCombined{paramIdx}),'k--');
        plot([1:numTrialsAllCombined{paramIdx}],perCorrectRightCombined{paramIdx}, 'r');
        plot([1:numTrialsAllCombined{paramIdx}],perCorrectLeftCombined{paramIdx}, 'b');
        plot([1:numTrialsAllCombined{paramIdx}],perCorrectAllCombined{paramIdx}, 'k','LineWidth',2);
        ylim([0 1.01]); ylabel('Percent Correct');
        trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
        updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
        title([trackName ' - delay loc: ' num2str(params.plotCategories(paramIdx,2)) ' - trial type:' updateType]);
    end
    
end
xlabel(['Trial Window (' num2str(params.trialBlockSize) ' trial moving average)']);
sgtitle(['S' anIdx ' performance']);
filename = [savedfiguresdir 'sessPerformanceMovingAvgAll_' trackName  '_AllAnimalsCombined'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

%% plot performance for all animals combined
figure('units','normalized','outerposition',[0 0 1 1]); hold on;
cmap = cbrewer('qual','Set2',numel(indices.animals));
for paramIdx = 1:size(params.plotCategories,1)
    %plot the performance over time
    if numTrialsAllCombined{paramIdx}
        subplot(size(params.plotCategories,1),3,(paramIdx)*3-2:(paramIdx)*3-1); hold on;
        plot([1:numTrialsAllCombined{paramIdx}],repmat(0.5,1,numTrialsAllCombined{paramIdx}),'k--');
        plot([1:numTrialsAllCombined{paramIdx}],perCorrectAllCombined{paramIdx}, 'Color',cmap(1,:),'LineWidth',2);
        xlabel(['Trial Window (' num2str(params.trialBlockSize) ' trial moving average)']);
        ylim([0 1.01]); ylabel('Percent Correct');
        trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
        updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
        title([trackName ' - delay location: ' num2str(params.plotCategories(paramIdx,2)) 'ypos - ' updateType]);
        
        %plot the performance distribution for each type of track
        subplot(size(params.plotCategories,1),3,(paramIdx)*3); hold on;
        edges = 0:0.025:1;
        perCorrectHist = histcounts(perCorrectAllCombined{paramIdx},edges);
        perCorrectHistNorm = perCorrectHist/nansum(perCorrectHist);
        plot([0.5 0.5],[0 1], 'k--'); ylim([0 0.33]);
        ylabel(['Proportion of trial windows']);
        if sum(~isnan(perCorrectHistNorm))
            h(1) = histogram('BinCounts', perCorrectHistNorm, 'BinEdges', edges);
            h(1).FaceAlpha = 0.2; xlim([0 1]);
            h(1).FaceColor = cmap(1,:);
        end
    end
end
sgtitle(['All animals performance']);
filename = [savedfiguresdir 'sessPerformanceMovingAvgAll_AllAnimalsCombined'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

%% plot performance as a function of delay length (in terms of position)
figure(200); clf;
for anIdx = 1:numel(indices.animals)
    %concat data for relevant tracks
    delayLocations = []; performanceVals = []; allDelayLengths = [];
    for paramIdx = 2:size(params.plotCategories,1)-1 %skip the short and update maze so that comparing tracks of equal length
        %plot the performance over time
        if numTrialsAll{anIdx}{paramIdx}
            delayLength = params.plotCategories(paramIdx,2);
            if isnan(delayLength) || delayLength == 275
                delayLength = 250; %counts as a 0 delay length for ymaze long
            end
            
            delayLocations = [delayLocations; repmat(delayLength,numTrialsAll{anIdx}{paramIdx},1)];
            allDelayLengths = [allDelayLengths; delayLength];
            performanceVals = [performanceVals; perCorrect{anIdx}{paramIdx}];
        end
    end
    
    % add the data from the update task
    paramIdx = size(params.plotCategories,1);
    delayLocationsWithUpdate = [delayLocations; repmat(0,numTrialsAll{anIdx}{paramIdx},1)];
    performanceValsWithUpdate = [performanceVals; perCorrect{anIdx}{paramIdx}];

    %plot the data as a scatter plot
    figure(100); hold on;
    plot([0 300], [0.5 0.5], 'k--');
    h1 = scatter(delayLocationsWithUpdate,performanceValsWithUpdate,[],cmap(anIdx,:));
    h2 = lsline;
    xlabel(['Delay Location (smaller is longer), 0 = update trials, 250 = ymazeLong trials']); xlim([115 260])
    ylabel('Percent Correct'); ylim([0 1.01])    
    title('Performance as a function of delay location')

    %plot the data for the violin plot
    figure(200); hold on;
    ax1(anIdx) = subplot(numel(indices.animals),1,anIdx);   
    violinplot(performanceValsWithUpdate, delayLocationsWithUpdate, 'ViolinColor', cmap(anIdx,:))
    xlabel(['Trial types']);
    if numel(unique(allDelayLengths))+1 == 6
        xticklabels({'Update','Delay 4','Delay 3','Delay 2','Delay 1','Visual guided'})
    elseif numel(unique(allDelayLengths)) == 2
        xticklabels({'Delay 1','Visual guided'})
    end
    
    ylabel('Percent Correct'); ylim([0 1.01])    
    linkaxes(ax1, 'y'); linkaxes(ax1, 'x');
    title(['S' num2str(indices.animals(anIdx))])
    sgtitle('Performance across different delay lengths and trial types')
end

figure(100);
set(gcf,'units','normalized','outerposition',[0 0 0.9 0.9]);
for anIdx = 1:numel(indices.animals)
    h2(anIdx).Color = cmap(anIdx,:);
end
legendCell = cellstr(num2str(indices.animals', 'S%-d'));
legend(h2,legendCell,'Location','SouthEast')
filename = [savedfiguresdir 'sessPerformanceVsDelayLength_AllAnimals_scatter'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

figure(200)
set(gcf,'units','normalized','outerposition',[0 0 0.9 0.9]);
filename = [savedfiguresdir 'sessPerformanceVsDelayLength_AllAnimals_violinplot'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

%% plot performance as a function of delay length (in terms of position) - all animals combined
%concat data for relevant tracks
delayLocations = []; performanceVals = [];
for paramIdx = 2:size(params.plotCategories,1)-1 %skip the short maze and update so that comparing tracks of equal length
    %plot the performance over time
    if numTrialsAllCombined{paramIdx}
        delayLength = params.plotCategories(paramIdx,2);
        if isnan(delayLength) && params.plotCategories(paramIdx,3) ~= 2 % if it is a ymazeLong but not an update trial
            delayLength = 250; %counts as a 0 delay length for ymaze long
        end
        delayLocations = [delayLocations; repmat(delayLength,numTrialsAllCombined{paramIdx},1)];
        performanceVals = [performanceVals; perCorrectAllCombined{paramIdx}];
        
    end
end

paramIdx = size(params.plotCategories,1);
delayLocationsWithUpdate = [delayLocations; repmat(0,numTrialsAllCombined{paramIdx},1)];
performanceValsWithUpdate = [performanceVals; perCorrectAllCombined{paramIdx}];

%plot the data as a scatter plot
figure; hold on;
plot([0 300], [0.5 0.5], 'k--');
h1 = scatter(delayLocations,performanceVals,[],cmap(1,:));
h2 = lsline;
xlabel(['Delay Location (smaller is longer)']); xlim([115 260])
ylabel('Percent Correct'); ylim([0 1.01])
title('Performance as a function of delay location')
set(gcf,'units','normalized','outerposition',[0 0 0.9 0.9]);
h2(1).Color = cmap(1,:);
filename = [savedfiguresdir 'sessPerformanceVsDelayLength_AllAnimalsCombined_scatter'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

%plot the data for the violin plot
figure; hold on;
ax1(1) = subplot(1,1,1);
violinplot(performanceValsWithUpdate, delayLocationsWithUpdate, 'ViolinColor', cmap(1,:))
xlabel(['Delay Location (smaller is longer)']);
ylabel('Percent Correct'); ylim([0 1.01])
linkaxes(ax1, 'y'); linkaxes(ax1, 'x');
title(['SAll'])
sgtitle('Performance as a function of delay location')
set(gcf,'units','normalized','outerposition',[0 0 0.9 0.9]);
filename = [savedfiguresdir 'sessPerformanceVsDelayLength_AllAnimalsCombined_violinplot'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

%% plot heatmaps of performance and time spent on sessions for all tracks
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    
    %get performance for each type of track for each session
    perCorrectHeatMap{anIdx} = nan(size(params.plotCategories,1),size(animaldata,1));
    timeHeatMap{anIdx} = nan(size(params.plotCategories,1),size(animaldata,1));
    for sessIdx = 1:size(animaldata,1)
        sessdata = animaldata(sessIdx,:);
        for paramIdx = 1:size(params.plotCategories,1)
            trialsFromWorldType = cellfun(@(x) find(x.trialWorld == params.plotCategories(paramIdx,1)), sessdata.trialTable,'UniformOutput',0);
            trialsFromDelayTypeTemp1 = cellfun(@(x) find(round(x.trialDelayLocation) <= params.plotCategories(paramIdx,2)), sessdata.trialTable,'UniformOutput',0);
            trialsFromDelayTypeTemp2 = cellfun(@(x) find(round(x.trialDelayLocation) >= params.plotCategories(paramIdx,2)-20), sessdata.trialTable,'UniformOutput',0);
            trialsFromUpdateType = cellfun(@(x) find(round(x.trialTypesUpdate) == params.plotCategories(paramIdx,3)), sessdata.trialTable,'UniformOutput',0);
            trialdata = [];
            for trialIdx = 1:numel(trialsFromWorldType)
                if ~isempty(trialsFromDelayTypeTemp1{trialIdx})
                    trialsFromDelayType{trialIdx} = intersect(trialsFromDelayTypeTemp1{trialIdx},trialsFromDelayTypeTemp2{trialIdx});
                else
                    trialsFromDelayType{trialIdx} = [];
                end
                if params.plotCategories(paramIdx,1) ~= 4
                    trialRows = trialsFromWorldType{trialIdx};
                elseif params.plotCategories(paramIdx,1) == 4 && params.plotCategories(paramIdx,3) ~= 2
                    trialRowsTemp = intersect(trialsFromWorldType{trialIdx},trialsFromDelayType{trialIdx});
                    trialRows = intersect(trialRowsTemp,trialsFromUpdateType{trialIdx});
                else  %on update trials, want any delay length not just nan trials
                    trialRows = intersect(trialsFromWorldType{trialIdx},trialsFromUpdateType{trialIdx});
                end
                trialdata = [trialdata; sessdata.trialTable{trialIdx,:}(trialRows,:)];
            end

            %loop through bins to calc percent correct
            if ~isempty(trialdata) && params.plotCategories(paramIdx,2) ~= 275
                perCorrectHeatMap{anIdx}(paramIdx,sessIdx) = nanmean(trialdata.trialOutcomes);
                timeHeatMap{anIdx}(paramIdx,sessIdx) = nansum(trialdata.trialDur)/60; %convert from seconds to minutes
            end
        end
    end
    
    %remove empty rows)
    if ismember(indices.animals(anIdx),[27 29])
        ymazeLongWarmupCat = find(params.plotCategories(:,2) == 250);
        perCorrectHeatMap{anIdx}(ymazeLongWarmupCat,:) = [];
    end
    anySess = nansum(perCorrectHeatMap{anIdx},2);
    perCorrectHeatMap{anIdx}(anySess == 0,:) = [];
    
    %plot the heatmap for each animal
    figure(300); hold on;
    subplot(numel(indices.animals),1,anIdx)
    cmap = cbrewer('div','PRGn',100);
    imAlpha = ones(size(perCorrectHeatMap{anIdx}));
    imAlpha(isnan(perCorrectHeatMap{anIdx})) = 0;
    imagesc(perCorrectHeatMap{anIdx}, 'AlphaData', imAlpha)
    colorbar
    set(gca,'color',[0.1 0.1 0.1 0.1])
    colormap(cmap);
    ylabel('Training level');
    yticks([1:7])
    yticklabels({'Visual cue short','Visual cue long','Delay 1','Delay 2', 'Delay 3', 'Delay 4', 'Update'})
    title(['S' num2str(indices.animals(anIdx))])
    
    figure(400); hold on;
    subplot(numel(indices.animals),1,anIdx)
    cmap = cbrewer('seq','Reds',100);
    h(anIdx) = imagesc(timeHeatMap{anIdx}, [0 60])
    colormap(cmap);
    ylabel('Training level');
    title(['S' num2str(indices.animals(anIdx))])
end
figure(300)
xlabel('Session'); sgtitle('Performance over sessions');
filename = [savedfiguresdir 'sessPerformanceHeatmapAll'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

figure(400)
colorbar
xlabel('Session'); sgtitle('Time spent at training level over sessions');
filename = [savedfiguresdir 'timeSpentAtTrainingLevelHeatmapAll'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

