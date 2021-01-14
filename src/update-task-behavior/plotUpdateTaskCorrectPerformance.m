function output = plotUpdateTaskCorrectPerformance(trackdata, indices, dirs, params)

params.trialBlockSize = 40;
savedfiguresdir = [dirs.behaviorfigdir 'percentCorrect\'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;
updateTypeKeySet = params.updateTypeMap.keys; updateTypeValueSet = params.updateTypeMap.values;
trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;

%get different parameters to calculate percent correct by like delay length
allDelayLocations = []; 
howMuchToRound = 0;
for anIdx = 1:numel(indices.animals)
    %get all the world 4 trials where the interesting stuff happens
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    trialRows = cellfun(@(x) find(x.trialWorld == 4), animaldata.trialTable,'UniformOutput',0);
    
    %get delay locations
    delayLocations = cell2mat(cellfun(@(x) round(x.trialDelayLocation/20)*20, animaldata.trialTable,'UniformOutput',0));
    delayLocations(isnan(delayLocations)) = [];
    allDelayLocations = [allDelayLocations; delayLocations];
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

%% plot performance as a function of delay length (in terms of position)
figure(200); clf;
for anIdx = 1:numel(indices.animals)
    %concat data for relevant tracks
    delayLocations = []; performanceVals = [];
    for paramIdx = 2:size(params.plotCategories,1)-1 %skip the short and update maze so that comparing tracks of equal length
        %plot the performance over time
        if numTrialsAll{anIdx}{paramIdx}
            delayLength = params.plotCategories(paramIdx,2);
            if isnan(delayLength)
                delayLength = 250; %counts as a 0 delay length for ymaze long
            end
            delayLocations = [delayLocations; repmat(delayLength,numTrialsAll{anIdx}{paramIdx},1)];
            performanceVals = [performanceVals; perCorrect{anIdx}{paramIdx}];
   
        end
    end

    %plot the data as a scatter plot
    figure(100); hold on;
    plot([0 300], [0.5 0.5], 'k--');
    h1 = scatter(delayLocations,performanceVals,[],cmap(anIdx,:));
    h2 = lsline;
    xlabel(['Delay Location (smaller is longer)']); xlim([115 260])
    ylabel('Percent Correct'); ylim([0 1.01])    
    title('Performance as a function of delay location')

    %plot the data for the violin plot
    figure(200); hold on;
    ax1(anIdx) = subplot(numel(indices.animals),1,anIdx);   
    violinplot(performanceVals, delayLocations, 'ViolinColor', cmap(anIdx,:))
    xlabel(['Delay Location (smaller is longer)']);
    ylabel('Percent Correct'); ylim([0 1.01])    
    linkaxes(ax1, 'y'); linkaxes(ax1, 'x');
    title(['S' num2str(indices.animals(anIdx))])
    sgtitle('Performance as a function of delay location')
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
            if ~isempty(trialdata)
                perCorrectHeatMap{anIdx}(paramIdx,sessIdx) = nanmean(trialdata.trialOutcomes);
                timeHeatMap{anIdx}(paramIdx,sessIdx) = nansum(trialdata.trialDur)/60; %convert from seconds to minutes
            end
        end
    end
    
    %plot the heatmap for each animal
    figure(300); hold on;
    subplot(numel(indices.animals),1,anIdx)
    cmap = cbrewer('div','PRGn',100);
    imagesc(perCorrectHeatMap{anIdx},[0 1])
    colormap(cmap);
    ylabel('Training level');
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
colorbar
xlabel('Session'); sgtitle('Performance over sessions');
filename = [savedfiguresdir 'sessPerformanceHeatmapAll'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

figure(400)
colorbar
xlabel('Session'); sgtitle('Time spent at training level over sessions');
filename = [savedfiguresdir 'timeSpentAtTrainingLevelHeatmapAll'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

