function plotHistByPosition(trialdata, positionData, binsTable, trialTypeInds, anIdx, worldIdx, paramIdx, indices, dirs, params);

%initialize variables
updateTypeKeySet = params.updateTypeMap.keys; updateTypeValueSet = params.updateTypeMap.values;
trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;
trialOutcome = {'correct','incorrect','all'};

%% loop through different trial types to plot data
for outIdx = 1:3
    %% compile averages, sems, etc for trials of interest
    %get which trials to grab
    if outIdx == 3
        outcomeInds = 1:numel(positionData.trialOutcomes); %use all the trials
    else
        outcomeInds = find(positionData.trialOutcomes == params.choiceMap(trialOutcome{outIdx}));
    end
    whichTrialsToPlot = intersect(outcomeInds,trialTypeInds);
    rightTrialInds = intersect(whichTrialsToPlot,find(positionData.trialTypesLeftRight == params.trialTypeMap('right')));
    leftTrialInds = intersect(whichTrialsToPlot,find(positionData.trialTypesLeftRight == params.trialTypeMap('left')));
    
    %get the values of interest for the specific trials
    positionDataToPlotRight = positionData.histY(rightTrialInds);
    positionDataToPlotLeft = positionData.histY(leftTrialInds);
    
    %manipulate the mean/hist values of the data structures into the format we want
    whichVarsAreMeans = find(cellfun(@(x) ~isempty(x),(regexp(positionDataToPlotRight{1}.Properties.VariableNames,'mean*')))); %get columns with mean values
    trialPosDataRight = rowfun(@(x) varfun(@(y) cell2mat(y)',x{1}(:,whichVarsAreMeans)), cell2table(positionDataToPlotRight),'OutputVariableNames','posDataToPlotRight'); %makes nTrialsxMetrics table
    trialPosDataLeft = rowfun(@(x) varfun(@(y) cell2mat(y)',x{1}(:,whichVarsAreMeans)), cell2table(positionDataToPlotLeft),'OutputVariableNames','posDataToPlotLeft');
    trialPosDataRight = trialPosDataRight.posDataToPlotRight; trialPosDataLeft = trialPosDataLeft.posDataToPlotLeft;
    meanPosDataRight = varfun(@(x) nanmean(x), trialPosDataRight); %makes meanxMetrics table
    semPosDataRight = varfun(@(x) nanstd(x)/sqrt(size(trialPosDataRight,1)), trialPosDataRight); %makes semxMetrics table
    meanPosDataLeft = varfun(@(x) nanmean(x), trialPosDataLeft);
    semPosDataLeft = varfun(@(x) nanstd(x)/sqrt(size(trialPosDataLeft,1)), trialPosDataLeft);
    
    %manipulate the all bins values of the data structures into the format we want
    %         whichVarsAreNotMeans = find(cellfun(@(x) isempty(x),(regexp(positionDataToPlotRight{1}.Properties.VariableNames,'mean*'))));
    %         allPosDataRightByTrial = rowfun(@(x) varfun(@(y) y',x{1}(:,whichVarsAreNotMeans)), cell2table(positionDataToPlotRight),'OutputVariableNames','posDataToPlotRight');
    %         allPosDataLeftByTrial = rowfun(@(x) varfun(@(y) y',x{1}(:,whichVarsAreNotMeans)), cell2table(positionDataToPlotRight),'OutputVariableNames','posDataToPlotLeft');
    %         allPosDataRight = varfun(@(x) arrayfun(@(y) vertcat(x{:,y}),(1:size(x,2))','UniformOutput',0), allPosDataRightByTrial.posDataToPlotRight);
    %         allPosDataLeft = varfun(@(x) arrayfun(@(y) vertcat(x{:,y}),(1:size(x,2))','UniformOutput',0), allPosDataLeftByTrial.posDataToPlotLeft);
    %         allPosDataRight.Properties.VariableNames = regexprep(allPosDataRight.Properties.VariableNames, 'Fun_Fun_', 'all'); %indicates ALL non-averaged values
    %         allPosDataLeft.Properties.VariableNames = regexprep(allPosDataLeft.Properties.VariableNames, 'Fun_Fun_', 'all');
    
    %replace the variable names so we can use them for plotting
    meanPosDataRight.Properties.VariableNames = regexprep(meanPosDataRight.Properties.VariableNames, 'Fun_Fun_', '');
    meanPosDataLeft.Properties.VariableNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'Fun_Fun_', '');
    semPosDataRight.Properties.VariableNames = regexprep(semPosDataRight.Properties.VariableNames, 'Fun_Fun_mean', 'sem');
    semPosDataLeft.Properties.VariableNames = regexprep(semPosDataLeft.Properties.VariableNames, 'Fun_Fun_mean', 'sem');
    trialPosDataRight.Properties.VariableNames = regexprep(trialPosDataRight.Properties.VariableNames, 'Fun_mean', 'indiv'); %indicates hist/mean for individual trials
    trialPosDataLeft.Properties.VariableNames = regexprep(trialPosDataLeft.Properties.VariableNames, 'Fun_mean', 'indiv');
    
    %% make a plot for each of the variables/metrics
    numTrialsTotal = size(trialPosDataRight,1) + size(trialPosDataLeft,1);
    varNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'mean', '');
    trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
    updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
    for varIdx = 1:size(varNames,2)
        %get bins to use and clean relevant data
        binsToUse = binsTable.yPos - min(binsTable.yPos)';
        meanR = meanPosDataRight.(['mean' varNames{varIdx}]); semR = semPosDataRight.(['sem' varNames{varIdx}]);
        meanL = meanPosDataLeft.(['mean' varNames{varIdx}]); semL = semPosDataLeft.(['sem' varNames{varIdx}]);
        
        %plot the individual traces
        figure; hold on;
        ax(1) = subplot(1,2,1); hold on;
        indivR = trialPosDataRight.(['indiv' varNames{varIdx}]);
        indivL = trialPosDataLeft.(['indiv' varNames{varIdx}]);
        allFullBins = find(sum(~isnan(indivR)) == size(indivR,1)); %look for bins that are all full
        plot(binsToUse, indivR,'Color', [1 0 0 0.2], 'LineWidth', 1); 
        plot(binsToUse, indivL, 'Color', [0 0 1 0.2], 'LineWidth', 1);
        xlim([binsToUse(min(allFullBins)-1) binsToUse(max(allFullBins)+1)]) 
        xlabel('Y Position'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
        
        %plot the averages
        ax(2) = subplot(1,2,2); hold on;
        plot(binsToUse, meanR, 'r-', 'LineWidth', 2);
        ciplot(meanR-semR, meanR+semR, binsToUse,'r-'); alpha(0.3); hold on;
        plot(binsToUse, meanL, 'b-', 'LineWidth', 2);
        ciplot(meanL-semL, meanL+semL, binsToUse,'b-'); alpha(0.3); hold on;
        xlabel('Y Position'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
        title([trialOutcome{outIdx} 'trials - delay loc: ' num2str(params.plotCategories(paramIdx,2)) ' - trial type:' updateType '  n=' num2str(numTrialsTotal) 'right = red, left = blue']);
        alpha(0.5); xlim([binsToUse(min(allFullBins)-1) binsToUse(max(allFullBins)+1)]) 
        linkaxes(ax,'y')
        
        sgtitle(['S' num2str(indices.animals(anIdx)) ' trajectories on' trackName]);
        filename = [dirs.behaviorfigdir varNames{varIdx} 'trajectories_' trackName '_S' num2str(indices.animals(anIdx)) '_' trialOutcome{outIdx}];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        %plot the traces as distributions for different parts of track
        indivRShort = indivR(:,allFullBins); indivLShort = indivL(:,allFullBins);
        figure; hold on;
        maxRight = max(max(indivRShort)); minRight = min(min(indivRShort));
        maxLeft = max(max(indivLShort)); minLeft = min(min(indivLShort));
        maxAll = max([maxRight maxLeft]); minAll = min([minRight minLeft]);
        histBins = [1:2:numel(allFullBins)]; edges = linspace(minAll, maxAll, 30); plotNums = [1:2:numel(allFullBins); 2:2:numel(allFullBins)+1]';
        %edges = -1.5:0.1:1.5;
        for posIdx = 1:numel(histBins)
            viewAnglesDistR = histcounts(indivRShort(:,histBins(posIdx):histBins(posIdx)+1), edges);
            viewAnglesDistRNorm = viewAnglesDistR/nansum(viewAnglesDistR);
            viewAnglesDistL = histcounts(indivLShort(:,histBins(posIdx):histBins(posIdx)+1), edges);
            viewAnglesDistLNorm = viewAnglesDistL/nansum(viewAnglesDistL);
            
            %plot correct trials
            ax2(posIdx) = subplot(numel(histBins),2,plotNums(numel(histBins)-posIdx+1,1)); hold on;
            h1 = histogram('BinCounts', viewAnglesDistRNorm, 'BinEdges', rad2deg(edges));
            h2 = histogram('BinCounts', viewAnglesDistLNorm, 'BinEdges', rad2deg(edges));
            h1.FaceAlpha = 0.2; h2.FaceAlpha = 0.2; 
            h1.FaceColor = [1 0 0]; h2.FaceColor = [0 0 1];
            if posIdx ~= 1; set(ax2(posIdx),'xticklabel',[]); end;
            if posIdx == numel(histBins); title(['View Angle Distributions on' trialOutcome{outIdx} 'Trials']); end;
            if mod(posIdx,2); ylabel(['y pos -' num2str(round(nanmean(binsToUse(histBins(posIdx):histBins(posIdx)+1))))]); end;
            set(gca,'tickdir','out')
        end
        sgtitle(['S' num2str(indices.animals(anIdx)) ' trajectory dist on ' trackName]);
        filename = [dirs.behaviorfigdir varNames{varIdx} 'trajectories_' trackName '_S' num2str(indices.animals(anIdx)) '_pos' '_' trialOutcome{outIdx} 'dists'];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end
end
end
