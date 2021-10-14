function plotPositionAroundUpdate(trialdata, positionData, binsTable, anIdx, paramIdx, indices, dirs, params);

if ~isempty(trialdata)
    %initialize variables
    if params.plotCategories(paramIdx,3) == 2
        savedfiguresdir = [dirs.behaviorfigdir 'trajectories\aroundupdate\'];
    elseif params.plotCategories(paramIdx,3) == 3
        savedfiguresdir = [dirs.behaviorfigdir 'trajectories\aroundstay\'];
    end
    if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;

    updateTypeKeySet = params.updateTypeMap.keys; updateTypeValueSet = params.updateTypeMap.values;
    trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;
    trialOutcome = {'correct','incorrect','all'};
    trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
    updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
    binWindowToPlot = 15; binSize = diff(binsTable.yPos(1:2));
    binsToUseForUpdate = -binWindowToPlot*binSize:binSize:binWindowToPlot*binSize;

    %% loop through different trial types to plot data
    for outIdx = 1:3
        %% compile averages, sems, etc for trials of interest
        %get which trials to grab
        if outIdx == 3
            correctTrials = find(positionData.trialOutcomes == params.choiceMap('correct'));
            incorrectTrials = find(positionData.trialOutcomes == params.choiceMap('incorrect'));
            whichTrialsToPlot = union(correctTrials, incorrectTrials); %use both correct and incorrect trials
        else
            whichTrialsToPlot = find(positionData.trialOutcomes == params.choiceMap(trialOutcome{outIdx}));
        end
        rightTrialInds = intersect(whichTrialsToPlot,find(positionData.trialTypesLeftRight == params.trialTypeMap('right')));
        leftTrialInds = intersect(whichTrialsToPlot,find(positionData.trialTypesLeftRight == params.trialTypeMap('left')));

        %get the values of interest for the specific trials
        positionDataToPlotRight = positionData.histY(rightTrialInds);
        positionDataToPlotLeft = positionData.histY(leftTrialInds);
        whereUpdateOccurredRight = positionData.trialUpdateLocation(rightTrialInds);
        whereUpdateOccurredLeft = positionData.trialUpdateLocation(leftTrialInds);

        %manipulate the mean/hist values of the data structures into the format we want
        if ~isempty(positionDataToPlotRight)
            whichVarsAreMeans = find(cellfun(@(x) ~isempty(x),(regexp(positionDataToPlotRight{1}.Properties.VariableNames,'mean*')))); %get columns with mean values
        elseif ~isempty(positionDataToPlotLeft)
            whichVarsAreMeans = find(cellfun(@(x) ~isempty(x),(regexp(positionDataToPlotLeft{1}.Properties.VariableNames,'mean*')))); %get columns with mean values            
        end
        trialPosDataRight = rowfun(@(x) varfun(@(y) cell2mat(y)',x{1}(:,whichVarsAreMeans)), cell2table(positionDataToPlotRight),'OutputVariableNames','posDataToPlotRight'); %makes nTrialsxMetrics table
        trialPosDataLeft = rowfun(@(x) varfun(@(y) cell2mat(y)',x{1}(:,whichVarsAreMeans)), cell2table(positionDataToPlotLeft),'OutputVariableNames','posDataToPlotLeft');
        trialPosDataRight = trialPosDataRight.posDataToPlotRight; trialPosDataLeft = trialPosDataLeft.posDataToPlotLeft;

        %get when the update occurred and look in specific window around that
        binsToUse = binsTable.yPos - min(binsTable.yPos)';
        whereUpdateOccurredAdjustedR = whereUpdateOccurredRight - min(binsTable.yPos); %adjust position values for the adjustment I make above
        whereUpdateOccurredAdjustedL = whereUpdateOccurredLeft - min(binsTable.yPos);
        whichBinUpdateOccurredR = lookup2(whereUpdateOccurredAdjustedR,binsToUse);
        whichBinUpdateOccurredL = lookup2(whereUpdateOccurredAdjustedL,binsToUse);
        whichBinUpdateOccurredR(isnan(whereUpdateOccurredAdjustedR)) = [];
        whichBinUpdateOccurredL(isnan(whereUpdateOccurredAdjustedL)) = [];
        trialPosDataRight(isnan(whereUpdateOccurredAdjustedR),:) = [];
        trialPosDataLeft(isnan(whereUpdateOccurredAdjustedL),:) = [];
        
        %get averages for when update occurred
        if ~isempty(whichBinUpdateOccurredR)
            trialPosDataUpdateRight = varfun(@(y) cell2mat(arrayfun(@(x) y(x,whichBinUpdateOccurredR(x)-binWindowToPlot:whichBinUpdateOccurredR(x)+binWindowToPlot), 1:numel(whichBinUpdateOccurredR), 'UniformOutput',0)'),trialPosDataRight);
            meanPosDataRight = varfun(@(x) nanmean(x,1), trialPosDataUpdateRight); %makes meanxMetrics table
            semPosDataRight = varfun(@(x) nanstd(x,[],1)/sqrt(size(trialPosDataUpdateRight,1)), trialPosDataUpdateRight); %makes semxMetrics table
            
            %replace the variable names so we can use them for plotting
            trialPosDataUpdateRight.Properties.VariableNames = regexprep(trialPosDataUpdateRight.Properties.VariableNames, 'Fun_Fun_mean', 'update'); %indicates hist/mean for individual trials
            meanPosDataRight.Properties.VariableNames = regexprep(meanPosDataRight.Properties.VariableNames, 'Fun_Fun_Fun_', '');
            semPosDataRight.Properties.VariableNames = regexprep(semPosDataRight.Properties.VariableNames, 'Fun_Fun_Fun_mean', 'sem');
            varNames = regexprep(meanPosDataRight.Properties.VariableNames, 'mean', '');
        else
            trialPosDataUpdateRight = [];
            meanPosDataRight = [];
            semPosDataRight = [];
        end
        if ~isempty(whichBinUpdateOccurredL)
            trialPosDataUpdateLeft = varfun(@(y) cell2mat(arrayfun(@(x) y(x,whichBinUpdateOccurredL(x)-binWindowToPlot:whichBinUpdateOccurredL(x)+binWindowToPlot), 1:numel(whichBinUpdateOccurredL), 'UniformOutput',0)'),trialPosDataLeft);
            meanPosDataLeft = varfun(@(x) nanmean(x,1), trialPosDataUpdateLeft);
            semPosDataLeft = varfun(@(x) nanstd(x,[],1)/sqrt(size(trialPosDataUpdateLeft,1)), trialPosDataUpdateLeft);
            
            %replace the variable names so we can use them for plotting
            trialPosDataUpdateLeft.Properties.VariableNames = regexprep(trialPosDataUpdateLeft.Properties.VariableNames, 'Fun_Fun_mean', 'update'); %indicates hist/mean for individual trials
            meanPosDataLeft.Properties.VariableNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'Fun_Fun_Fun_', '');
            semPosDataLeft.Properties.VariableNames = regexprep(semPosDataLeft.Properties.VariableNames, 'Fun_Fun_Fun_mean', 'sem');
            varNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'mean', '');
        else
            trialPosDataUpdateLeft = [];
            meanPosDataLeft = [];
            semPosDataLeft = [];
        end

        %% make a plot for each of the variables/metrics 
        numTrialsTotal = size(trialPosDataUpdateRight,1) + size(trialPosDataUpdateLeft,1);
        for varIdx = [2 3 5]
            %get bins to use and clean relevant data
            if ~isempty(trialPosDataUpdateRight)
                meanR = meanPosDataRight.(['mean' varNames{varIdx}]); semR = semPosDataRight.(['sem' varNames{varIdx}]);
                indivR = trialPosDataUpdateRight.(['update' varNames{varIdx}]);
            else
                meanR = nan(size(binsToUseForUpdate)); semR = nan(size(binsToUseForUpdate));
                indivR = nan(size(binsToUseForUpdate));
            end
            if ~isempty(trialPosDataUpdateLeft)
                meanL = meanPosDataLeft.(['mean' varNames{varIdx}]); semL = semPosDataLeft.(['sem' varNames{varIdx}]);
                indivL = trialPosDataUpdateLeft.(['update' varNames{varIdx}]);
            else
                meanL = nan(size(binsToUseForUpdate)); semL = nan(size(binsToUseForUpdate));
                indivL = nan(size(binsToUseForUpdate));
            end

            %plot the individual traces around the location at which the update occurred
            figure(varIdx); hold on;
            plot([0 0], [min(min((indivR))) max(max((indivR)))], 'k--');
            ax(outIdx) = subplot(4,3,outIdx); hold on;
            plot(binsToUseForUpdate, indivR,'Color', [0 1 0 0.2], 'LineWidth', 1);
            plot(binsToUseForUpdate, indivL, 'Color', [1 0 1 0.2], 'LineWidth', 1);
            xlim([min(binsToUseForUpdate) max(binsToUseForUpdate)])
            xlabel('Position from Update'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
            title([trialOutcome{outIdx} 'trials n=' num2str(numTrialsTotal)]);

            %plot the averages
            ax2(outIdx) = subplot(4,3,outIdx+3); hold on;
            plot([0 0], [min(meanR) max(meanR)], 'k--');
            plot(binsToUseForUpdate, meanR, 'g-', 'LineWidth', 2);
            ciplot(meanR-semR, meanR+semR, binsToUseForUpdate,'g-'); alpha(0.3); hold on;
            plot(binsToUseForUpdate, meanL, 'm-', 'LineWidth', 2);
            ciplot(meanL-semL, meanL+semL, binsToUseForUpdate,'m-'); alpha(0.3); hold on;
            xlabel('Position from Update'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
            alpha(0.5); xlim([min(binsToUseForUpdate) max(binsToUseForUpdate)])

            %plot the heat maps
            ax3(outIdx) = subplot(4,3,outIdx+6); hold on;
            indivTrialsForHeatmap = [-1*indivR; indivL]; %flip the indivL so they're all the same direction
            [~, sortIdx] = sort(sum(indivTrialsForHeatmap(:,1:binWindowToPlot+1),2));
            imagesc('CData',indivTrialsForHeatmap(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmap))) abs(max(max(indivTrialsForHeatmap)))]);
            if ~isnan(limVal)
                caxis([-limVal*0.5 limVal*0.5]);                 
            end
            set(gca,'YDir','normal');
            title('Velocity towards initial side')

            ax3(outIdx) = subplot(4,3,outIdx+9); hold on;
            indivTrialsForHeatmap = [indivR; -1*indivL]; %flip the indivL so they're all the same direction
            [~, sortIdx] = sort(sum(indivTrialsForHeatmap(:,1:binWindowToPlot+1),2));
            imagesc('CData',indivTrialsForHeatmap(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmap))) abs(max(max(indivTrialsForHeatmap)))]);
            if ~isnan(limVal)
                caxis([-limVal*0.5 limVal*0.5]);
            end
            set(gca,'YDir','normal');
            title('Velocity towards update side')
            
            if isnumeric(anIdx)
                anID = num2str(indices.animals(anIdx));
            else
                anID = anIdx;
            end
            sgtitle(['S' anID ' trajectories around update']);
            filename = [savedfiguresdir varNames{varIdx} 'trajectoriesaroundupdate_S' anID];
            saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

            %plot just as difference after the update cue occurs
            figure(varIdx+10)
            ax11(outIdx) = subplot(3,3,outIdx); hold on;
            indivTrialsForHeatmap = [indivR; -1*indivL];
            indivTrialsForHeatmapValAtHalf = indivTrialsForHeatmap(:,binWindowToPlot+1);
            indivTrialsForHeatmapAdjusted = indivTrialsForHeatmap-indivTrialsForHeatmapValAtHalf;
            [~, sortIdx] = sort(sum(indivTrialsForHeatmapAdjusted,2));
            imagesc('CData',indivTrialsForHeatmapAdjusted(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmapAdjusted))) abs(max(max(indivTrialsForHeatmapAdjusted)))]);
            if ~isnan(limVal)
                caxis([-limVal limVal]);
            end
            set(gca,'YDir','normal');
            title('Velocity towards update side')

            ax12(outIdx) = subplot(3,3,outIdx+3); hold on;
            indivTrialsForHeatmap = [-1*indivR; indivL];
            indivTrialsForHeatmapValAtHalf = indivTrialsForHeatmap(:,binWindowToPlot+1);
            indivTrialsForHeatmapAdjusted = indivTrialsForHeatmap-indivTrialsForHeatmapValAtHalf;
            [~, sortIdx] = sort(sum(indivTrialsForHeatmapAdjusted,2));
            imagesc('CData',indivTrialsForHeatmapAdjusted(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmapAdjusted))) abs(max(max(indivTrialsForHeatmapAdjusted)))]);
            if ~isnan(limVal)
                caxis([-limVal limVal]);
            end
            set(gca,'YDir','normal');
            title('Velocity towards initial side')

            ax10(outIdx) = subplot(3,3,outIdx+6); hold on;
            meanRadjusted = mean(indivTrialsForHeatmapAdjusted,1);
            semRadjusted = std(indivTrialsForHeatmapAdjusted,[],1)/size(indivTrialsForHeatmapAdjusted,1);

            plot([0 0], [min(meanRadjusted) max(meanRadjusted)], 'k--');
            plot(binsToUseForUpdate, meanRadjusted, 'g-', 'LineWidth', 2);
            ciplot(meanRadjusted-semRadjusted, meanRadjusted+semRadjusted, binsToUseForUpdate,'g-'); alpha(0.3); hold on;
            xlabel('Position from Update'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
            alpha(0.5); xlim([min(binsToUseForUpdate) max(binsToUseForUpdate)])
            title([trialOutcome{outIdx} 'trials n=' num2str(numTrialsTotal)]);

            sgtitle(['S' anID ' velocity change after update']);
            filename = [savedfiguresdir varNames{varIdx} 'trajectoriesaroundupdatedifffrombaseline_S' anID];
            saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        end
    end
end

end
