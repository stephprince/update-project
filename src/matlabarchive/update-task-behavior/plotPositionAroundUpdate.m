function plotPositionAroundUpdate(trialdata, positionData, binsTable, anIdx, paramIdx, indices, dirs, params);

if ~isempty(trialdata)
    %initialize variables
    savedfiguresdir = [dirs.behaviorfigdir 'trajectories\aroundupdate\'];
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
        whichVarsAreMeans = find(cellfun(@(x) ~isempty(x),(regexp(positionDataToPlotRight{1}.Properties.VariableNames,'mean*')))); %get columns with mean values
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
        trialPosDataUpdateRight = varfun(@(y) cell2mat(arrayfun(@(x) y(x,whichBinUpdateOccurredR(x)-binWindowToPlot:whichBinUpdateOccurredR(x)+binWindowToPlot), 1:numel(whichBinUpdateOccurredR), 'UniformOutput',0)'),trialPosDataRight);
        trialPosDataUpdateLeft = varfun(@(y) cell2mat(arrayfun(@(x) y(x,whichBinUpdateOccurredL(x)-binWindowToPlot:whichBinUpdateOccurredL(x)+binWindowToPlot), 1:numel(whichBinUpdateOccurredL), 'UniformOutput',0)'),trialPosDataLeft);   

        %get averages for when update occurred
        meanPosDataRight = varfun(@(x) nanmean(x,1), trialPosDataUpdateRight); %makes meanxMetrics table
        semPosDataRight = varfun(@(x) nanstd(x,[],1)/sqrt(size(trialPosDataUpdateRight,1)), trialPosDataUpdateRight); %makes semxMetrics table
        meanPosDataLeft = varfun(@(x) nanmean(x,1), trialPosDataUpdateLeft);
        semPosDataLeft = varfun(@(x) nanstd(x,[],1)/sqrt(size(trialPosDataUpdateLeft,1)), trialPosDataUpdateLeft);

        %replace the variable names so we can use them for plotting
        trialPosDataUpdateRight.Properties.VariableNames = regexprep(trialPosDataUpdateRight.Properties.VariableNames, 'Fun_Fun_mean', 'update'); %indicates hist/mean for individual trials
        trialPosDataUpdateLeft.Properties.VariableNames = regexprep(trialPosDataUpdateLeft.Properties.VariableNames, 'Fun_Fun_mean', 'update'); %indicates hist/mean for individual trials
        meanPosDataRight.Properties.VariableNames = regexprep(meanPosDataRight.Properties.VariableNames, 'Fun_Fun_Fun_', '');
        meanPosDataLeft.Properties.VariableNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'Fun_Fun_Fun_', '');
        semPosDataRight.Properties.VariableNames = regexprep(semPosDataRight.Properties.VariableNames, 'Fun_Fun_Fun_mean', 'sem');
        semPosDataLeft.Properties.VariableNames = regexprep(semPosDataLeft.Properties.VariableNames, 'Fun_Fun_Fun_mean', 'sem');

        %% make a plot for each of the variables/metrics
        varNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'mean', '');
        numTrialsTotal = size(trialPosDataUpdateRight,1) + size(trialPosDataUpdateLeft,1);
        for varIdx = 5%[2 3 5]
            %get bins to use and clean relevant data
            meanR = meanPosDataRight.(['mean' varNames{varIdx}]); semR = semPosDataRight.(['sem' varNames{varIdx}]);
            meanL = meanPosDataLeft.(['mean' varNames{varIdx}]); semL = semPosDataLeft.(['sem' varNames{varIdx}]);
            indivR = trialPosDataUpdateRight.(['update' varNames{varIdx}]);
            indivL = trialPosDataUpdateLeft.(['update' varNames{varIdx}]);

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
            caxis([-limVal*0.5 limVal*0.5]); set(gca,'YDir','normal');
            title('Velocity towards initial side')
            
            ax3(outIdx) = subplot(4,3,outIdx+9); hold on;
            indivTrialsForHeatmap = [indivR; -1*indivL]; %flip the indivL so they're all the same direction
            [~, sortIdx] = sort(sum(indivTrialsForHeatmap(:,1:binWindowToPlot+1),2));
            imagesc('CData',indivTrialsForHeatmap(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmap))) abs(max(max(indivTrialsForHeatmap)))]);
            caxis([-limVal*0.5 limVal*0.5]); set(gca,'YDir','normal');
            title('Velocity towards update side')

%             ax4(outIdx) = subplot(4,3,outIdx+9); hold on;
%             [~, sortIdx] = sort(sum(indivL(:,1:binWindowToPlot+1),2));
%             imagesc('CData',indivL(sortIdx,:), 'XData', binsToUseForUpdate)
%             xlabel('Position from Update'); ylabel('Trials');
%             cmap = flipud(cbrewer('div', 'RdBu',100));
%             colormap(cmap); colorbar
%             limVal = min([abs(min(min(indivL))) abs(max(max(indivL)))]);
%             caxis([-limVal limVal]); set(gca,'YDir','normal');
%             title('Initial left trials')

            sgtitle(['S' num2str(indices.animals(anIdx)) ' trajectories around update']);
            filename = [savedfiguresdir varNames{varIdx} 'trajectoriesaroundupdate_S' num2str(indices.animals(anIdx))];
            saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

            %plot just as difference after the update cue occurs
            figure(varIdx+10)
            ax11(outIdx) = subplot(3,3,outIdx); hold on;
            indivTrialsForHeatmap = [indivR; -1*indivL];
            %indivTrialsForHeatmapSecondHalf = indivTrialsForHeatmap(:,binWindowToPlot+2:end);
            indivTrialsForHeatmapValAtHalf = indivTrialsForHeatmap(:,binWindowToPlot+1);
            indivTrialsForHeatmapAdjusted = indivTrialsForHeatmap-indivTrialsForHeatmapValAtHalf;
            [~, sortIdx] = sort(sum(indivTrialsForHeatmapAdjusted,2));
            imagesc('CData',indivTrialsForHeatmapAdjusted(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmapAdjusted))) abs(max(max(indivTrialsForHeatmapAdjusted)))]);
            caxis([-limVal limVal]); set(gca,'YDir','normal');
            title('Velocity towards update side')
            
            ax12(outIdx) = subplot(3,3,outIdx+3); hold on;
            indivTrialsForHeatmap = [-1*indivR; indivL];
            %indivTrialsForHeatmapSecondHalf = indivTrialsForHeatmap(:,binWindowToPlot+2:end);
            indivTrialsForHeatmapValAtHalf = indivTrialsForHeatmap(:,binWindowToPlot+1);
            indivTrialsForHeatmapAdjusted = indivTrialsForHeatmap-indivTrialsForHeatmapValAtHalf;
            [~, sortIdx] = sort(sum(indivTrialsForHeatmapAdjusted,2));
            imagesc('CData',indivTrialsForHeatmapAdjusted(sortIdx,:), 'XData', binsToUseForUpdate)
            xlabel('Position from Update'); ylabel('Trials');
            cmap = flipud(cbrewer('div', 'RdBu',100));
            colormap(cmap); colorbar
            limVal = min([abs(min(min(indivTrialsForHeatmapAdjusted))) abs(max(max(indivTrialsForHeatmapAdjusted)))]);
            caxis([-limVal limVal]); set(gca,'YDir','normal');
            title('Velocity towards initial side')
            
            

%             ax12(outIdx) = subplot(3,3,outIdx+3); hold on;
%             indivLSecondHalf = indivL(:,binWindowToPlot+2:end);
%             indivLValAtHalf = indivL(:,binWindowToPlot+1);
%             indivLAdjusted = indivLSecondHalf-indivLValAtHalf;
%             [~, sortIdx] = sort(sum(indivLAdjusted,2));
%             imagesc('CData',indivLAdjusted(sortIdx,:), 'XData', binsToUseForUpdate(binWindowToPlot+2:end))
%             xlabel('Position from Update'); ylabel('Trials');
%             cmap = flipud(cbrewer('div', 'RdBu',100));
%             colormap(cmap); colorbar
%             limVal = min([abs(min(min(indivLAdjusted))) abs(max(max(indivLAdjusted)))]);
%             caxis([-limVal limVal]); set(gca,'YDir','normal');
%             title('Initial left trials')

            ax10(outIdx) = subplot(3,3,outIdx+6); hold on;
            meanRadjusted = mean(indivTrialsForHeatmapAdjusted,1); 
            semRadjusted = std(indivTrialsForHeatmapAdjusted,[],1)/size(indivTrialsForHeatmapAdjusted,1);
            %meanLadjusted = mean(indivLAdjusted,1);
            %semLadjusted = std(indivLAdjusted,[],1)/size(indivLAdjusted,1);

            plot([0 0], [min(meanRadjusted) max(meanRadjusted)], 'k--');
            plot(binsToUseForUpdate, meanRadjusted, 'g-', 'LineWidth', 2);
            ciplot(meanRadjusted-semRadjusted, meanRadjusted+semRadjusted, binsToUseForUpdate,'g-'); alpha(0.3); hold on;
            %plot(binsToUseForUpdate(binWindowToPlot+2:end), meanLadjusted, 'm-', 'LineWidth', 2);
            %ciplot(meanLadjusted-semLadjusted, meanLadjusted+semLadjusted, binsToUseForUpdate(binWindowToPlot+2:end),'m-'); alpha(0.3); hold on;
            xlabel('Position from Update'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
            alpha(0.5); xlim([min(binsToUseForUpdate) max(binsToUseForUpdate)]) 
            title([trialOutcome{outIdx} 'trials n=' num2str(numTrialsTotal)]);

            sgtitle(['S' num2str(indices.animals(anIdx)) ' velocity change after update']);
            filename = [savedfiguresdir varNames{varIdx} 'trajectoriesaroundupdatedifffrombaseline_S' num2str(indices.animals(anIdx))];
            saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        end
    end
end