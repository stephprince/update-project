function plotHistByPosition(trialdata, positionData, binsTable, anIdx, paramIdx, indices, dirs, params);

if ~isempty(trialdata)
    %initialize variables
    savedfiguresdir = [dirs.behaviorfigdir 'trajectories\'];
    if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;

    delayLoc = params.plotCategories(paramIdx,2);
    updateTypeKeySet = params.updateTypeMap.keys; updateTypeValueSet = params.updateTypeMap.values;
    trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;
    trialOutcome = {'correct','incorrect','all'};

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

        %manipulate the mean/hist values of the data structures into the format
        %we want
        if ~isempty(positionDataToPlotRight)
            whichVarsAreMeans = find(cellfun(@(x) ~isempty(x),(regexp(positionDataToPlotRight{1}.Properties.VariableNames,'mean*')))); %get columns with mean values
            trialPosDataRight = rowfun(@(x) varfun(@(y) cell2mat(y)',x{1}(:,whichVarsAreMeans)), cell2table(positionDataToPlotRight),'OutputVariableNames','posDataToPlotRight'); %makes nTrialsxMetrics table
            trialPosDataRight = trialPosDataRight.posDataToPlotRight; 
            meanPosDataRight = varfun(@(x) nanmean(x,1), trialPosDataRight); %makes meanxMetrics table
            semPosDataRight = varfun(@(x) nanstd(x,[],1)/sqrt(size(trialPosDataRight,1)), trialPosDataRight); %makes semxMetrics table

            meanPosDataRight.Properties.VariableNames = regexprep(meanPosDataRight.Properties.VariableNames, 'Fun_Fun_', '');
            semPosDataRight.Properties.VariableNames = regexprep(semPosDataRight.Properties.VariableNames, 'Fun_Fun_mean', 'sem');
            trialPosDataRight.Properties.VariableNames = regexprep(trialPosDataRight.Properties.VariableNames, 'Fun_mean', 'indiv'); %indicates hist/mean for individual trials
            varNames = regexprep(meanPosDataRight.Properties.VariableNames, 'mean', '');    
        else
            trialPosDataRight = [];
        end
        if ~isempty(positionDataToPlotLeft)
            whichVarsAreMeans = find(cellfun(@(x) ~isempty(x),(regexp(positionDataToPlotLeft{1}.Properties.VariableNames,'mean*')))); %get columns with mean values
            trialPosDataLeft = rowfun(@(x) varfun(@(y) cell2mat(y)',x{1}(:,whichVarsAreMeans)), cell2table(positionDataToPlotLeft),'OutputVariableNames','posDataToPlotLeft');
            trialPosDataLeft = trialPosDataLeft.posDataToPlotLeft;
            meanPosDataLeft = varfun(@(x) nanmean(x,1), trialPosDataLeft);
            semPosDataLeft = varfun(@(x) nanstd(x,[],1)/sqrt(size(trialPosDataLeft,1)), trialPosDataLeft);

            meanPosDataLeft.Properties.VariableNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'Fun_Fun_', '');
            semPosDataLeft.Properties.VariableNames = regexprep(semPosDataLeft.Properties.VariableNames, 'Fun_Fun_mean', 'sem');
            trialPosDataLeft.Properties.VariableNames = regexprep(trialPosDataLeft.Properties.VariableNames, 'Fun_mean', 'indiv');
            varNames = regexprep(meanPosDataLeft.Properties.VariableNames, 'mean', '');
        else
            trialPosDataLeft = [];
        end   

        %% make a plot for each of the variables/metrics
        numTrialsTotal = size(trialPosDataRight,1) + size(trialPosDataLeft,1);
        trackName = trackTypeKeySet{trackTypeValueSet{params.plotCategories(paramIdx,1)}};
        updateType = updateTypeKeySet{updateTypeValueSet{params.plotCategories(paramIdx,3)}};
        if exist('varNames') %if any of the left/right trials have values
            for varIdx = 5%[2 3 5] %don't really care about the transvelocity or ypos
                %get bins to use and clean relevant data
                binsToUse = binsTable.yPos - min(binsTable.yPos)';
                meanR = nan(size(binsToUse))'; semR = nan(size(binsToUse))';
                meanL = nan(size(binsToUse))'; semL = nan(size(binsToUse))';
                if ~isempty(trialPosDataRight)
                    meanR = meanPosDataRight.(['mean' varNames{varIdx}]); semR = semPosDataRight.(['sem' varNames{varIdx}]);
                end
                if ~isempty(trialPosDataLeft)
                    meanL = meanPosDataLeft.(['mean' varNames{varIdx}]); semL = semPosDataLeft.(['sem' varNames{varIdx}]);
                end

                desperate = 1;
                
                %plot the individual traces
                figure(varIdx); hold on;
                set(gcf,'units','normalized','position',[0 0 0.8 0.8]); hold on;
                ax(outIdx) = subplot(2,3,outIdx); hold on;
                indivR = nan(size(binsToUse))'; indivL = nan(size(binsToUse))';
                if ~isempty(trialPosDataRight)
                    indivR = trialPosDataRight.(['indiv' varNames{varIdx}]);
                    allFullBins = find(sum(~isnan(indivR)) >= size(indivR,1)/2); %look for bins that are at least half full
                end
                if ~isempty(trialPosDataLeft)
                    indivL = trialPosDataLeft.(['indiv' varNames{varIdx}]);
                    allFullBins = find(sum(~isnan(indivL),1) >= size(indivL,1)/2); %look for bins that are at least half full
                end
                
                if ~desperate
                    plot(binsToUse, indivR,'Color', [0 1 0 0.2], 'LineWidth', 1); 
                    plot(binsToUse, indivL, 'Color', [1 0 1 0.2], 'LineWidth', 1);
                    xlim([binsToUse(min(allFullBins)) binsToUse(max(allFullBins))]) 
                    xlabel('Y Position'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');

                    %plot the averages
                    ax(outIdx+3) = subplot(2,3,outIdx+3); hold on;
                    if ~isnan(nansum(meanR))
                        plot(binsToUse, meanR, 'g-', 'LineWidth', 2);
                        ciplot(meanR-semR, meanR+semR, binsToUse,'g'); alpha(0.3); hold on;
                    end
                    if ~isnan(nansum(meanL))
                        plot(binsToUse, meanL, 'm-', 'LineWidth', 2);
                        ciplot(meanL-semL, meanL+semL, binsToUse,'m'); alpha(0.3); hold on;
                    end
                    xlabel('Y Position'); ylabel(varNames{varIdx}); set(gca,'tickdir','out');
                    title([trialOutcome{outIdx} 'trials - delay loc: ' num2str(params.plotCategories(paramIdx,2)) ' - trial type:' updateType '  n=' num2str(numTrialsTotal) 'right = red, left = blue']);
                    alpha(0.5); xlim([binsToUse(min(allFullBins)) binsToUse(max(allFullBins))]) 
                    linkaxes(ax,'y')

                    sgtitle(['S' num2str(indices.animals(anIdx)) ' trajectories on' trackName]);
                    filename = [savedfiguresdir varNames{varIdx} 'trajectories_' trackName '_S' num2str(indices.animals(anIdx)) 'delayloc' num2str(delayLoc) '_alloutcomes'];
                    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
                end

                %plot the traces as distributions for different parts of track
                indivRShort = indivR(:,allFullBins); indivLShort = indivL(:,allFullBins);
                figure(varIdx+100); hold on;
                set(gcf,'units','normalized','position',[0 0 0.8 0.8]); hold on;
                maxRight = max(max(indivRShort)); minRight = min(min(indivRShort));
                maxLeft = max(max(indivLShort)); minLeft = min(min(indivLShort));
                maxAll = max([maxRight maxLeft]); minAll = min([minRight minLeft]);
                if varIdx == 5
                    if maxAll > deg2rad(60); maxAll = deg2rad(60); end;
                    if minAll < deg2rad(-60); minAll = deg2rad(-60); end;
                end
                edges = linspace(minAll, maxAll, 60); 
                histBins = [1:2:numel(allFullBins)]; 
                if numel(histBins)*2 > numel(allFullBins); histBins = histBins(1:end-1); end
                %edges = -1.5:0.1:1.5;
                for posIdx = 1:numel(histBins)
                    if ~isnan(maxRight) %if there are no real values
                        viewAnglesDistR = histcounts(indivRShort(:,histBins(posIdx):histBins(posIdx)+1), edges);
                        viewAnglesDistRNorm = viewAnglesDistR/nansum(viewAnglesDistR);
                    else
                        viewAnglesDistRNorm = nan;
                    end
                    if ~isnan(maxLeft)
                        viewAnglesDistL = histcounts(indivLShort(:,histBins(posIdx):histBins(posIdx)+1), edges);
                        viewAnglesDistLNorm = viewAnglesDistL/nansum(viewAnglesDistL);
                    else
                        viewAnglesDistLNorm = nan;
                    end
                    if outIdx == 3
                        if ~isnan(maxRight) || ~isnan(maxLeft)
                            indivAllShort = [indivRShort; indivLShort];
                            viewAnglesDistAll =  histcounts(indivAllShort(:,histBins(posIdx):histBins(posIdx)+1), edges);
                            viewAnglesDistAllNorm = viewAnglesDistAll/nansum(viewAnglesDistAll);
                        end
                    end

                    %plot correct trials
                    plotNums = [1:3:numel(allFullBins)*2; 2:3:numel(allFullBins)*2+1; 3:3:numel(allFullBins)*2+2]'; %make some extra just in case
                    ax2(posIdx) = subplot(numel(histBins),3,plotNums(numel(histBins)-posIdx+1,outIdx)); hold on;
                    if outIdx ~= 3 %if not all trials, separate left and right
                        if sum(~isnan(viewAnglesDistRNorm))
                            h1 = histogram('BinCounts', viewAnglesDistRNorm, 'BinEdges', rad2deg(edges));
                            h1.FaceAlpha = 0.2; 
                            h1.FaceColor = [0 1 0]; 
                        end
                        if sum(~isnan(viewAnglesDistLNorm))
                            h2 = histogram('BinCounts', viewAnglesDistLNorm, 'BinEdges', rad2deg(edges));
                            h2.FaceAlpha = 0.2;
                            h2.FaceColor = [1 0 1];
                        end
                    else %if it is all trials, combine the distributions
                        if sum(~isnan(viewAnglesDistAllNorm))
                            h1 = histogram('BinCounts', viewAnglesDistAllNorm, 'BinEdges', rad2deg(edges));
                            h1.FaceAlpha = 0.2; 
                            h1.FaceColor = [0 0 0]; 
                        end
                    end
                    if posIdx ~= 1; set(ax2(posIdx),'xticklabel',[]); end;
                    if posIdx == numel(histBins); title([varNames{varIdx} 'distributions on' trialOutcome{outIdx} 'Trials']); end;
                    if mod(posIdx,2); ylabel(['y pos -' num2str(round(nanmean(binsToUse(histBins(posIdx):histBins(posIdx)+1))))]); end;
                    set(gca,'tickdir','out')
                end
                sgtitle(['S' num2str(indices.animals(anIdx)) ' trajectory dist on ' trackName]);
                filename = [savedfiguresdir varNames{varIdx} 'trajectories_' trackName '_S' num2str(indices.animals(anIdx)) 'delayloc' num2str(delayLoc) '_alloutcomesdists'];
                saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
               
            end
        end
    end
    close all;
end
