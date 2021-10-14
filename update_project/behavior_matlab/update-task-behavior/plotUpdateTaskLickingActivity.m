function plotUpdateTaskLickingActivity(trackdata, indices, dirs, params)
%SP 191122

savedfiguresdir = [dirs.behaviorfigdir 'lickingActivity/'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end

% get lick data from set time around reward
params.lickTimeWindow = 4/params.constSampRateTime; %looks at 4 sec before and after
params.lickBinSize = 0.5/params.constSampRateTime; %bin licks in half a second bins

%get edges for plotting and pooling of data
plotedges = [-params.lickTimeWindow+0.5:params.lickBinSize:params.lickTimeWindow-0.5]*params.constSampRateTime;
binedges = -params.lickTimeWindow:params.lickBinSize:params.lickTimeWindow;
rawedges = -params.lickTimeWindow+0.5:params.lickTimeWindow-0.5;
[counts, idx] = histc(rawedges',binedges);

%loop through animals to getlicking activity around the reward zone
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    for worldIdx = 1:4 %skip linear bc don't care about percent correct
        %compile all trials for each track across days
        trialRows = cellfun(@(x) find(x.trialWorld == worldIdx), animaldata.trialTable,'UniformOutput',0);
        trialdata = [];
        for trialIdx = 1:numel(trialRows)
            trialdata = [trialdata; animaldata.trialTable{trialIdx,:}(trialRows{trialIdx},:)];
        end
        
        %get licking activity across all trials
        licksAroundRewardTemp = nan(size(trialdata,1),size(binedges,2));
        licksAroundRewardRaw{anIdx}{worldIdx} = nan(size(trialdata,1),params.lickTimeWindow*2);
        for trialIdx = 1:size(trialdata,1)
            virmendata = trialdata(trialIdx,:).resampledTrialData{1};    
            choiceMadeInds = find(virmendata.taskState == params.taskStatesMap('choiceMade')); %find when animal makes choice
            if isempty(choiceMadeInds)
                lickDataBinned = nan(size(binedges,2),1);
            else
                enteredZoneInd = choiceMadeInds(1); %when first entered reward zone
                lickWindow = (enteredZoneInd - params.lickTimeWindow):(enteredZoneInd + params.lickTimeWindow); %look at time window where the lick happened
                windowStart = max([min(lickWindow) 1]); diffFromStart = abs(min(lickWindow) - windowStart);
                windowEnd = min([max(lickWindow) size(virmendata,1)]); diffFromEnd = max(lickWindow) - windowEnd;
                lickData = [nan(diffFromStart,1); diff(virmendata.numLicks(windowStart:windowEnd)); nan(diffFromEnd,1)];
                lickDataBinned = accumarray(idx,lickData',[size(binedges,2),1]); %get lick counts in larger bins
                lickDataBinned(lickDataBinned<0) = 0;
            end
            licksAroundRewardRaw{anIdx}{worldIdx}(trialIdx,:) = lickData';
            licksAroundRewardTemp(trialIdx,:) = lickDataBinned'; %add these licks
        end
        
        %get averages across trials
        trialOutcomes{anIdx}{worldIdx} = trialdata.trialOutcomes;
        licksAroundReward{anIdx}{worldIdx} = licksAroundRewardTemp(:,1:end-1); %get rid of extra bin
        licksAroundRewardAllCorrect{anIdx}{worldIdx} = licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 1,:);
        licksAroundRewardAllIncorrect{anIdx}{worldIdx} = licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 0,:);
        licksAroundRewardAvgCorrect{anIdx}{worldIdx} = nanmean(licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 1,:),1);
        licksAroundRewardSemCorrect{anIdx}{worldIdx} = nanstd(licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 1,:),[],1)/sqrt(size(licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 1,:),1));
        licksAroundRewardAvgIncorrect{anIdx}{worldIdx} = nanmean(licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 0,:),1);
        licksAroundRewardSemIncorrect{anIdx}{worldIdx} = nanstd(licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 0,:),[],1)/sqrt(size(licksAroundReward{anIdx}{worldIdx}(trialdata.trialOutcomes == 0,:),1));
    end
end

%% plot lick data
for anIdx = 1:numel(indices.animals)
    figure('units','normalized','outerposition',[0 0 0.9 0.9]); hold on;
    for worldIdx = 1:4
        %plot lick averages for correct and incorrect
        subplot(3,4,worldIdx); hold on;
        plot([0 0],[0 max(licksAroundRewardAvgCorrect{anIdx}{worldIdx}*1.1)], 'k--');
        if sum(~isnan(licksAroundRewardAvgIncorrect{anIdx}{worldIdx}))
            plot(plotedges, licksAroundRewardAvgIncorrect{anIdx}{worldIdx}, 'k-', 'LineWidth', 2);
            ciplot(licksAroundRewardAvgIncorrect{anIdx}{worldIdx}-licksAroundRewardSemIncorrect{anIdx}{worldIdx}, licksAroundRewardAvgIncorrect{anIdx}{worldIdx}+licksAroundRewardSemIncorrect{anIdx}{worldIdx},plotedges,'k-');
            alpha(0.5);
        end
        if sum(~isnan(licksAroundRewardAvgCorrect{anIdx}{worldIdx}))
            plot(plotedges, licksAroundRewardAvgCorrect{anIdx}{worldIdx}, 'm-', 'LineWidth', 2);
            ciplot(licksAroundRewardAvgCorrect{anIdx}{worldIdx}-licksAroundRewardSemCorrect{anIdx}{worldIdx}, licksAroundRewardAvgCorrect{anIdx}{worldIdx}+licksAroundRewardSemCorrect{anIdx}{worldIdx},plotedges,'m-');
            alpha(0.5); hold on;
            %ylim([0 max(licksAroundRewardAvgCorrect{anIdx}{worldIdx}*1.1)]);
        end
        trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;
        trackName = trackTypeKeySet{trackTypeValueSet{worldIdx}};
        title(trackName); ylabel('Average Licks'); 
        
        %plot lick rasters for correct and incorrect
%        subplot(2,4,worldIdx+4); hold on;        
%         for i = 1:size(licksAroundRewardRaw{anIdx}{worldIdx},1)
%             [ypoints, xpoints] = find(licksAroundRewardRaw{anIdx}{worldIdx}(i,:) == 1);
%             times = (-params.lickTimeWindow:params.lickTimeWindow)*params.constSampRateTime;
%             if trialOutcomes{anIdx}{worldIdx}(i) == 1
%                 plot(times(xpoints),ypoints*i,'.m','MarkerSize',0.35);
%             else
%                 plot(times(xpoints),ypoints*i,'.k','MarkerSize',0.35);
%             end
%         end
%         plot([0 0],[0 size(licksAroundRewardRaw{anIdx}{worldIdx},1)], 'k--')
        firstHalf = 1:numel(plotedges)/2;
        [~,sortInd] = sort(sum(licksAroundRewardAllCorrect{anIdx}{worldIdx}(:,firstHalf),2));
        licksCorrectSorted = licksAroundRewardAllCorrect{anIdx}{worldIdx}(sortInd,:);
        [~,sortInd] = sort(sum(licksAroundRewardAllIncorrect{anIdx}{worldIdx}(:,firstHalf),2));
        licksIncorrectSorted = licksAroundRewardAllIncorrect{anIdx}{worldIdx}(sortInd,:);
        ax(1) = subplot(3,4,worldIdx+4); hold on;        
        cmapCorrect = cbrewer('seq','RdPu',100);
        imagesc('CData',licksCorrectSorted, 'XData', plotedges, 'YData',1:size(licksCorrectSorted,2), [0 10])
        xlim([min(plotedges) max(plotedges)])
        %ylim([0 size(licksCorrectSorted,2)])

        ax(2) = subplot(3,4,worldIdx+8); hold on;     
        cmapIncorrect = cbrewer('seq','Greys',100);
        imagesc('CData',licksIncorrectSorted, 'XData', plotedges, 'YData',1:size(licksIncorrectSorted,2), [0 10])
        xlabel('Time (s)'); ylabel('Trials'); %ylim([0 max([size(licksAroundRewardRaw{anIdx}{worldIdx},1) 1])]) ;
        xlim([min(plotedges) max(plotedges)])
        %ylim([0 size(licksIncorrectSorted,2)])
        
        colormap(ax(1), cmapCorrect)
        colormap(ax(2),cmapIncorrect)
        if worldIdx == 1
            colorbar(ax(1))
        	colorbar(ax(2))
        end
        clear licksCorrectSorted licksIncorrectSorted
    end
    sgtitle(['S' num2str(indices.animals(anIdx)) ' licksAroundRewardZone - Correct vs. Incorrect']);
    filename = [savedfiguresdir 'licksAroundRewardZone_S' num2str(indices.animals(anIdx)) '_CorrectvIncorrect'];
    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end

%% plot licking data for all animals
figure('units','normalized','outerposition',[0 0 0.9 0.9]); hold on;
for worldIdx = 1:4
    %concatenate data
    licksAroundRewardAllCorrectAllAnimals{worldIdx} = []; licksAroundRewardAllIncorrectAllAnimals{worldIdx} = [];
    for anIdx = 1:numel(indices.animals)
        licksAroundRewardAllCorrectAllAnimals{worldIdx} = [licksAroundRewardAllCorrectAllAnimals{worldIdx}; licksAroundRewardAllCorrect{anIdx}{worldIdx}];
        licksAroundRewardAllIncorrectAllAnimals{worldIdx} = [licksAroundRewardAllIncorrectAllAnimals{worldIdx}; licksAroundRewardAllIncorrect{anIdx}{worldIdx}];
    end
    licksAroundRewardAvgCorrectAllAnimals{worldIdx} = nanmean(licksAroundRewardAllCorrectAllAnimals{worldIdx},1);
    licksAroundRewardSemCorrectAllAnimals{worldIdx} = nanstd(licksAroundRewardAllCorrectAllAnimals{worldIdx},[],1)/sqrt(size(licksAroundRewardAllCorrectAllAnimals{worldIdx},1));
    licksAroundRewardAvgIncorrectAllAnimals{worldIdx} = nanmean(licksAroundRewardAllIncorrectAllAnimals{worldIdx},1);
    licksAroundRewardSemIncorrectAllAnimals{worldIdx} = nanstd(licksAroundRewardAllIncorrectAllAnimals{worldIdx},[],1)/sqrt(size(licksAroundRewardAllIncorrectAllAnimals{worldIdx},1));
    
    %plot lick averages for correct and incorrect
    subplot(3,4,worldIdx); hold on;
    plot([0 0],[0 max(licksAroundRewardAvgCorrectAllAnimals{worldIdx}*1.1)], 'k--');
    if sum(~isnan(licksAroundRewardAvgIncorrectAllAnimals{worldIdx}))
        plot(plotedges, licksAroundRewardAvgIncorrectAllAnimals{worldIdx}, 'k-', 'LineWidth', 2);
        ciplot(licksAroundRewardAvgIncorrectAllAnimals{worldIdx}-licksAroundRewardSemIncorrectAllAnimals{worldIdx}, licksAroundRewardAvgIncorrectAllAnimals{worldIdx}+licksAroundRewardSemIncorrectAllAnimals{worldIdx},plotedges,'k-');
        alpha(0.5);
    end
    if sum(~isnan(licksAroundRewardAvgCorrectAllAnimals{worldIdx}))
        plot(plotedges, licksAroundRewardAvgCorrectAllAnimals{worldIdx}, 'm-', 'LineWidth', 2);
        ciplot(licksAroundRewardAvgCorrectAllAnimals{worldIdx}-licksAroundRewardSemCorrectAllAnimals{worldIdx}, licksAroundRewardAvgCorrectAllAnimals{worldIdx}+licksAroundRewardSemCorrectAllAnimals{worldIdx},plotedges,'m-');
        alpha(0.5); hold on;
    end
    trackTypeKeySet = params.trackTypeMap.keys; trackTypeValueSet = params.trackTypeMap.values;
    trackName = trackTypeKeySet{trackTypeValueSet{worldIdx}};
    title(trackName); ylabel('Average Licks');
    
    %plot lick rasters for correct and incorrect
  
    firstHalf = 1:numel(plotedges)/2;
    [~,sortInd] = sort(sum(licksAroundRewardAllCorrectAllAnimals{worldIdx}(:,firstHalf),2));
    licksCorrectSorted = licksAroundRewardAllCorrectAllAnimals{worldIdx}(sortInd,:);
    [~,sortInd] = sort(sum(licksAroundRewardAllIncorrectAllAnimals{worldIdx}(:,firstHalf),2));
    licksIncorrectSorted = licksAroundRewardAllIncorrectAllAnimals{worldIdx}(sortInd,:);
    ax(1) = subplot(3,4,worldIdx+4); hold on;
    cmapCorrect = cbrewer('seq','RdPu',100);
    imagesc('CData',licksCorrectSorted, 'XData', plotedges, 'YData',1:size(licksCorrectSorted,2), [0 10])
    xlim([min(plotedges) max(plotedges)])
    
    ax(2) = subplot(3,4,worldIdx+8); hold on;
    cmapIncorrect = cbrewer('seq','Greys',100);
    imagesc('CData',licksIncorrectSorted, 'XData', plotedges, 'YData',1:size(licksIncorrectSorted,2), [0 10])
    xlabel('Time (s)'); ylabel('Trials');
    xlim([min(plotedges) max(plotedges)])
    
    colormap(ax(1), cmapCorrect)
    colormap(ax(2),cmapIncorrect)
    if worldIdx == 1
        colorbar(ax(1))
        colorbar(ax(2))
    end
    clear licksCorrectSorted licksIncorrectSorted
end
sgtitle(['All animals - licksAroundRewardZone - Correct vs. Incorrect']);
filename = [savedfiguresdir 'licksAroundRewardZone_AllAnimals_CorrectvIncorrect'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end
