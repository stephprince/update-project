function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,params,behaviordata);

animals = unique(indices.behaviorindex(:,1));
trainingoptions = {'linear','shaping','choice1side','choice2side','choice1side_short','continuousalt'};
statsoutput = [];

%% concatenate data across sessions
allsessdata = concatDiamondMazeSessions(animals, indices, behaviordata, trainingoptions);

% %% plot all behavior metrics (don't really care about linear track performance so track options start at 2)
for anIdx = 1:length(animals)
    for trackIdx = length(trainingoptions)
        % plot metrics as a function of position throughout the trial
        %plotDiamondTrackMetricsByPosition(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);

        % plot percent correct
        plotDiamondTrackCorrectPerformance(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);

        % plot licking (as a function of distance from reward and trials since correct)
        plotDiamondTrackLicks(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs, params);

        % plot average trial duration for correct, failed, incorrect trials for each session
        %plotDiamondTrackBoxplotDist(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}),animals(anIdx),trainingoptions{trackIdx},dirs,'dur');

        % plot continuous alt performance
        %if strcmp(trainingoptions{trackIdx},'continuousalt')
        %  plotContinuousAltPerformance(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);
        %end
    end
end
close all;

%% plot all behavior trial metrics in 50 trial blocks
blockSizesAll = [20 30 40 50 60 70];
subplotSizesAll = [100 200 300];
for anIdx = 1:length(animals) 
  for subplotSizeIdx = 2:numel(subplotSizesAll)
      subplotSize = subplotSizesAll(subplotSizeIdx); %size for sliding window and size for subplot figs
      for blockIdx = 4%1:numel(blockSizesAll)
          trackdata = allsessdata(animals(anIdx)).continuousalt;
          blockSize = blockSizesAll(blockIdx);
          numTrials = numel(trackdata.sessOutcomesAll);
          numSubplots = ceil(numTrials/subplotSize);
          subplotStartEnds = [[1; [subplotSize+1:subplotSize:subplotSize*numSubplots]'], [subplotSize:subplotSize:subplotSize*numSubplots]'];
          %concat perconnect data
          perCorrectPerBlock = []; %this will be the metric to plot
          for iterIdx = 1:(numTrials-blockSize+1) %get moving averages for each blocksizes
            trialBins = iterIdx:iterIdx+blockSize-1;
            trialBlockOutcomes = trackdata.sessOutcomesAll(trialBins);
            perCorrectTemp = sum(trialBlockOutcomes == 1)/length(trialBlockOutcomes); %can't just average bc -1 indicates incomplete trial
            perCorrectPerBlock = [perCorrectPerBlock; perCorrectTemp];
          end
          perCorrectPerSubplot = nan(numSubplots,subplotSize,1);
          for subplotIdx = 1:numSubplots     %then get vals for each subplot
            if subplotStartEnds(subplotIdx,2) > numel(perCorrectPerBlock)
              trialBins = subplotStartEnds(subplotIdx,1):numel(perCorrectPerBlock); %if up to the last trials, just average whatever is there
            else
              trialBins = subplotStartEnds(subplotIdx,1):subplotStartEnds(subplotIdx,2);
            end
            perCorrectPerSubplot(subplotIdx,1:numel(perCorrectPerBlock(trialBins))) = perCorrectPerBlock(trialBins);
          end


          %concat licks pre reward data
          timeWindow = 1000; %looks at 300 samples before and after (3*0.01s = 3 secs)
          lickBinSize = 50; %bin licks 20 samples before and after
          binedges = -timeWindow:lickBinSize:timeWindow; rawedges = -timeWindow+0.5:timeWindow-0.5;
          [counts, idx] = histc(rawedges',binedges);
          correctLicksAvg = nan(1,timeWindow/lickBinSize*2); incorrectLicksAvg = nan(1,timeWindow/lickBinSize*2);
          correctLicksSEM = nan(1,timeWindow/lickBinSize*2); incorrectLicksSEM = nan(1,timeWindow/lickBinSize*2); %this will be the metric to plot
          for subplotIdx = 1:numSubplots
            if subplotStartEnds(subplotIdx,2) > numTrials
              trialBins = subplotStartEnds(subplotIdx,1):numTrials; %if up to the last trials, just average whatever is there
            else
              trialBins = subplotStartEnds(subplotIdx,1):subplotStartEnds(subplotIdx,2);
            end

            licksAroundRewardTemp = [];
            for trialIdx = trialBins(1):trialBins(end)
              rewardZoneInds = find(ismember(trackdata.currentZone{trialIdx},[1,2])); %find when enters reward zone
              if isempty(rewardZoneInds) %if animal never entered a rewarding/nonrewarding end zone
                licksAroundRewardTemp = [licksAroundRewardTemp; nan(1,timeWindow/lickBinSize*2+1)];
              else
                enteredZoneInd = rewardZoneInds(1);  %first time the animal enters the reward zone
                lickWindow = (enteredZoneInd - timeWindow):(enteredZoneInd + timeWindow); %look at time window where the lick happened
                if max(lickWindow) > size(trackdata.numLicks{trialIdx},2) || min(lickWindow) < 1%if the trial ends before the time window has closed
                  if max(lickWindow) > size(trackdata.numLicks{trialIdx},2) && min(lickWindow) < 1
                    lickDataTemp = diff(trackdata.numLicks{trialIdx});
                    extraSamples2AddFirst = nan(1,abs(1-min(lickWindow)));
                    extraSamples2AddLast = nan(1,abs(size(trackdata.numLicks{trialIdx},2) - max(lickWindow)));
                  elseif max(lickWindow) > size(trackdata.numLicks{trialIdx},2)
                    lickDataTemp = diff(trackdata.numLicks{trialIdx}(lickWindow(1):end));
                    extraSamples2Add = nan(1,(timeWindow*2)-length(lickDataTemp));
                    lickData = [lickDataTemp, extraSamples2Add];
                  elseif min(lickWindow) < 1
                    lickDataTemp = diff(trackdata.numLicks{trialIdx}(1:lickWindow(end)));
                    extraSamples2Add = nan(1,(timeWindow*2)-length(lickDataTemp));
                    lickData = [extraSamples2Add, lickDataTemp];
                  end
                else
                  lickData = diff(trackdata.numLicks{trialIdx}(lickWindow)); %find lick times
                end

                lickDataBinned = accumarray(idx,lickData',[size(binedges,2),1]); %get lick counts in larger bins
                licksAroundRewardTemp = [licksAroundRewardTemp; lickDataBinned']; %find these licks and concatenate
              end
            end

            %calculate averages
            licksAroundReward = licksAroundRewardTemp(:,1:end-1); %get rid of extra bin
            correctTrialsInBlock = find(trackdata.sessOutcomesAll(trialBins) == 1);
            incorrectTrialsInBlock = find(trackdata.sessOutcomesAll(trialBins) == 0);
            correctLicksAvg(subplotIdx,:) = nanmean(licksAroundReward(correctTrialsInBlock,:));
            correctLicksSEM(subplotIdx,:) = nanstd(licksAroundReward(correctTrialsInBlock,:))/sqrt(size(licksAroundReward(correctTrialsInBlock,:),1));
            incorrectLicksAvg(subplotIdx,:) = nanmean(licksAroundReward(incorrectTrialsInBlock,:));
            incorrectLicksSEM(subplotIdx,:)= nanstd(licksAroundReward(incorrectTrialsInBlock,:))/sqrt(size(licksAroundReward(incorrectTrialsInBlock,:),1));
          end

          %concatenate trajectories and position related values through the track
          metrics2plot = {'positionX','positionY','viewAngle'};
          trackDataClean = cleanTeleportEvents(trackdata); %clean the data to get rid of teleportation events for plotting
          for metricIdx = 1:size(metrics2plot,2)
            hists2plot.([metrics2plot{metricIdx} 'Hists']) = calcHistByPosition(trackDataClean,metrics2plot{metricIdx}); %bin data so able to plot across multiple trials
          end

          numPosYBins = numel(hists2plot.positionXHists.posYbins);
          trajectoriesRightCorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesRightCorrectSEM = nan(numSubplots,numPosYBins,1);
          trajectoriesLeftCorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesLeftCorrectSEM = nan(numSubplots,numPosYBins,1);
          trajectoriesRightIncorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesRightIncorrectSEM = nan(numSubplots,numPosYBins,1);
          trajectoriesLeftIncorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesLeftIncorrectSEM = nan(numSubplots,numPosYBins,1);
          for subplotIdx = 1:numSubplots
            %get trial types within this subplot group
            if subplotStartEnds(subplotIdx,2) > numTrials %get trial bins to look at
                trialBins = subplotStartEnds(subplotIdx,1):numTrials; %if up to the last trials, just average whatever is there
            else
                trialBins = subplotStartEnds(subplotIdx,1):subplotStartEnds(subplotIdx,2);
            end
            correctTrialsInBlock = find(trackdata.sessOutcomesAll(trialBins) == 1);
            incorrectTrialsInBlock = find(trackdata.sessOutcomesAll(trialBins) == 0);
            rightTrials = find(trackdata.startLocAll(trialBins) == 1); %this only works right now bc trial types start on opposite sides, will have to change in future
            leftTrials = find(trackdata.startLocAll(trialBins) == 3);

            %grab and average different types of trajectories
            xPosInYBins = hists2plot.positionXHists.histY(trialBins,:);
            trajectoriesRightCorrect{subplotIdx} = xPosInYBins(intersect(correctTrialsInBlock,rightTrials),:);
            trajectoriesLeftCorrect{subplotIdx} = xPosInYBins(intersect(correctTrialsInBlock,leftTrials),:);
            trajectoriesRightIncorrect{subplotIdx} = xPosInYBins(intersect(incorrectTrialsInBlock,rightTrials),:);
            trajectoriesLeftIncorrect{subplotIdx} = xPosInYBins(intersect(incorrectTrialsInBlock,leftTrials),:);
            trajectoriesRightCorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(correctTrialsInBlock,rightTrials),:)); trajectoriesRightCorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(correctTrialsInBlock,rightTrials),:))/sqrt(numel(intersect(correctTrialsInBlock,rightTrials)));
            trajectoriesLeftCorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(correctTrialsInBlock,leftTrials),:)); trajectoriesLeftCorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(correctTrialsInBlock,leftTrials),:))/sqrt(numel(intersect(correctTrialsInBlock,leftTrials)));
            trajectoriesRightIncorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(incorrectTrialsInBlock,rightTrials),:)); trajectoriesRightIncorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(incorrectTrialsInBlock,rightTrials),:))/sqrt(numel(intersect(incorrectTrialsInBlock,rightTrials)));
            trajectoriesLeftIncorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(incorrectTrialsInBlock,leftTrials),:)); trajectoriesLeftIncorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(incorrectTrialsInBlock,leftTrials),:))/sqrt(numel(intersect(incorrectTrialsInBlock,leftTrials)));

            %get view angles over y position
            viewAngleInYBins = hists2plot.viewAngleHists.histY(trialBins,:);
            viewAnglesRightCorrect{subplotIdx} = viewAngleInYBins(intersect(correctTrialsInBlock,rightTrials),:);
            viewAnglesLeftCorrect{subplotIdx} = viewAngleInYBins(intersect(correctTrialsInBlock,leftTrials),:);
            viewAnglesRightIncorrect{subplotIdx} = viewAngleInYBins(intersect(incorrectTrialsInBlock,rightTrials),:);
            viewAnglesLeftIncorrect{subplotIdx} = viewAngleInYBins(intersect(incorrectTrialsInBlock,leftTrials),:);
          end

          %plot all the data
          figure('units','normalized','position',[0 0 0.8 0.8]); hold on;
          for subplotIdx = 1:numSubplots

            %percent correct data
            ax1(subplotIdx) = subplot(5,numSubplots,subplotIdx); hold on;
            plot(1:subplotSize, repmat([0.25 0.5 0.75],subplotSize,1), 'Color', [0 0 0 0.1])
            plot(1:subplotSize, perCorrectPerSubplot(subplotIdx,:), 'm-', 'LineWidth', 2);
            xlabel('Trials (moving average)'); ylim([0 1.01]);
            title(['Trials ' num2str(subplotStartEnds(subplotIdx,1)) ' to ' num2str(subplotStartEnds(subplotIdx,2))])
            if subplotIdx == 1; ylabel('Proportion correct'); else; set(ax1(subplotIdx),'yticklabel',[]); end;

            %trajectories on the track
            ax2(subplotIdx) = subplot(5,numSubplots,numSubplots+subplotIdx); hold on;
            ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins);
            if subplotIdx == round(numSubplots/2); title('Trajectories - Correct Trials'); end
            p1 = plot(ybins(1:end-1), fliplr(-1*trajectoriesRightCorrect{subplotIdx}(:,1:end-1)), 'Color', [1 0 0 0.2], 'LineWidth', 1); %have to flip right now and multiple by negative bc actually on opposite sides of track
            p2 = plot(ybins(1:end-1), trajectoriesLeftCorrect{subplotIdx}(:,1:end-1), 'Color', [0 0 1 0.2], 'LineWidth', 1);
            xlim([0 140]); ylim([-35 35]); set(ax2(subplotIdx),'xticklabel',[]);
            if subplotIdx == 1; ylabel('X position'); else; set(ax2(subplotIdx),'yticklabel',[]); end;

            %view angles on the track
            ax3(subplotIdx) = subplot(5,numSubplots,numSubplots*2+subplotIdx); hold on;
            ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins);
            p1 = plot(ybins(1:end-1), rad2deg(-1*(fliplr(viewAnglesRightCorrect{subplotIdx}(:,1:end-1)-pi))), 'Color', [1 0 0 0.2], 'LineWidth', 1); %have to flip right now and multiple by negative bc actually on opposite sides of track
            p2 = plot(ybins(1:end-1), rad2deg(-1*viewAnglesLeftCorrect{subplotIdx}(:,1:end-1)), 'Color', [0 0 1 0.2], 'LineWidth', 1);
            xlim([0 140]); ylim([-90 90]);
            if subplotIdx == 1; ylabel('View angle (rad)'); else; set(ax3(subplotIdx),'yticklabel',[]); end; xlabel('Y position');

            %plot view angle distributions on the track over time
            ax4(subplotIdx) = subplot(5,numSubplots,numSubplots*3+subplotIdx); hold on;
            ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins);
            rightTrialsViewAnglesTemp = -1*(fliplr(viewAnglesRightCorrect{subplotIdx}(:,1:end-1)-pi));
            leftTrialsViewAnglesTemp = -1*viewAnglesLeftCorrect{subplotIdx};
            rightTrialsViewAngles = rightTrialsViewAnglesTemp(:,4:22); leftTrialsViewAngles = leftTrialsViewAnglesTemp(:,4:22); %get rid of first bins where all 0 vals
            ybinsShort = ybins(4:22);
            rightTrialsViewAngles(:,ybinsShort > 105) = []; leftTrialsViewAngles(:,ybinsShort > 105) = [];
            ybinsShort(ybinsShort > 105) = [];
            edges = -1.5:0.1:1.5;
            rightTrialsViewAnglesDist = histcounts(rightTrialsViewAngles(:,7:end), edges); %second half of track
            leftTrialsViewAnglesDist = histcounts(leftTrialsViewAngles(:,7:end), edges);
            rightTrialsViewAnglesDist = rightTrialsViewAnglesDist/nansum(rightTrialsViewAnglesDist);
            leftTrialsViewAnglesDist = leftTrialsViewAnglesDist/nansum(leftTrialsViewAnglesDist);
            h1 = histogram('BinCounts', rightTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
            h2 = histogram('BinCounts', leftTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
            h1.FaceAlpha = 0.2; h2.FaceAlpha = 0.2; xlim([-90 90]);
            h1.FaceColor = [1 0 0]; h2.FaceColor = [0 0 1];
            if subplotIdx == round(numSubplots/2); title('View Angle Distribution in Second Half of Track - Correct Trials'); end
            if subplotIdx == 1; ylabel('Proportion of all position bins'); else; set(ax4(subplotIdx),'yticklabel',[]); end; xlabel('View angle (rad)');

            %licks around reward zone
            ax5(subplotIdx) = subplot(5,numSubplots,numSubplots*4+subplotIdx); hold on;
            plotedges = [-timeWindow+0.5:lickBinSize:timeWindow-0.5]*params.constSampRateTime; %get edges for plotting and pooling of data
            plot([0 0],[0 max(max(correctLicksAvg*1.1))], 'k--');
            plot(plotedges, correctLicksAvg(subplotIdx,:), 'g-', 'LineWidth', 2);
            ciplot(correctLicksAvg(subplotIdx,:)-correctLicksSEM(subplotIdx,:), correctLicksAvg(subplotIdx,:)+correctLicksSEM(subplotIdx,:),plotedges,'g-');
            plot(plotedges, incorrectLicksAvg(subplotIdx,:), 'k-', 'LineWidth', 2);
            ciplot(incorrectLicksAvg(subplotIdx,:)-incorrectLicksSEM(subplotIdx,:), incorrectLicksAvg(subplotIdx,:)+incorrectLicksSEM(subplotIdx,:),plotedges,'k-');
            alpha(0.5); xlabel('Time (s)'); ylim([0 max(max(correctLicksAvg*1.1))]);
            if subplotIdx == 1; ylabel('Lick rate (Hz)'); else; set(ax5(subplotIdx),'yticklabel',[]); end;
            if subplotIdx == round(numSubplots/2); title('Licks around reward'); end;
          end
          linkaxes(ax1,'xy'); linkaxes(ax2,'xy'); linkaxes(ax3,'xy'); linkaxes(ax4,'xy'); linkaxes(ax5,'xy');

          %save the figure
          sgtitle(['S' num2str(animals(anIdx)) ' performance']);
          filename = [dirs.behaviorfigdir 'sessPerformanceAll_S' num2str(animals(anIdx)) '_blocksize' num2str(blockSize) '_subplotsize' num2str(subplotSize) '_allmetrics'];
          saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

          %plot the view angle distributions throughout the track
          figure('units','normalized','position',[0 0 0.4 0.8]); hold on;
          viewAnglesRightCorrectAll = []; viewAnglesLeftCorrectAll = [];
          viewAnglesRightIncorrectAll = []; viewAnglesLeftIncorrectAll = [];
          for subplotIdx = 1:numSubplots
              viewAnglesRightCorrectAll = [viewAnglesRightCorrectAll; viewAnglesRightCorrect{subplotIdx}];
              viewAnglesLeftCorrectAll = [viewAnglesLeftCorrectAll; viewAnglesLeftCorrect{subplotIdx}];
              viewAnglesRightIncorrectAll = [viewAnglesRightIncorrectAll; viewAnglesRightIncorrect{subplotIdx}];
              viewAnglesLeftIncorrectAll = [viewAnglesLeftIncorrectAll; viewAnglesLeftIncorrect{subplotIdx}];
          end
          ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins); ybinsShort = ybins(4:22);
          viewAnglesRightCorrectAllTemp = -1*(fliplr(viewAnglesRightCorrectAll(:,1:end-1)-pi));
          viewAnglesLeftCorrectAllTemp = -1*viewAnglesLeftCorrectAll;
          viewAnglesRightCorrectAllTemp = viewAnglesRightCorrectAllTemp(:,4:22);
          viewAnglesLeftCorrectAllTemp = viewAnglesLeftCorrectAllTemp(:,4:22);
          viewAnglesRightIncorrectAllTemp = -1*(fliplr(viewAnglesRightIncorrectAll(:,1:end-1)-pi));
          viewAnglesLeftIncorrectAllTemp = -1*viewAnglesLeftIncorrectAll;
          viewAnglesRightIncorrectAllTemp = viewAnglesRightIncorrectAllTemp(:,4:22);
          viewAnglesLeftIncorrectAllTemp = viewAnglesLeftIncorrectAllTemp(:,4:22);

          edges = -2:0.1:2; histBins = [1:2:ybinsShort]; plotNums = [1:2:ybinsShort; 2:2:ybinsShort+1]';
          for posIdx = 1:numel(histBins)
              rightCorrectTrialsViewAnglesDist = histcounts(viewAnglesRightCorrectAllTemp(:,histBins(posIdx):histBins(posIdx)+1), edges); %second half of track
              leftCorrectTrialsViewAnglesDist = histcounts(viewAnglesLeftCorrectAllTemp(:,histBins(posIdx):histBins(posIdx)+1), edges);
              rightCorrectTrialsViewAnglesDist = rightCorrectTrialsViewAnglesDist/nansum(rightCorrectTrialsViewAnglesDist);
              leftCorrectTrialsViewAnglesDist = leftCorrectTrialsViewAnglesDist/nansum(leftCorrectTrialsViewAnglesDist);
              rightIncorrectTrialsViewAnglesDist = histcounts(viewAnglesRightIncorrectAllTemp(:,histBins(posIdx):histBins(posIdx)+1), edges); %second half of track
              leftIncorrectTrialsViewAnglesDist = histcounts(viewAnglesLeftIncorrectAllTemp(:,histBins(posIdx):histBins(posIdx)+1), edges);
              rightIncorrectTrialsViewAnglesDist = rightIncorrectTrialsViewAnglesDist/nansum(rightIncorrectTrialsViewAnglesDist);
              leftIncorrectTrialsViewAnglesDist = leftIncorrectTrialsViewAnglesDist/nansum(leftIncorrectTrialsViewAnglesDist);

              %plot correct trials
              ax10(posIdx) = subplot(numel(histBins),2,plotNums(numel(histBins)-posIdx+1,1)); hold on;
              h1 = histogram('BinCounts', rightCorrectTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
              h2 = histogram('BinCounts', leftCorrectTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
              h1.FaceAlpha = 0.2; h2.FaceAlpha = 0.2; xlim([-90 90]);
              h1.FaceColor = [1 0 0]; h2.FaceColor = [0 0 1];
              if posIdx ~= 1; set(ax10(posIdx),'xticklabel',[]); end;
              if posIdx == numel(histBins); title('View Angle Distributions on Correct Trials'); end;
              if mod(posIdx,2); ylabel(['y pos -' num2str(round(nanmean(ybinsShort(histBins(posIdx):histBins(posIdx)+1))))]); end;
              set(gca,'tickdir','out')

              %plot incorrect trials
              ax11(posIdx) = subplot(numel(histBins),2,plotNums(numel(histBins)-posIdx+1,2)); hold on;
              h1 = histogram('BinCounts', rightIncorrectTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
              h2 = histogram('BinCounts', leftIncorrectTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
              h1.FaceAlpha = 0.2; h2.FaceAlpha = 0.2; xlim([-90 90]);
              h1.FaceColor = [1 0 0]; h2.FaceColor = [0 0 1];
              if posIdx ~= 1; set(ax11(posIdx),'xticklabel',[]); end;
              if posIdx == numel(histBins); title('View Angle Distributions on Incorrect Trials'); end;
              if mod(posIdx,2); ylabel(['y pos -' num2str(round(nanmean(ybinsShort(histBins(posIdx):histBins(posIdx)+1))))]); end;
              set(gca,'tickdir','out')
          end
          if subplotIdx == round(numSubplots/2); title('View Angle Distribution in Second Half of Track - Correct Trials'); end
          filename = [dirs.behaviorfigdir 'viewAngleDistVsYPos_S' num2str(animals(anIdx))];
          saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
      end
  end
  close all;
end
