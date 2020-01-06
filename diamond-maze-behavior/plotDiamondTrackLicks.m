function plotDiamondTrackLicks(trackdata, animal, track, dirs, params);
%SP 191122

if ~isempty(trackdata.sessInfo)

  % get plot info for plotting
  plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
  correctTrials = find(trackdata.sessOutcomesAll == 1);
  incorrectTrials = find(trackdata.sessOutcomesAll == 0);
  eastTrials = find(cell2mat(cellfun(@(x) ismember(2, x), trackdata.currentZone, 'UniformOutput',0))); %need to check which value is right vs left
  westTrials = find(cell2mat(cellfun(@(x) ismember(1, x), trackdata.currentZone, 'UniformOutput',0)));
  northTrials = find(trackdata.startLocAll == 1);
  southTrials = find(trackdata.startLocAll == 3);
  rightTrials = sort(([intersect(southTrials, eastTrials); intersect(northTrials,westTrials)]));
  leftTrials = sort([intersect(southTrials, westTrials); intersect(northTrials,eastTrials)]);

  % get lick data from set time around reward
  timeWindow = 300; %looks at 300 samples before and after (3*0.01s = 3 secs)
  lickBinSize = 20; %bin licks 20 samples before and after
  licksAroundRewardTemp = [];

  %get edges for plotting and pooling of data
  plotedges = [-timeWindow+0.5:lickBinSize:timeWindow-0.5]*params.constSampRateTime;
  binedges = -timeWindow:lickBinSize:timeWindow;
  rawedges = -timeWindow+0.5:timeWindow-0.5;
  [counts, idx] = histc(rawedges',binedges);

  for trialIdx = 1:size(trackdata.time,1)
    rewardZoneInds = find(ismember(trackdata.currentZone{trialIdx},[1,2])); %find when enters reward zone
    if isempty(rewardZoneInds) %if animal never entered a rewarding/nonrewarding end zone
      licksAroundRewardTemp = [licksAroundRewardTemp; nan(1,timeWindow/lickBinSize*2+1)];
    else
      enteredZoneInd = rewardZoneInds(1);  %first time the animal enters the reward zone
      rewardInds = find(diff(trackdata.numRewards{trialIdx})); %find any rewards given (sometimes give extra before or after)
      rewardTimeDiffFromZone = rewardInds - enteredZoneInd; %how far away were rewards from entering the time zone

      if ~isempty(rewardInds) && sum(rewardTimeDiffFromZone < -5) %that is, a reward occurred before the animal entered the reward zone (with some time wiggle room)
        licksAroundRewardTemp = [licksAroundRewardTemp; nan(1,timeWindow/lickBinSize*2+1)];
      else
        % NOTE - had to change the code below because there are no rewards on incorrect trials so couldn't compare incorrect and correct
        % NOTE - should include something where don't count if I rewarded before they entered the zone
        % closestReward = min(rewardTimeDiffFromZone); %find the closest reward (should be the one delivered by the task)
        % closestRewardInd = (rewardTimeDiffFromZone == min(rewardTimeDiffFromZone));
        % lickWindow = (rewardInds(closestRewardInd) - timeWindow):(rewardInds(closestRewardInd) + timeWindow); %look at time window where the lick happened
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
  end

  %calculate averages
  licksAroundReward = licksAroundRewardTemp(:,1:end-1); %get rid of extra bin
  licksAroundRewardAvg = nanmean(licksAroundReward);
  licksAroundRewardSem = nanstd(licksAroundReward)/sqrt(size(licksAroundReward,1));
  licksAroundRewardAvgCorrect = nanmean(licksAroundReward(correctTrials,:));
  licksAroundRewardSemCorrect = nanstd(licksAroundReward(correctTrials,:))/sqrt(size(licksAroundReward(correctTrials,:),1));
  licksAroundRewardAvgIncorrect = nanmean(licksAroundReward(incorrectTrials,:));
  licksAroundRewardSemIncorrect = nanstd(licksAroundReward(incorrectTrials,:))/sqrt(size(licksAroundReward(incorrectTrials,:),1));
  licksAroundRewardAvgCorrectRight = nanmean(licksAroundReward(intersect(correctTrials,rightTrials),:));
  licksAroundRewardSemCorrectRight = nanstd(licksAroundReward(intersect(correctTrials,rightTrials),:))/sqrt(size(licksAroundReward(intersect(correctTrials,rightTrials),:),1));
  licksAroundRewardAvgCorrectLeft = nanmean(licksAroundReward(intersect(correctTrials,leftTrials),:));
  licksAroundRewardSemCorrectLeft = nanstd(licksAroundReward(intersect(correctTrials,leftTrials),:))/sqrt(size(licksAroundReward(intersect(correctTrials,leftTrials),:),1));
  licksAroundRewardAvgIncorrectRight = nanmean(licksAroundReward(intersect(incorrectTrials,rightTrials),:));
  licksAroundRewardSemIncorrectRight = nanstd(licksAroundReward(intersect(incorrectTrials,rightTrials),:))/sqrt(size(licksAroundReward(intersect(correctTrials,rightTrials),:),1));
  licksAroundRewardAvgIncorrectLeft = nanmean(licksAroundReward(intersect(incorrectTrials,leftTrials),:));
  licksAroundRewardSemIncorrectLeft = nanstd(licksAroundReward(intersect(incorrectTrials,leftTrials),:))/sqrt(size(licksAroundReward(intersect(correctTrials,leftTrials),:),1));

  %% plot averages and individual traces
  figure; hold on;
  plot([0 0],[0 max(licksAroundRewardAvg*1.1)], 'k--')
  plot(plotedges, licksAroundRewardAvg, 'k-', 'LineWidth', 2);
  ciplot(licksAroundRewardAvg-licksAroundRewardSem, licksAroundRewardAvg+licksAroundRewardSem,plotedges,'k-');
  alpha(0.5);
  xlabel('Time (s)'); ylabel('Average Licks'); set(gca,'tickdir','out'); ylim([0 max(licksAroundRewardAvg*1.1)]);
  title(['S' num2str(animal) ' performance on ' track ' track - licksAroundRewardZone']);
  filename = [dirs.behaviorfigdir 'licksAroundRewardZone_' track  '_S' num2str(animal)];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %correct vs. incorrect trials
  figure; hold on;
  plot([0 0],[0 max(licksAroundRewardAvgCorrect*1.1)], 'k--');
  plot(plotedges, licksAroundRewardAvgIncorrect, 'r-', 'LineWidth', 2);
  ciplot(licksAroundRewardAvgIncorrect-licksAroundRewardSemIncorrect, licksAroundRewardAvgIncorrect+licksAroundRewardSemIncorrect,plotedges,'r-');
  alpha(0.5);
  plot(plotedges, licksAroundRewardAvgCorrect, 'g-', 'LineWidth', 2);
  ciplot(licksAroundRewardAvgCorrect-licksAroundRewardSemCorrect, licksAroundRewardAvgCorrect+licksAroundRewardSemCorrect,plotedges,'g-');
  alpha(0.5); hold on;
  xlabel('Time (s)'); ylabel('Average Licks'); set(gca,'tickdir','out'); ylim([0 max(licksAroundRewardAvgCorrect*1.1)]);
  title(['S' num2str(animal) ' performance on ' track ' track - licksAroundRewardZone - Correct vs. Incorrect']);
  filename = [dirs.behaviorfigdir 'licksAroundRewardZone_' track  '_S' num2str(animal) '_CorrectvIncorrect'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %left vs. right trials
  figure; hold on;
  plot([0 0],[0 max([licksAroundRewardAvgCorrectRight, licksAroundRewardAvgCorrectLeft]*1.1)], 'k--');
  plot(plotedges, licksAroundRewardAvgCorrectRight, 'r-', 'LineWidth', 2);
  ciplot(licksAroundRewardAvgCorrectRight-licksAroundRewardSemCorrectRight, licksAroundRewardAvgCorrectRight+licksAroundRewardSemCorrectRight,plotedges,'r-');
  alpha(0.5);
  plot(plotedges, licksAroundRewardAvgIncorrectRight, 'r--', 'LineWidth', 2);
  ciplot(licksAroundRewardAvgIncorrectRight-licksAroundRewardSemIncorrectRight, licksAroundRewardAvgIncorrectRight+licksAroundRewardSemIncorrectRight,plotedges,'r--');
  alpha(0.3);
  plot(plotedges, licksAroundRewardAvgCorrectLeft, 'b-', 'LineWidth', 2);
  ciplot(licksAroundRewardAvgCorrectLeft-licksAroundRewardSemCorrectLeft, licksAroundRewardAvgCorrectLeft+licksAroundRewardSemCorrectLeft,plotedges,'b-');
  alpha(0.5); hold on;
  plot(plotedges, licksAroundRewardAvgIncorrectLeft, 'b--', 'LineWidth', 2);
  ciplot(licksAroundRewardAvgIncorrectLeft-licksAroundRewardSemIncorrectLeft, licksAroundRewardAvgIncorrectLeft+licksAroundRewardSemIncorrectLeft,plotedges,'b--');
  alpha(0.5); hold on;
  xlabel('Time (s)'); ylabel('Average Licks'); set(gca,'tickdir','out'); ylim([0 max(licksAroundRewardAvgCorrectRight*1.1)]);
  title(['S' num2str(animal) ' performance on ' track ' track - licksAroundRewardZone - Left vs. Right']);
  filename = [dirs.behaviorfigdir 'licksAroundRewardZone_' track  '_S' num2str(animal) '_LeftvsRight'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

end
