function plotDiamondTrackLicks(trackdata, animal, track, dirs);
%SP 191122

if ~isempty(trackdata.sessInfo)

  % get plot info for plotting
  plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
  correctTrials = find(trackdata.sessOutcomesAll == 1);
  incorrectTrials = find(trackdata.sessOutcomesAll == 0);
  rightTrials = cell2mat(cellfun(@(x) ismember(1, x), trackdata.currentZone, 'UniformOutput',0));
  leftTrials = cell2mat(cellfun(@(x) ismember(2, x), trackdata.currentZone, 'UniformOutput',0));

  % get lick data from set time around reward
  timeWindow = 300; %looks at 300 samples before and after (3*0.01s = 3 secs)
  licksAroundReward = [];
  for trialIdx = 1:size(trackdata.time,1)
    rewardZoneInds = find(ismember(trackdata.currentZone{trialIdx},[1,2])); %find when enters reward zone
    enteredZoneInd = rewardZoneInds(1);
    rewardInds = find(diff(trackdata.numRewards{trialIdx}));
    rewardTimeDiffFromZone = abs(rewardInds - enteredZoneInd);
    closestReward = min(rewardTimeDiffFromZone);
    closestRewardInd = (rewardTimeDiffFromZone == min(rewardTimeDiffFromZone));
    if closestReward < 5
      lickWindow = (rewardInds(closestRewardInd) - timeWindow):(rewardInds(closestRewardInd) + timeWindow);
      lickData = diff(trackdata.numLicks{trialIdx}(lickWindow));
      licksAroundReward = [licksAroundReward; lickData];
    else
      licksAroundReward = [licksAroundReward; nan(1,timeWindow+1)];
    end
  end
  licksAroundRewardSum = nansum(licksAroundReward)

  %% plot averages and individual traces
  figure; hold on;
  plot(posYbins(1:end-1), angleHistYCorrectRightAvg, 'g-', 'LineWidth', 2);
  ciplot(angleHistYCorrectRightAvg-angleHistYCorrectRightStd, angleHistYCorrectRightAvg+angleHistYCorrectRightStd,posYbins(1:end-1),'g-');
  alpha(0.5);
  xlabel('Y-position'); ylabel('numLicks');
  set(gca,'tickdir','out'); ylim([0 1.01*max(m)]);
  title(['S' num2str(animal) ' performance on ' track ' track - licksAroundReward']);
  filename = [dirs.behaviorfigdir 'licksAroundReward_' track  '_S' num2str(animal)];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end
