function plotViewAngle(trackdata, animal, track, dirs);
%SP 191122

if ~isempty(trackdata.sessInfo) && ~isempty(trackdata.viewAngle)

  % get plot info for plotting
  plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
  correctTrials = find(trackdata.sessOutcomesAll == 1);
  incorrectTrials = find(trackdata.sessOutcomesAll == 0);
  rightTrials = cell2mat(cellfun(@(x) ismember(1, x), trackdata.currentZone, 'UniformOutput',0));
  leftTrials = cell2mat(cellfun(@(x) ismember(2, x), trackdata.currentZone, 'UniformOutput',0));

  %clean the data to get rid of teleportation events for plotting
  minY = []; minX = []; maxX = []; maxY = [];
  for trialIdx = 1:size(trackdata.time,1)
    %find when animal is in main part of track and large position jumps
    altTrackInds = find(trackdata.currentWorld{trialIdx} == 1);
    teleportEvents = find(abs(diff(trackdata.positionY{trialIdx})) > 10);
    if isempty(teleportEvents);
      bins2keep = altTrackInds;
    else
      bins2throw = find(ismember(altTrackInds,teleportEvents));
      bins2keep = altTrackInds(1):altTrackInds(bins2throw);
    end

    %clean the vectors for each trial
    positionXClean{trialIdx} = trackdata.positionX{trialIdx}(bins2keep);
    positionYClean{trialIdx} = trackdata.positionY{trialIdx}(bins2keep);
    viewAngleClean{trialIdx} = trackdata.viewAngle{trialIdx}(bins2keep);

    %get bins for averaging
    minY = [minY; min(positionYClean{trialIdx})];
    maxY = [maxY; max(positionYClean{trialIdx})];
    minX = [minX; min(positionXClean{trialIdx})];
    maxX = [maxX; min(positionXClean{trialIdx})];
  end

  %get averages for right and left turns by binning by position
  minYpos = min(minY); minXpos = min(minX);
  maxYpos = max(maxY); maxXpos = max(maxX);
  posYbins = linspace(minYpos,maxYpos,50);
  posXbins = linspace(minXpos,maxXpos,50);

  %get histogram of position for the vector and apply the indices to view Angle
  for trialIdx = 1:size(trackdata.time,1)
    [n edg binsUsedX] = histcounts(positionXClean{trialIdx},posXbins);
    [n edg binsUsedY] = histcounts(positionYClean{trialIdx},posYbins);
    angleHistXTemp = []; angleHistYTemp = [];
    for binIdx = 1:size(binsUsedX,2)
      data2addX = nan(size(binsUsedX));
      angleHistX(trialIdx,binsUsedX) = viewAngleClean{trialIdx};
      angleHistY(trialIdx,binsUsedY) = viewAngleClean{trialIdx};
    end
  end
  angleHistX(angleHistX == 0) = nan;
  angleHistY(angleHistY == 0) = nan;
  angleHistXCorrectRight = angleHistX(union(correctTrials,rightTrials),:);
  angleHistXIncorrectRight = angleHistX(union(incorrectTrials,rightTrials),:);
  angleHistYCorrectRight = angleHistY(union(correctTrials,rightTrials),:);
  angleHistYIncorrectRight = angleHistY(union(incorrectTrials,rightTrials),:);
  angleHistXCorrectRightAvg = nanmean(angleHistXCorrectRight);
  angleHistYCorrectRightAvg = nanmean(angleHistYCorrectRight);
  angleHistXCorrectRightStd = nanstd(angleHistXCorrectRight);
  angleHistYCorrectRightStd = nanstd(angleHistYCorrectRight);

  %% plot averages and individual traces
  figure; hold on;
  plot(posYbins(1:end-1), angleHistYCorrectRightAvg, 'g-', 'LineWidth', 2);
  ciplot(angleHistYCorrectRightAvg-angleHistYCorrectRightStd, angleHistYCorrectRightAvg+angleHistYCorrectRightStd,posYbins(1:end-1),'g-');
  alpha(0.5);
  xlabel('Y-position'); ylabel('View Angle ');
  set(gca,'tickdir','out'); ylim([0 1.01*max(m)]);
  title(['S' num2str(animal) ' performance on ' track ' track - viewAnglePosY']);
  filename = [dirs.behaviorfigdir 'viewAnglePosY_' track  '_S' num2str(animal)];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end
