function plotViewAngle(trackdata, animal, track, dirs);
%SP 191122

if ~isempty(trackdata.sessInfo) && ~isempty(trackdata.viewAngle)

  % get plot info for plotting
  plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
  correctTrials = find(trackdata.sessOutcomesAll == 1);
  incorrectTrials = find(trackdata.sessOutcomesAll == 0);
  rightTrials = find(cell2mat(cellfun(@(x) ismember(2, x), trackdata.currentZone, 'UniformOutput',0)));
  leftTrials = find(cell2mat(cellfun(@(x) ismember(1, x), trackdata.currentZone, 'UniformOutput',0)));

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
    maxX = [maxX; max(positionXClean{trialIdx})];
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
    uniquebinsX = unique(binsUsedX); uniquebinsY = unique(binsUsedY);
    angleHistX(trialIdx,:) = nan(size(posXbins));
    angleHistY(trialIdx,:) = nan(size(posYbins));
    for binIdx = 1:size(uniquebinsX,2)
      bins2avg = find(binsUsedX == uniquebinsX(binIdx));
      viewAngleAvg = nanmean(viewAngleClean{trialIdx}(bins2avg));
      angleHistX(trialIdx,uniquebinsX(binIdx)) = viewAngleAvg;
    end
    for binIdx = 1:size(uniquebinsY,2)
      bins2avg = find(binsUsedY == uniquebinsY(binIdx));
      viewAngleAvg = nanmean(viewAngleClean{trialIdx}(bins2avg));
      angleHistY(trialIdx,uniquebinsY(binIdx)) = viewAngleAvg;
    end
  end

  %% compile all the data and clean it up
  trialOutcome = {'correct','incorrect'};
  trialTurn = {'right','left'}; trialPlottype = {'X','Y'};
  for axIdx = 1:2
    for turnIdx = 1:2
      for outIdx = 1:2
        %get average, sem for the different trial types
        trialtype1 = eval([trialOutcome{outIdx} 'Trials']);
        trialtype2 = eval([trialTurn{turnIdx} 'Trials']);
        angleHist = eval(['angleHist' trialPlottype{axIdx}]);
        angleHistSubtype = angleHist(intersect(trialtype1, trialtype2),:);
        angleHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).all = angleHistSubtype;
        posBinsRaw = eval(['pos' trialPlottype{axIdx} 'bins']);
        posBins = posBinsRaw;

        %clean the data from nans
        tempmean = nanmean(angleHistSubtype);
        tempsem = nanstd(angleHistSubtype)/sqrt(size(angleHistSubtype,1));
        numtrials = size(angleHistSubtype,1);
        posBins(isnan(tempmean)) = [];
        tempmean(isnan(tempmean)) = []; tempsem(isnan(tempsem)) = [];
        angleHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).avg =  tempmean;
        angleHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).sem = tempsem;
        angleHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posbins = posBins;
        angleHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posbinsraw = posBinsRaw;
        angleHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).numtrials = numtrials;
      end
    end
  end

  %% plot averages and correct vs. incorrect trials
  for axIdx = 1:2
    for outIdx = 1:2
      %get averages to plot
      avg2plotRight = angleHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).avg;
      avg2plotLeft = angleHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).avg;
      sem2plotRight = angleHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).sem;
      sem2plotLeft = angleHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).sem;
      pos2plotRight = angleHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posbins;
      pos2plotLeft = angleHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).posbins;
      numTrialsRight = angleHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).numtrials;
      numTrialsLeft = angleHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).numtrials;
      numTrials = numTrialsRight + numTrialsLeft;

      %plot averages
      if strcmp(trialPlottype{axIdx},'X');
        avg2plotRight = abs(avg2plotRight);
        sem2plotRight = abs(sem2plotRight);
      end;
      figure; hold on;
      ciplot(avg2plotRight-sem2plotRight, avg2plotRight+sem2plotRight,pos2plotRight,'r'); alpha(0.3);
      ciplot(avg2plotLeft-sem2plotLeft, avg2plotLeft+sem2plotLeft,pos2plotLeft,'b'); alpha(0.3);
      plot(pos2plotRight, avg2plotRight, 'r-', 'LineWidth', 2);
      plot(pos2plotLeft, avg2plotLeft, 'b-', 'LineWidth', 2);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel('View Angle'); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - viewAngle - ' trialOutcome{outIdx} 'n=' num2str(numTrials)]);
      filename = [dirs.behaviorfigdir 'viewAngle_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx}];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

      %plot individual traces
      figure; hold on;
      plot(angleHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).all',angleHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posbinsraw,'r'); alpha(0.3);
      plot(angleHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).all',angleHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).posbinsraw,'b'); alpha(0.3);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel('View Angle'); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - viewAngle - ' trialOutcome{outIdx} 'n=' num2str(numTrials)]);
      filename = [dirs.behaviorfigdir 'viewAngle_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_indivtrials'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end
  end
  close all;
end
