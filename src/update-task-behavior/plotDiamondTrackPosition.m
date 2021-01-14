function plotDiamondTrackPosition(trackdata, animal, track, dirs);

if ~isempty(trackdata.sessInfo)
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
    velocTransClean{trialIdx} = trackdata.velocTrans{trialIdx}(bins2keep);
    velocRotClean{trialIdx} = trackdata.velocRot{trialIdx}(bins2keep);

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
    velocRotHistX(trialIdx,:) = nan(size(posXbins));
    velocTransHistX(trialIdx,:) = nan(size(posXbins));
    velocRotHistY(trialIdx,:) = nan(size(posYbins));
    velocTransHistY(trialIdx,:) = nan(size(posYbins));
    posHistX(trialIdx,:) = nan(size(posXbins));
    posHistY(trialIdx,:) = nan(size(posYbins));
    for binIdx = 1:size(uniquebinsX,2)
      bins2avg = find(binsUsedX == uniquebinsX(binIdx));
      velocTransAvg = nanmean(velocTransClean{trialIdx}(bins2avg));
      velocTransHistX(trialIdx,uniquebinsX(binIdx)) = velocTransAvg;
      velocRotAvg = nanmean(velocRotClean{trialIdx}(bins2avg));
      velocRotHistX(trialIdx,uniquebinsX(binIdx)) = velocRotAvg;
      posAvg = nanmean(positionYClean{trialIdx}(bins2avg));
      posHistX(trialIdx,uniquebinsX(binIdx)) = posAvg;
    end
    for binIdx = 1:size(uniquebinsY,2)
      bins2avg = find(binsUsedY == uniquebinsY(binIdx));
      velocTransAvg = nanmean(velocTransClean{trialIdx}(bins2avg));
      velocTransHistY(trialIdx,uniquebinsY(binIdx)) = velocTransAvg;
      velocRotAvg = nanmean(velocRotClean{trialIdx}(bins2avg));
      velocRotHistY(trialIdx,uniquebinsY(binIdx)) = velocRotAvg;
      posAvg = nanmean(positionXClean{trialIdx}(bins2avg));
      posHistY(trialIdx,uniquebinsY(binIdx)) = posAvg;
    end
  end

  %% compile all the data and clean it up
  %velocity
  trialOutcome = {'correct','incorrect'}; trialTurn = {'right','left'};
  velocType = {'Trans','Rot'}; trialPlottype = {'X','Y'};
  for velocIdx = 1:2
    for axIdx = 1:2
      for turnIdx = 1:2
        for outIdx = 1:2
          %get average, sem for the different trial types
          trialtype1 = eval([trialOutcome{outIdx} 'Trials']);
          trialtype2 = eval([trialTurn{turnIdx} 'Trials']);
          velocHist = eval(['veloc' velocType{velocIdx} 'Hist' trialPlottype{axIdx}]);
          velocHistSubtype = velocHist(intersect(trialtype1, trialtype2),:);
          velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).all = velocHistSubtype;
          posBins = eval(['pos' trialPlottype{axIdx} 'bins']);

          %clean the data from nans for veloc
          tempmean = nanmean(velocHistSubtype);
          tempsem = nanstd(velocHistSubtype)/sqrt(size(velocHistSubtype,1));
          numtrials = size(velocHistSubtype,1);
          posBins(isnan(tempmean)) = [];
          tempmean(isnan(tempmean)) = []; tempsem(isnan(tempsem)) = [];
          velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).avg =  tempmean;
          velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).sem = tempsem;
          velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posbins = posBins;
          velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).numtrials = numtrials;

        end
      end
    end
  end

  %position
  for axIdx = 1:2
    for turnIdx = 1:2
      for outIdx = 1:2
        %get average, sem for the different trial types
        trialtype1 = eval([trialOutcome{outIdx} 'Trials']);
        trialtype2 = eval([trialTurn{turnIdx} 'Trials']);
        posHist = eval(['posHist' trialPlottype{axIdx}]);
        posHistSubtype = posHist(intersect(trialtype1, trialtype2),:);
        posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).all = posHistSubtype;
        posBinsRaw = eval(['pos' trialPlottype{axIdx} 'bins']);
        posBins = posBinsRaw;

        %clean the data from nans for veloc
        tempmean = nanmean(posHistSubtype);
        tempsem = nanstd(posHistSubtype)/sqrt(size(posHistSubtype,1));
        numtrials = size(posHistSubtype,1);
        posBins(isnan(tempmean)) = [];
        tempmean(isnan(tempmean)) = []; tempsem(isnan(tempsem)) = [];
        posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).avg =  tempmean;
        posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).sem = tempsem;
        posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posbins = posBins;
        posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posbinsraw = posBinsRaw;
        posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).numtrials = numtrials;
      end
    end
  end

  %% plot averages and correct vs. incorrect trials for velocity
  for velocIdx = 1:2
    for axIdx = 1:2
      %plot right vs. left for correct vs. incorrect
      for outIdx = 1:2
        %get averages to plot
        avg2plotRight = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).avg;
        avg2plotLeft = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).avg;
        sem2plotRight = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).sem;
        sem2plotLeft = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).sem;
        pos2plotRight = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posbins;
        pos2plotLeft = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).posbins;
        numTrialsRight = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).numtrials;
        numTrialsLeft = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).numtrials;
        numTrials = numTrialsRight + numTrialsLeft;

        % if strcmp(trialPlottype{axIdx},'X');
        %   avg2plotRight = abs(avg2plotRight);
        %   sem2plotRight = abs(sem2plotRight);
        % end;

        figure; hold on;
        ciplot(avg2plotRight-sem2plotRight, avg2plotRight+sem2plotRight,pos2plotRight,'r'); alpha(0.3);
        ciplot(avg2plotLeft-sem2plotLeft, avg2plotLeft+sem2plotLeft,pos2plotLeft,'b'); alpha(0.3);
        plot(pos2plotRight, avg2plotRight, 'r-', 'LineWidth', 2);
        plot(pos2plotLeft, avg2plotLeft, 'b-', 'LineWidth', 2);
        xlabel([trialPlottype{axIdx} ' Position']); ylabel(['Velocity' velocType{velocIdx}]); set(gca,'tickdir','out');
        title(['S' num2str(animal) ' performance on ' track ' track -' velocType{velocIdx} 'veloc - ' trialOutcome{outIdx} 'n=' num2str(numTrials)]);
        filename = [dirs.behaviorfigdir 'veloc' velocType{velocIdx} '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_rightvsleft'];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
      end

      %plot correct vs. incorrect for right vs. left trials
      for turnIdx = 1:2
        avg2plotCorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.avg;
        avg2plotIncorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.avg;
        sem2plotCorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.sem;
        sem2plotIncorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.sem;
        pos2plotCorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.posbins;
        pos2plotIncorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.posbins;
        numTrialsCorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.numtrials;
        numTrialsIncorrect = velocHistData.(velocType{velocIdx}).(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.numtrials;
        numTrials = numTrialsRight + numTrialsLeft;

        figure; hold on;
        ciplot(avg2plotCorrect-sem2plotCorrect, avg2plotCorrect+sem2plotCorrect,pos2plotCorrect,'g'); alpha(0.3);
        ciplot(avg2plotIncorrect-sem2plotIncorrect, avg2plotIncorrect+sem2plotIncorrect,pos2plotIncorrect,'r'); alpha(0.3);
        plot(pos2plotCorrect, avg2plotCorrect, 'g-', 'LineWidth', 2);
        plot(pos2plotIncorrect, avg2plotIncorrect, 'r-', 'LineWidth', 2);
        xlabel([trialPlottype{axIdx} ' Position']); ylabel(['Velocity' velocType{velocIdx}]); set(gca,'tickdir','out');
        title(['S' num2str(animal) ' performance on ' track ' track -' velocType{velocIdx} 'veloc - ' trialTurn{turnIdx} 'n=' num2str(numTrials)]);
        filename = [dirs.behaviorfigdir 'veloc' velocType{velocIdx} '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialTurn{turnIdx} '_correctvsincorrect'];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
      end
    end
  end

  %% plot averages and correct vs. incorrect trials for position
  for axIdx = 1:2
    %plot right vs. left for correct vs. incorrect
    for outIdx = 1:2
      %get averages to plot
      avg2plotRight = posHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).avg;
      avg2plotLeft = posHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).avg;
      sem2plotRight = posHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).sem;
      sem2plotLeft = posHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).sem;
      pos2plotRight = posHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posbins;
      pos2plotLeft = posHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).posbins;
      numTrialsRight = posHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).numtrials;
      numTrialsLeft = posHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).numtrials;
      numTrials = numTrialsRight + numTrialsLeft;

      % if strcmp(trialPlottype{axIdx},'X');
      %   avg2plotRight = abs(avg2plotRight);
      %   sem2plotRight = abs(sem2plotRight);
      % end;

      figure; hold on;
      ciplot(avg2plotRight-sem2plotRight, avg2plotRight+sem2plotRight,pos2plotRight,'r'); alpha(0.3);
      ciplot(avg2plotLeft-sem2plotLeft, avg2plotLeft+sem2plotLeft,pos2plotLeft,'b'); alpha(0.3);
      plot(pos2plotRight, avg2plotRight, 'r-', 'LineWidth', 2);
      plot(pos2plotLeft, avg2plotLeft, 'b-', 'LineWidth', 2);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(['Position']); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - veloc - ' trialOutcome{outIdx} 'n=' num2str(numTrials)]);
      filename = [dirs.behaviorfigdir 'position_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_rightvsleft'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

      %plot individual traces
      figure; hold on;
      plot(posHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).all',posHistData.(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posbinsraw,'r'); alpha(0.3);
      plot(posHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).all',posHistData.(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).posbinsraw,'b'); alpha(0.3);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(['Position']); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - veloc - ' trialOutcome{outIdx} 'n=' num2str(numTrials)]);
      filename = [dirs.behaviorfigdir 'position_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_rightvsleft_indivtrials'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end

    %plot correct vs. incorrect for right vs. left trials
    for turnIdx = 1:2
      avg2plotCorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.avg;
      avg2plotIncorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.avg;
      sem2plotCorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.sem;
      sem2plotIncorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.sem;
      pos2plotCorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.posbins;
      pos2plotIncorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.posbins;
      numTrialsCorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).correct.numtrials;
      numTrialsIncorrect = posHistData.(trialPlottype{axIdx}).(trialTurn{turnIdx}).incorrect.numtrials;
      numTrials = numTrialsRight + numTrialsLeft;

      figure; hold on;
      ciplot(avg2plotCorrect-sem2plotCorrect, avg2plotCorrect+sem2plotCorrect,pos2plotCorrect,'g'); alpha(0.3);
      ciplot(avg2plotIncorrect-sem2plotIncorrect, avg2plotIncorrect+sem2plotIncorrect,pos2plotIncorrect,'r'); alpha(0.3);
      plot(pos2plotCorrect, avg2plotCorrect, 'g-', 'LineWidth', 2);
      plot(pos2plotIncorrect, avg2plotIncorrect, 'r-', 'LineWidth', 2);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(['Position']); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - pos - ' trialTurn{turnIdx} 'n=' num2str(numTrials)]);
      filename = [dirs.behaviorfigdir 'veloc_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialTurn{turnIdx} '_correctvsincorrect'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end
  end
  close all;
end
