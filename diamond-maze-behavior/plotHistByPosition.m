function output = plotHistByPosition(hists2plotConcat, trialtypes, metrics, animal, track, dirs);

fnames = fieldnames(hists2plotConcat);
trialOutcome = {'correct','incorrect'}; trialTurn = {'right','left'}; trialPlottype = {'X','Y'};

for metricIdx = 1:size(fnames,1)
  for axIdx = 1:2
    for outIdx = 1:2

      %get averages to plot on left and right trials
      metric2plot = metrics{metricIdx};
      fnames = fieldnames(hists2plotConcat.(metric2plot).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}));
      for fieldIdx = 1:size(fnames)
        for turnIdx = 1:2
          data2plot.([fnames{fieldIdx} trialTurn{turnIdx}]) = hists2plotConcat.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).(fnames{fieldIdx});
        end
      end
      data2plot.numTrialsTotal = data2plot.numTrialsright + data2plot.numTrialsleft;

      %plot the averages
      figure; hold on;
      ciplot(data2plot.avgright-data2plot.semright, data2plot.avgright+data2plot.semright,data2plot.posBinsright,'r'); alpha(0.3);
      ciplot(data2plot.avgleft-data2plot.semleft, data2plot.avgleft+data2plot.semleft,data2plot.posBinsleft,'b'); alpha(0.3);
      plot(data2plot.posBinsright, data2plot.avgright, 'r-', 'LineWidth', 2);
      plot(data2plot.posBinsleft, data2plot.avgleft, 'b-', 'LineWidth', 2);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(metric2plot); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - ' metric2plot ' - ' trialOutcome{outIdx} 'n=' num2str(data2plot.numTrialsTotal) 'right = red, left = blue']);
      filename = [dirs.behaviorfigdir metric2plot '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx}];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

      %plot the individual traces
      figure; hold on;
      indivTracesRight = hists2plotConcat.(metric2plot).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).all';
      indivTracesLeft = hists2plotConcat.(metric2plot).(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).all';
      posBins2Plot = hists2plotConcat.(metric2plot).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posBinsRaw;
      plot(indivTracesRight, posBins2Plot,'r'); alpha(0.3);
      plot(indivTracesLeft, posBins2Plot,'b'); alpha(0.3);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(metric2plot); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - ' metric2plot ' - ' trialOutcome{outIdx} 'n=' num2str(data2plot.numTrialsTotal) 'right = red, left = blue']);
      filename = [dirs.behaviorfigdir metric2plot '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_indivtrials'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

      %plot the traces as distributions for different parts of the track
      numSubplots = 5; subPlotBinsize = length(posBins2Plot)/numSubplots;
      subplotIndsTemp = [1:subPlotBinsize:length(posBins2Plot)]-1;
      subplotInds = [1, subplotIndsTemp(2:end), length(posBins2Plot)];
      maxRight = max(max(indivTracesRight)); minRight = min(min(indivTracesRight));
      maxLeft = max(max(indivTracesLeft)); minLeft = min(min(indivTracesLeft));
      maxAll = max([maxRight maxLeft]); minAll = min([minRight minLeft]);
      edges = linspace(minAll, maxAll, 40);
      figure('units','normalized','outerposition',[0 0 1 1]); hold on;
      for subIdx = 1:length(subplotInds)
        subHistRight = histcounts(indivTracesRight(subplotInds(subIdx),:),edges);
        subHistsNormRight(subIdx,:) = subHistRight/nansum(subHistRight);
        subHistLeft = histcounts(indivTracesLeft(subplotInds(subIdx),:),edges);
        subHistsNormLeft(subIdx,:) = subHistLeft/nansum(subHistLeft);
        subplot(length(subplotInds),1,subIdx); hold on;
        plot(edges(1:end-1), subHistsNormRight(subIdx,:), 'r');
        plot(edges(1:end-1), subHistsNormLeft(subIdx,:), 'b');
        xlabel([trialPlottype{axIdx} ' Position']); ylabel(metric2plot); set(gca,'tickdir','out');
        title([metric2plot ' dist at ' num2str(posBins2Plot(subplotInds(subIdx))) ' ' trialPlottype{axIdx} ' Position'])
      end
      sgtitle(['S' num2str(animal) ' performance on ' track ' track - ' metric2plot ' - ' trialOutcome{outIdx} 'n=' num2str(data2plot.numTrialsTotal) 'right = red, left = blue']);
      filename = [dirs.behaviorfigdir metric2plot '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_indivtrialsdists'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end
  end
end
close all;
