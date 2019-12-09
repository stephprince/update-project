function output = plotHistByPosition(hists2plotConcat, trialtypes, metrics, animal, track, dirs);

fnames = fieldnames(hists2plotConcat);
trialOutcome = {'correct','incorrect'}; trialTurn = {'right','left'}; trialPlottype = {'X','Y'};

for metricIdx = 1:size(fnames)
  for axIdx = 1:2
    for outIdx = 1:2

      %get averages to plot on left and right trials
      metric2plot = metrics{metricIdx};
      fnames = fieldnames(output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}));
      for fieldIdx = 1:size(fnames)
        for turnIdx = 1:2
          data2plot.([fnames{fieldIdx} trialTurn{turnIdx}]) = output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).(fnames{fieldIdx});
        end
      end
      data2plot.numTrialsTotal = data2plot.numTrialsRight + numTrialsLeft;

      %plot the averages
      figure; hold on;
      ciplot(data2plot.avgRight-data2plot.semRight, data2plot.avgRight+data2plot.semRight,data2plot.posBinsRight,'r'); alpha(0.3);
      ciplot(data2plot.avgLeft-data2plot.semLeft, data2plot.avgLeft+data2plot.semLeft,data2plot.posBinsLeft,'b'); alpha(0.3);
      plot(data2plot.posBinsRight, data2plot.avgRight, 'r-', 'LineWidth', 2);
      plot(data2plot.posBinsLeft, data2plot.avgLeft, 'b-', 'LineWidth', 2);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(metric2plot); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - ' metric2plot ' - ' trialOutcome{outIdx} 'n=' num2str(data2plot.numTrialsTotal) 'right = red, left = blue']);
      filename = [dirs.behaviorfigdir metric2plot '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx}];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

      %plot the individual traces
      figure; hold on;
      indivTracesRight = output.(metric2plot).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).all';
      indivTracesLeft = output.(metric2plot).(trialPlottype{axIdx}).left.(trialOutcome{outIdx}).all';
      posBins2Plot = output.(metric2plot).(trialPlottype{axIdx}).right.(trialOutcome{outIdx}).posBinsRaw;
      plot(indivTracesRight, posBins2Plot,'r'); alpha(0.3);
      plot(indivTracesLeft, posBins2Plot,'b'); alpha(0.3);
      xlabel([trialPlottype{axIdx} ' Position']); ylabel(metric2plot); set(gca,'tickdir','out');
      title(['S' num2str(animal) ' performance on ' track ' track - ' metric2plot ' - ' trialOutcome{outIdx} 'n=' num2str(data2plot.numTrialsTotal) 'right = red, left = blue']);
      filename = [dirs.behaviorfigdir metric2plot '_' track  '_S' num2str(animal) '_pos' trialPlottype{axIdx} '_' trialOutcome{outIdx} '_indivtrials'];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

      %plot the traces as distributions for different parts of the track


    end
  end
end
