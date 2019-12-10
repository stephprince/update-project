function output = concatHistByPosition(hists2plot,trialtypes, metrics)

fnames = fieldnames(hists2plot);
trialOutcome = {'correct','incorrect'}; trialTurn = {'right','left'}; trialPlottype = {'X','Y'};

for metricIdx = 1:size(metrics,2)
  for axIdx = 1:2
    for turnIdx = 1:2
      for outIdx = 1:2
        %get data and trial types to look at
        metric2plot = metrics{metricIdx};
        trialtype1 = trialtypes.([trialOutcome{outIdx} 'Trials']);
        trialtype2 = trialtypes.([trialTurn{turnIdx} 'Trials']);
        metricHist = hists2plot.([metric2plot 'Hists']).(['hist' trialPlottype{axIdx}]);

        %get histogram info for specific trial subtypes
        metricHistTrialType = metricHist(intersect(trialtype1, trialtype2),:);
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).all = metricHistTrialType;
        posBinsAll = hists2plot.([metric2plot 'Hists']).(['pos' trialPlottype{axIdx} 'bins']);
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posBinsAll = posBinsAll;
        posBinsRaw = posBinsAll;

        %get average, sem for the different trial types
        tempmean = nanmean(metricHistTrialType);
        tempsem = nanstd(metricHistTrialType)/sqrt(size(metricHistTrialType,1));
        numTrials = size(metricHistTrialType,1);

        %clean the data from nans and fill structure
        posBinsAll(isnan(tempmean)) = []; tempmean(isnan(tempmean)) = []; tempsem(isnan(tempsem)) = [];
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).avg =  tempmean;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).sem = tempsem;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posBinsRaw = posBinsRaw;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posBins = posBinsAll;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).numTrials = numTrials;
      end
    end
  end
end
