function plotDiamondTrackMetricsByPosition(trackdata, animal, track, dirs);

metrics2plot = {'velocRot','velocTrans','position','viewAngle'};

if ~isempty(trackdata.sessInfo)

  % get plot info for plotting
  plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
  trialtypes.correctTrials = find(trackdata.sessOutcomesAll == 1);
  trialtypes.incorrectTrials = find(trackdata.sessOutcomesAll == 0);
  trialtypes.rightTrials = find(cell2mat(cellfun(@(x) ismember(2, x), trackdata.currentZone, 'UniformOutput',0)));
  trialtypes.leftTrials = find(cell2mat(cellfun(@(x) ismember(1, x), trackdata.currentZone, 'UniformOutput',0)));

  %clean the data to get rid of teleportation events for plotting
  trackDataClean = cleanTeleportEvents(trackdata);

  %get histogram of position for the vector and apply the indices to other metrics
  for metricIdx = 1:size(metrics2plot,1)
    hists2plot.([metrics2plot{metricIdx} 'Hists']) = calcHistByPosition(trackdataClean,metrics2plot{metricIdx});
  end

  % compile all the data and clean it up
  hists2plotConcat = concatHistByPosition(hists2plot, trialtypes, metrics2plot);

  % plot averages and individual traces
  plotHistByPosition(hists2plotConcat, trialtypes, metrics2plot, animal, track, dirs);
  
end
