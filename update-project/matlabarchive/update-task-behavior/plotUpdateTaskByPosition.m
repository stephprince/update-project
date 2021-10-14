function plotUpdateTaskByPosition(trackdata, animal, track, dirs);

metrics2plot = {'velocRot','velocTrans','positionX','positionY','viewAngle'};

if ~isempty(trackdata.sessInfo)

  % get plot info for plotting
  plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
  trialtypes.correctTrials = find(trackdata.sessOutcomesAll == 1);
  trialtypes.incorrectTrials = find(trackdata.sessOutcomesAll == 0);
  eastTrials = find(cell2mat(cellfun(@(x) ismember(2, x), trackdata.currentZone, 'UniformOutput',0))); %need to check which value is right vs left
  westTrials = find(cell2mat(cellfun(@(x) ismember(1, x), trackdata.currentZone, 'UniformOutput',0)));
  northTrials = find(trackdata.startLocAll == 1);
  southTrials = find(trackdata.startLocAll == 3);
  trialtypes.rightTrials = sort(([intersect(southTrials, eastTrials); intersect(northTrials,westTrials)]));
  trialtypes.leftTrials = sort([intersect(southTrials, westTrials); intersect(northTrials,eastTrials)]);

  %clean the data to get rid of teleportation events for plotting
  trackDataClean = cleanTeleportEvents(trackdata);

  %get histogram of position for the vector and apply the indices to other metrics
  for metricIdx = 1:size(metrics2plot,2)
    hists2plot.([metrics2plot{metricIdx} 'Hists']) = calcHistByPosition(trackDataClean,metrics2plot{metricIdx});
  end

  % compile all the data and clean it up
  hists2plotConcat = concatHistByPosition(hists2plot, trialtypes, metrics2plot);

  % plot averages and individual traces
  plotHistByPosition(hists2plotConcat, trialtypes, metrics2plot, animal, track, dirs);

end
