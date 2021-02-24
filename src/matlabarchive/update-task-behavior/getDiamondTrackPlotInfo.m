function plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs)

  %get session and trial intervals from structure
  plotInfo.sessInfo = trackdata.sessInfo;
  plotInfo.numSessions = plotInfo.sessInfo(end,4); %last value for sessions counter
  plotInfo.numTrials = sum(trackdata.numTrialsAll);
  plotInfo.daySessionInfo = plotInfo.sessInfo(logical(diff(plotInfo.sessInfo(:,4))),[2,4]); %matrix of sessions for each day
  plotInfo.dayIntervalsByTrial = [[1; find(diff(trackdata.sessInfo(:,2)))+1], [find(diff(trackdata.sessInfo(:,2))); length(trackdata.sessInfo(:,2))]];
  plotInfo.dayIntervalsBySession = [[1; find(diff(plotInfo.daySessionInfo(:,1)))+1], [find(diff(plotInfo.daySessionInfo(:,1))); length(plotInfo.daySessionInfo(:,1))]];

  %make background fill structures to indicate day/sessions/trial blocks
  plotInfo.backgroundInfoDays = plotInfo.dayIntervalsBySession(1:2:end,:)+[-0.5 0.5];
  plotInfo.backgroundInfoDays = [plotInfo.backgroundInfoDays, fliplr(plotInfo.backgroundInfoDays)];
  plotInfo.fillInfoDays = repmat([0 0 1.01 1.01],size(plotInfo.backgroundInfoDays,1),1);
  plotInfo.backgroundInfoTrials = plotInfo.dayIntervalsByTrial(1:2:end,:)+[-0.5 0.5];
  plotInfo.backgroundInfoTrials = [plotInfo.backgroundInfoTrials, fliplr(plotInfo.backgroundInfoTrials)];
  plotInfo.fillInfoTrials = repmat([0 0 1.01 1.01],size(plotInfo.backgroundInfoTrials,1),1);

  %define color mappings
  plotInfo.trialSubtype = {'All','Left','Right','Same','Alt'};
  plotInfo.colorSubtype = 'kbrmg';
  plotInfo.trialPairs = {'Left','Right'; 'Same','Alt'};
