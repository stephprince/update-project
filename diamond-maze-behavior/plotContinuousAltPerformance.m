function output = plotContinuousAltPerformance(trackdata, animal, track, dirs)
%SP 190925 this function plots correct performance of animals with
%different slices of data
%outputs figure data for combining across animals

if ~isempty(trackdata.sessInfo)
    %% initialize variables
    %get session and trial intervals from structure
    trialBlockSize = 5;
    trackInfo = trackdata.sessInfo;
    plotInfo.numSessions = trackInfo(end,4); %last value for sessions counter
    plotInfo.numTrials = sum(trackdata.numTrialsAll);
    plotInfo.daySessionInfo = trackInfo(logical(diff(trackInfo(:,4))),[2,4]); %matrix of sessions for each day
    plotInfo.dayIntervalsByTrial = [[1; find(diff(trackdata.sessInfo(:,2)))+1], [find(diff(trackdata.sessInfo(:,2))); length(trackdata.sessInfo(:,2))]];
    plotInfo.dayIntervalsBySession = [[1; find(diff(plotInfo.daySessionInfo(:,1)))+1], [find(diff(plotInfo.daySessionInfo(:,1))); length(plotInfo.daySessionInfo(:,1))]];
    
    %make background fill structures to indicate day/sessions/trial blocks
    plotInfo.backgroundInfoDays = plotInfo.dayIntervalsBySession(1:2:end,:)+[-0.5 0.5];
    plotInfo.backgroundInfoDays = [plotInfo.backgroundInfoDays, fliplr(plotInfo.backgroundInfoDays)];
    plotInfo.fillInfoDays = repmat([0 0 1.01 1.01],size(plotInfo.backgroundInfoDays,1),1);
        
    %% plot percent correct trials for each session
    trackdata.trialsSinceCorrectAll
    % plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'perCorrect');
    % plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'perIncorrect');
    % plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numCorrect');
    % plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numTrials');
    
end
    
end