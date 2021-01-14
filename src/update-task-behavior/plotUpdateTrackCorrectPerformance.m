function output = plotUpdateTaskCorrectPerformance(trackdata, animal, track, dirs)
%SP 190925 this function plots correct performance of animals with
%different slices of data
%outputs figure data for combining across animals

if ~isempty(trackdata.sessInfo)
    %% initialize variables
    %get session and trial intervals from structure
    plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
    trackInfo = trackdata.sessInfo;

    %% plot percent correct trials for each session
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numRewards');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'perCorrect');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'perIncorrect');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numCorrect');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numTrials');

    %% plot percent correct trials in trial blocks
    plotDiamondTrackMetricsByTrial(trackdata,animal,track,dirs,plotInfo,'perCorrect');
    plotDiamondTrackMetricsByTrial(trackdata,animal,track,dirs,plotInfo,'perIncorrect');

    %% plot percent correct trials in sliding window
    plotDiamondTrackMetricsByTrial_SlidingWindow(trackdata,animal,track,dirs,plotInfo,'perCorrect');
    plotDiamondTrackMetricsByTrial_SlidingWindow(trackdata,animal,track,dirs,plotInfo,'perIncorrect');
end

end
