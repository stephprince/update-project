function statsoutput = plotUpdateTaskBehaviorMetrics(dirs,indices,params,behaviordata)

% plot percent correct
plotUpdateTaskCorrectPerformance(behaviordata, indices, dirs, params);

% plot licking activity
plotUpdateTaskLickingActivity(behaviordata, indices, dirs, params);

% plot all behavior metrics
plotUpdateTaskTrajectories(behaviordata, indices, dirs, params);

%% extra stuff
%concatenate trajectories and position related values through the track
metrics2plot = {'positionX','positionY','viewAngle'};
trackDataClean = cleanTeleportEvents(trackdata); %clean the data to get rid of teleportation events for plotting
for metricIdx = 1:size(metrics2plot,2)
    hists2plot.([metrics2plot{metricIdx} 'Hists']) = calcHistByPosition(trackDataClean,metrics2plot{metricIdx}); %bin data so able to plot across multiple trials
end

numPosYBins = numel(hists2plot.positionXHists.posYbins);
trajectoriesRightCorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesRightCorrectSEM = nan(numSubplots,numPosYBins,1);
trajectoriesLeftCorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesLeftCorrectSEM = nan(numSubplots,numPosYBins,1);
trajectoriesRightIncorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesRightIncorrectSEM = nan(numSubplots,numPosYBins,1);
trajectoriesLeftIncorrectAvg = nan(numSubplots,numPosYBins,1); trajectoriesLeftIncorrectSEM = nan(numSubplots,numPosYBins,1);
for subplotIdx = 1:numSubplots
    %get trial types within this subplot group
    if subplotStartEnds(subplotIdx,2) > numTrials %get trial bins to look at
        trialBins = subplotStartEnds(subplotIdx,1):numTrials; %if up to the last trials, just average whatever is there
    else
        trialBins = subplotStartEnds(subplotIdx,1):subplotStartEnds(subplotIdx,2);
    end
    correctTrialsInBlock = find(trackdata.sessOutcomesAll(trialBins) == 1);
    incorrectTrialsInBlock = find(trackdata.sessOutcomesAll(trialBins) == 0);
    rightTrials = find(trackdata.startLocAll(trialBins) == 1); %this only works right now bc trial types start on opposite sides, will have to change in future
    leftTrials = find(trackdata.startLocAll(trialBins) == 3);
    
    %grab and average different types of trajectories
    xPosInYBins = hists2plot.positionXHists.histY(trialBins,:);
    trajectoriesRightCorrect{subplotIdx} = xPosInYBins(intersect(correctTrialsInBlock,rightTrials),:);
    trajectoriesLeftCorrect{subplotIdx} = xPosInYBins(intersect(correctTrialsInBlock,leftTrials),:);
    trajectoriesRightIncorrect{subplotIdx} = xPosInYBins(intersect(incorrectTrialsInBlock,rightTrials),:);
    trajectoriesLeftIncorrect{subplotIdx} = xPosInYBins(intersect(incorrectTrialsInBlock,leftTrials),:);
    trajectoriesRightCorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(correctTrialsInBlock,rightTrials),:)); trajectoriesRightCorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(correctTrialsInBlock,rightTrials),:))/sqrt(numel(intersect(correctTrialsInBlock,rightTrials)));
    trajectoriesLeftCorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(correctTrialsInBlock,leftTrials),:)); trajectoriesLeftCorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(correctTrialsInBlock,leftTrials),:))/sqrt(numel(intersect(correctTrialsInBlock,leftTrials)));
    trajectoriesRightIncorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(incorrectTrialsInBlock,rightTrials),:)); trajectoriesRightIncorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(incorrectTrialsInBlock,rightTrials),:))/sqrt(numel(intersect(incorrectTrialsInBlock,rightTrials)));
    trajectoriesLeftIncorrectAvg(subplotIdx,:) = nanmean(xPosInYBins(intersect(incorrectTrialsInBlock,leftTrials),:)); trajectoriesLeftIncorrectSEM(subplotIdx,:) = nanstd(xPosInYBins(intersect(incorrectTrialsInBlock,leftTrials),:))/sqrt(numel(intersect(incorrectTrialsInBlock,leftTrials)));
    
    %get view angles over y position
    viewAngleInYBins = hists2plot.viewAngleHists.histY(trialBins,:);
    viewAnglesRightCorrect{subplotIdx} = viewAngleInYBins(intersect(correctTrialsInBlock,rightTrials),:);
    viewAnglesLeftCorrect{subplotIdx} = viewAngleInYBins(intersect(correctTrialsInBlock,leftTrials),:);
    viewAnglesRightIncorrect{subplotIdx} = viewAngleInYBins(intersect(incorrectTrialsInBlock,rightTrials),:);
    viewAnglesLeftIncorrect{subplotIdx} = viewAngleInYBins(intersect(incorrectTrialsInBlock,leftTrials),:);
end

%trajectories on the track
ax2(subplotIdx) = subplot(5,numSubplots,numSubplots+subplotIdx); hold on;
ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins);
if subplotIdx == round(numSubplots/2); title('Trajectories - Correct Trials'); end
p1 = plot(ybins(1:end-1), fliplr(-1*trajectoriesRightCorrect{subplotIdx}(:,1:end-1)), 'Color', [1 0 0 0.2], 'LineWidth', 1); %have to flip right now and multiple by negative bc actually on opposite sides of track
p2 = plot(ybins(1:end-1), trajectoriesLeftCorrect{subplotIdx}(:,1:end-1), 'Color', [0 0 1 0.2], 'LineWidth', 1);
xlim([0 140]); ylim([-35 35]); set(ax2(subplotIdx),'xticklabel',[]);
if subplotIdx == 1; ylabel('X position'); else; set(ax2(subplotIdx),'yticklabel',[]); end;

%view angles on the track
ax3(subplotIdx) = subplot(5,numSubplots,numSubplots*2+subplotIdx); hold on;
ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins);
p1 = plot(ybins(1:end-1), rad2deg(-1*(fliplr(viewAnglesRightCorrect{subplotIdx}(:,1:end-1)-pi))), 'Color', [1 0 0 0.2], 'LineWidth', 1); %have to flip right now and multiple by negative bc actually on opposite sides of track
p2 = plot(ybins(1:end-1), rad2deg(-1*viewAnglesLeftCorrect{subplotIdx}(:,1:end-1)), 'Color', [0 0 1 0.2], 'LineWidth', 1);
xlim([0 140]); ylim([-90 90]);
if subplotIdx == 1; ylabel('View angle (rad)'); else; set(ax3(subplotIdx),'yticklabel',[]); end; xlabel('Y position');

%plot view angle distributions on the track over time
ax4(subplotIdx) = subplot(5,numSubplots,numSubplots*3+subplotIdx); hold on;
ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins);
rightTrialsViewAnglesTemp = -1*(fliplr(viewAnglesRightCorrect{subplotIdx}(:,1:end-1)-pi));
leftTrialsViewAnglesTemp = -1*viewAnglesLeftCorrect{subplotIdx};
rightTrialsViewAngles = rightTrialsViewAnglesTemp(:,4:22); leftTrialsViewAngles = leftTrialsViewAnglesTemp(:,4:22); %get rid of first bins where all 0 vals
ybinsShort = ybins(4:22);
rightTrialsViewAngles(:,ybinsShort > 105) = []; leftTrialsViewAngles(:,ybinsShort > 105) = [];
ybinsShort(ybinsShort > 105) = [];
edges = -1.5:0.1:1.5;
rightTrialsViewAnglesDist = histcounts(rightTrialsViewAngles(:,7:end), edges); %second half of track
leftTrialsViewAnglesDist = histcounts(leftTrialsViewAngles(:,7:end), edges);
rightTrialsViewAnglesDist = rightTrialsViewAnglesDist/nansum(rightTrialsViewAnglesDist);
leftTrialsViewAnglesDist = leftTrialsViewAnglesDist/nansum(leftTrialsViewAnglesDist);
h1 = histogram('BinCounts', rightTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
h2 = histogram('BinCounts', leftTrialsViewAnglesDist, 'BinEdges', rad2deg(edges));
h1.FaceAlpha = 0.2; h2.FaceAlpha = 0.2; xlim([-90 90]);
h1.FaceColor = [1 0 0]; h2.FaceColor = [0 0 1];
if subplotIdx == round(numSubplots/2); title('View Angle Distribution in Second Half of Track - Correct Trials'); end
if subplotIdx == 1; ylabel('Proportion of all position bins'); else; set(ax4(subplotIdx),'yticklabel',[]); end; xlabel('View angle (rad)');

%plot the view angle distributions throughout the track
figure('units','normalized','position',[0 0 0.4 0.8]); hold on;
viewAnglesRightCorrectAll = []; viewAnglesLeftCorrectAll = [];
viewAnglesRightIncorrectAll = []; viewAnglesLeftIncorrectAll = [];
for subplotIdx = 1:numSubplots
    viewAnglesRightCorrectAll = [viewAnglesRightCorrectAll; viewAnglesRightCorrect{subplotIdx}];
    viewAnglesLeftCorrectAll = [viewAnglesLeftCorrectAll; viewAnglesLeftCorrect{subplotIdx}];
    viewAnglesRightIncorrectAll = [viewAnglesRightIncorrectAll; viewAnglesRightIncorrect{subplotIdx}];
    viewAnglesLeftIncorrectAll = [viewAnglesLeftIncorrectAll; viewAnglesLeftIncorrect{subplotIdx}];
end
ybins = hists2plot.positionXHists.posYbins - min(hists2plot.positionXHists.posYbins); ybinsShort = ybins(4:22);
viewAnglesRightCorrectAllTemp = -1*(fliplr(viewAnglesRightCorrectAll(:,1:end-1)-pi));
viewAnglesLeftCorrectAllTemp = -1*viewAnglesLeftCorrectAll;
viewAnglesRightCorrectAllTemp = viewAnglesRightCorrectAllTemp(:,4:22);
viewAnglesLeftCorrectAllTemp = viewAnglesLeftCorrectAllTemp(:,4:22);
viewAnglesRightIncorrectAllTemp = -1*(fliplr(viewAnglesRightIncorrectAll(:,1:end-1)-pi));
viewAnglesLeftIncorrectAllTemp = -1*viewAnglesLeftIncorrectAll;
viewAnglesRightIncorrectAllTemp = viewAnglesRightIncorrectAllTemp(:,4:22);
viewAnglesLeftIncorrectAllTemp = viewAnglesLeftIncorrectAllTemp(:,4:22);


