function statsoutput = plotUpdateTaskBehaviorMetrics(dirs,indices,params,behaviordata)

% plot all behavior metrics
plotUpdateTaskTrajectories(behaviordata, indices, dirs, params);

% plot delay information
plotUpdateTaskDelayInfo(behaviordata, indices, dirs, params);

% plot percent correct
plotUpdateTaskCorrectPerformance(behaviordata, indices, dirs, params);

% plot licking activity
plotUpdateTaskLickingActivity(behaviordata, indices, dirs, params);





