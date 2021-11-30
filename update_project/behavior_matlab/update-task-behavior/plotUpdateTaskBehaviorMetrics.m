function statsoutput = plotUpdateTaskBehaviorMetrics(dirs,indices,params,behaviordata)

% plot percent correct
plotUpdateTaskCorrectPerformance(behaviordata, indices, dirs, params);
close all;

% plot all behavior metrics
plotUpdateTaskTrajectories(behaviordata, indices, dirs, params);
close all;

% plot licking activity
plotUpdateTaskLickingActivity(behaviordata, indices, dirs, params);
close all;

% plot delay information
plotUpdateTaskDelayInfo(behaviordata, indices, dirs, params);





