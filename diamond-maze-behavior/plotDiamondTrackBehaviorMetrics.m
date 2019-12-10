function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,params,behaviordata);

animals = unique(indices.behaviorindex(:,1));
trainingoptions = {'linear','shaping','choice1side','choice2side','choice1side_short','continuousalt'};
statsoutput = [];

%% concatenate data across sessions
allsessdata = concatDiamondMazeSessions(animals, indices, behaviordata, trainingoptions);

%% plot all behavior metrics (don't really care about linear track performance so track options start at 2)
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        % plot percent correct
        plotDiamondTrackCorrectPerformance(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);

        % plot licking (as a function of distance from reward and trials since correct)
        plotDiamondTrackLicks(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs, params);

        % plot average trial duration for correct, failed, incorrect trials for each session
        plotDiamondTrackBoxplotDist(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}),animals(anIdx),trainingoptions{trackIdx},dirs,'dur');

        % plot metrics as a function of position throughout the trial
        plotDiamondTrackMetricsByPosition(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);

        % plot continuous alt performance
        if strcmp(trainingoptions{trackIdx},'continuousalt')
          plotContinuousAltPerformance(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);
        end
    end
end
