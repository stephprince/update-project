function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,params,behaviordata);

animals = unique(indices.behaviorindex(:,1));
trainingoptions = {'linear','shaping','choice1side','choice2side','choice1side_short','continuousalt'};
statsoutput = [];

%% concatenate data across sessions
allsessdata = concatDiamondMazeSessions(animals, indices, behaviordata, trainingoptions);

%% plot percent correct
for anIdx = 1:length(animals)
    %don't really care about linear track performance
    for trackIdx = 2:length(trainingoptions);
        plotDiamondTrackCorrectPerformance(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);
    end
end

%% plot continuous alt performance
for anIdx = 1:length(animals)
    for trackIdx = find(strcmp(trainingoptions, 'continuousalt'))
        plotContinuousAltPerformance(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);
    end
end

%% plot average trial duration for correct, failed, incorrect trials for each session
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        plotDiamondTrackBoxplotDist(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}),animals(anIdx),trainingoptions{trackIdx},dirs,'dur');
    end
end

%% plot licking (as a function of distance from reward and trials since correct)
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        plotDiamondTrackLicks(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs, params);
    end
end

%% plot view angle averages throughout the trial
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        plotViewAngle(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);
    end
end

%% plot position and velocity as a function of time
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        plotDiamondTrackPosition(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}), animals(anIdx), trainingoptions{trackIdx}, dirs);
    end
end
