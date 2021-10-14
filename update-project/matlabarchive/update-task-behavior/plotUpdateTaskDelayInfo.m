function plotUpdateTaskDelayInfo(trackdata, indices, dirs, params)

close all;

%% setup info
%get directories to save in
savedfiguresdir = [dirs.behaviorfigdir 'delayInformation\'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;

%get different parameters to calculate percent correct by like delay length
allDelayLocations = []; 
howMuchToRound = 20;
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    delayLocations = cell2mat(cellfun(@(x) round(x.trialDelayLocation/howMuchToRound)*howMuchToRound, animaldata.trialTable,'UniformOutput',0));
    delayLocations(isnan(delayLocations)) = [];
    allDelayLocations = [allDelayLocations; delayLocations];
end
params.delayLocations = sort(unique(allDelayLocations), 'descend');
params.plotCategories = [2 nan 1; 3 nan 1; [repmat(4,numel(params.delayLocations),1) params.delayLocations ones(numel(params.delayLocations),1)]; 4 nan 2];

%% create position histograms for all the animals
delayLocationsAll = []; delayDurationsAll = []; trialDurationsAll = []; trackIDsAll = []; trialDurationsDelayOnlyAll = []; delayDurationsAllByParam = cell(1,size(params.plotCategories,1)-1,1);
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    delayLocations = []; delayDurations = []; trialDurations = []; trackIDs = []; trialDurationsDelayOnly = [];
    for paramIdx = 1:size(params.plotCategories,1)-1 %not include the update trials here bc those are falsely nan
        %% compile all trials for each track across days for each world
        %get trials to pick
        trialdata = getTrialsOfInterest(animaldata, params, paramIdx, howMuchToRound);
        
        %get when the duration info
        delayDurationsTemp = trialdata.trialDelayDuration;
        delayDurationsTemp(isnan(delayDurationsTemp)) = 0; %if no delay, the duration was 0
        trialDurationsTemp = trialdata.trialDur;
        
        %plot delay length in time as a function of position
        %concat data for relevant tracks
        delayLength{paramIdx} = params.plotCategories(paramIdx,2);
        if isnan(delayLength{paramIdx})
            delayLength{paramIdx} = 250; %counts as a 0 delay length for ymaze long
        end
        if paramIdx ~= 1 %only calculate delay locations/durations if in the long maze or the update task
            delayLocations = [delayLocations; repmat(delayLength{paramIdx},numel(delayDurationsTemp),1)];
            delayDurations = [delayDurations; delayDurationsTemp];
            trialDurationsDelayOnly  = [trialDurationsDelayOnly; trialDurationsTemp];
        end
        
        %get trial durations only for each type of world
        trialDurations = [trialDurations; trialDurationsTemp];
        trackIdx = params.plotCategories(paramIdx,1);
        trackIDs = [trackIDs; repmat(trackIdx, numel(delayDurationsTemp),1)];
        
        delayDurationsAllByParam{paramIdx} = [delayDurationsAllByParam{paramIdx}; delayDurationsTemp];
    end
    
    %get data for the all animals combined info
    delayLocationsAll = [delayLocationsAll; delayLocations];
    delayDurationsAll = [delayDurationsAll; delayDurations];
    trialDurationsAll = [trialDurationsAll; trialDurations];
    trackIDsAll = [trackIDsAll; trackIDs];
    trialDurationsDelayOnlyAll = [trialDurationsDelayOnlyAll; trialDurationsDelayOnly];
    
    %plot the data for the violin plot
    cmap = cbrewer('qual','Set2',numel(indices.animals));
    figure(100); hold on;
    ax1(anIdx) = subplot(numel(indices.animals),1,anIdx);
    violinplot(delayDurations, delayLocations, 'ViolinColor', cmap(anIdx,:))    
    xlabel(['Delay Location (smaller is longer)']);
    ylabel('Delay Duration (s)');
    linkaxes(ax1, 'y'); linkaxes(ax1, 'x');
    title(['S' num2str(indices.animals(anIdx))])
    sgtitle('Delay durations in time for each delay location')
    filename = [savedfiguresdir 'delayDurationVsDelayLength_AllAnimals_violinplot'];
    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    
    figure(200); hold on;
    ax1(anIdx) = subplot(numel(indices.animals),1,anIdx);
    violinplot(trialDurationsDelayOnly, delayLocations, 'ViolinColor', cmap(anIdx,:))
    xlabel(['Delay Location (smaller is longer)']);
    ylabel('Trial Duration (s)');
    linkaxes(ax1, 'y'); linkaxes(ax1, 'x');
    title(['S' num2str(indices.animals(anIdx))])
    sgtitle('Trial durations in time for each delay location')
    filename = [savedfiguresdir 'trialDurationVsDelayLength_AllAnimals_violinplot'];
    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    
    figure(300); hold on;
    ax1(anIdx) = subplot(numel(indices.animals),1,anIdx);
    violinplot(trialDurations, trackIDs, 'ViolinColor', cmap(anIdx,:))
    xlabel(['track IDs']);
    ylabel('Trial Duration (s)');
    linkaxes(ax1, 'y'); linkaxes(ax1, 'x');
    title(['S' num2str(indices.animals(anIdx))])
    sgtitle('Trial durations for each track')
    filename = [savedfiguresdir 'trialDurationVsTrack_AllAnimals_violinplot'];
    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end
close all;

%% plot for all animals

cmap = cbrewer('seq','YlOrRd',numel(delayDurationsAllByParam));
figure(700); hold on; clf;
for paramIdx = 3:numel(delayDurationsAllByParam) %skip the no delay parts
    if ~isempty(delayDurationsAllByParam{paramIdx})
        edges = linspace(min(delayDurationsAllByParam{paramIdx}), 90, 50); %look up to 60 seconds
        delayDurHist = histc(delayDurationsAllByParam{paramIdx},edges);
        delayDurHistNorm = delayDurHist./sum(delayDurHist);
        plot(edges,cumsum(delayDurHistNorm), 'Color', cmap(paramIdx,:), 'linewidth', 2); hold on;
    end
    whichDelays{paramIdx-2} = num2str(delayLength{paramIdx});
    delaysMean{paramIdx-2} = nanmean(delayDurationsAllByParam{paramIdx});
    delaysStd{paramIdx-2} = nanstd(delayDurationsAllByParam{paramIdx})/sqrt(numel(delayDurationsAllByParam{paramIdx}));
end
xlabel('Delay duration (s)'); ylabel('Cumulative Fraction')
ylim([0.1 1.01]); xlim([0 60]);
legend(whichDelays)
title('Delay durations in time for each delay location')
filename = [savedfiguresdir 'delayDurationCumFract_AllAnimalsCombined_violinplot'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

figure(400); hold on;
violinplot(delayDurationsAll, delayLocationsAll, 'ViolinColor', cmap(anIdx,:))
xlabel(['Delay Location (smaller is longer)']);
ylabel('Delay Duration (s)');
sgtitle('Delay durations in time for each delay location')
filename = [savedfiguresdir 'delayDurationVsDelayLength_AllAnimalsCombined_violinplot'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

figure(500); hold on;
violinplot(trialDurationsDelayOnlyAll, delayLocationsAll, 'ViolinColor', cmap(anIdx,:))
xlabel(['Delay Location (smaller is longer)']);
ylabel('Trial Duration (s)');
sgtitle('Trial durations in time for each delay location')
filename = [savedfiguresdir 'trialDurationVsDelayLength_AllAnimalsCombined_violinplot'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

figure(600); hold on;
violinplot(trialDurationsAll, trackIDsAll, 'ViolinColor', cmap(anIdx,:))
xlabel(['track IDs']);
ylabel('Trial Duration (s)');
sgtitle('Trial durations for each track')
filename = [savedfiguresdir 'trialDurationVsTrack_AllAnimalsCombined_violinplot'];
saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

        