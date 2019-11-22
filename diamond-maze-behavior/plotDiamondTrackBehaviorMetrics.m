function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,behaviordata);

animals = unique(indices.behaviorindex(:,1));
trainingoptions = {'linear','shaping','choice1side','choice2side','choice1side_short','continuousalt'};
statsoutput = [];

%% concatenate data across sessions
allsessdata = concatDiamondMazeSessions(animals, indices, behaviordata, trainingoptions);

%% plot percent correct
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions) %don't really care about linear track performance
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
    for trackIdx = 2:length(trainingoptions) %don't really care about linear track performance
        plotDiamondTrackBoxplotDist(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}),animals(anIdx),trainingoptions{trackIdx},dirs,'dur');
    end
end

%% plot view angle averages throughout the trial

%% plot licking (as a function of distance from reward and trials since correct)

%% plot velocity throughout the track

%% plot position as a functin of time


%% plot position traces for all trials color coded
for anIdx = 1:length(animals)
    for trackIdx = 3:length(trainingoptions)

    end
end




%% plot position traces
% plots them using the raw trial data traces before they're resampled so can't average
% for anIdx = 1:length(animals)
%     inclsess = find(indices.behaviorindex(:,1) == animals(anIdx));
%     for trackIdx = 2:length(trainingoptions)
%         track = trainingoptions{trackIdx};
%         for sessIdx = 1:size(inclsess,1)
%             sessindex = indices.behaviorindex(inclsess(sessIdx),:);
%             sessdata = behaviordata.bySession{sessindex(1)}{sessindex(2)}{sessindex(3)};
%             trialdata = behaviordata.byTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
%             if strcmp(sessdata.trainingtype,track)
%                 clr = 'rmygcbkrmygcbkrmygcbkrmygcbkrmygcbkrmygcbkrygcbk';
%                 figure; hold on;
%                 for trialIdx = 1:sessdata.numTrials
%                     phaseInds = trialdata{trialIdx}.phaseInds;
%                     for phaseIdx = 1:size(phaseInds,1)
%                         xpos = trialdata{trialIdx}.positionX(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
%                         ypos = trialdata{trialIdx}.positionY(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
%                         plot(xpos, ypos, [clr(trialIdx) 'o'],'MarkerSize',2);
%                         xlabel('x-axis'); ylabel('y-axis');
%                         ylim([0 700]); xlim([-200 200]);
%                         title(['S' num2str(animals(anIdx)) ' position over trials - sess ' num2str(sessIdx)])
%                     end
%                 end
%                 filename = [dirs.behaviorfigdir 'position_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx)) '_' num2str(sessindex(2))];
%                 saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
%             end
%         end
%     end
% end

for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        track = trainingoptions{trackIdx};

        %plot individual trajectories during choice and encoding phases
        figure('units','normalized','outerposition',[0 0 1 1]); hold on;
        clr = 'krg'; trialTypes = [-1 0 1];
        for trialTypeIdx = 1:3
            trialsToPlot = find(allsessdata(animals(anIdx)).(track).sessOutcomesAll == trialTypes(trialTypeIdx));
            subplot(1,2,1); hold on;
            plot(allsessdata(animals(anIdx)).(track).posXEnc(trialsToPlot,:)',allsessdata(animals(anIdx)).(track).posYEnc(trialsToPlot,:)',clr(trialTypeIdx));
            title('Encoding'); xlabel('X position'); ylabel('Y position');
            subplot(1,2,2); hold on;
            plot(allsessdata(animals(anIdx)).(track).posXChoice(trialsToPlot,:)',allsessdata(animals(anIdx)).(track).posYChoice(trialsToPlot,:)',clr(trialTypeIdx));
            title('Choice'); xlabel('X position'); ylabel('Y position');
        end
        sgtitle(['S' num2str(animals(anIdx)) ' individual trajectories during choice/encoding'])
        filename = [dirs.behaviorfigdir 'trajectoriesindiv_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

        %plot average trajectories during choice and encoding phase
        figure('units','normalized','outerposition',[0 0 1 1]); hold on;
        clr = 'krg'; trialTypes = [-1 0 1];
        for trialTypeIdx = 1:3
            %get all the different trial types so I can average
            trialOutcomes = find(allsessdata(animals(anIdx)).(track).sessOutcomesAll == trialTypes(trialTypeIdx));
            [trialstemp,times,vals] = find(allsessdata(animals(anIdx)).(track).posYEnc > 600);
            trialStartNorth = unique(trialstemp);
            [trialstemp,times,vals] = find(allsessdata(animals(anIdx)).(track).posYEnc < 200);
            trialStartSouth = unique(trialstemp);
            [trialstemp,times,vals] = find(allsessdata(animals(anIdx)).(track).posXEnc > 5);
            turnPathEast = unique(trialstemp);
            [trialstemp,times,vals] = find(allsessdata(animals(anIdx)).(track).posXEnc < -5);
            turnPathWest = unique(trialstemp);

            %average different types
            northeastTrials = intersect(intersect(trialStartNorth,turnPathEast),trialOutcomes);
            northwestTrials = intersect(intersect(trialStartNorth,turnPathWest),trialOutcomes);
            southeastTrials = intersect(intersect(trialStartSouth,turnPathEast),trialOutcomes);
            southwestTrials = intersect(intersect(trialStartSouth,turnPathWest),trialOutcomes);

            %plot the averages
%             subplot(1,2,1); hold on;
%             posXAvg = nanmean(allsessdata(animals(anIdx)).(track).posXEnc(northeastTrials,:),1);
%             posXSEM = nanstd(allsessdata(animals(anIdx)).(track).posXEnc(northeastTrials,:),1)/sqrt(size(allsessdata(animals(anIdx)).(track).posXEnc(northeastTrials,:),1));
%             posYAvg = nanmean(allsessdata(animals(anIdx)).(track).posYEnc(northeastTrials,:),1);
%             posYSEM = nanstd(allsessdata(animals(anIdx)).(track).posYEnc(northeastTrials,:),1)/sqrt(size(allsessdata(animals(anIdx)).(track).posYEnc(northeastTrials,:),1));
%             plot(posXAvg,posYAvg,clr(trialTypeIdx),'LineWidth',2);
%             ciplot(posXAvg-posXSEM,posXAvg+posXSEM);
%             plot(nanmean(allsessdata(animals(anIdx)).(track).posXEnc(northeastTrials,:),1),nanmean(allsessdata(animals(anIdx)).(track).posYEnc(northeastTrials,:),1),clr(trialTypeIdx),'LineWidth',2);
%
%             title('Encoding'); xlabel('X position'); ylabel('Y position');
%             subplot(1,2,2); hold on;
%             plot(allsessdata(animals(anIdx)).(track).posXChoice(trialsToPlot,:)',allsessdata(animals(anIdx)).(track).posYChoice(trialsToPlot,:)',clr(trialTypeIdx));
%             title('Choice'); xlabel('X position'); ylabel('Y position');
        end
        sgtitle(['S' num2str(animals(anIdx)) ' individual trajectories during choice/encoding'])
        filename = [dirs.behaviorfigdir 'trajectoriesavg_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

    end
end
