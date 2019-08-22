function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,behaviordata);

trainingoptions = {'linear','shaping','choice1side'};
animals = unique(indices.behaviorindex(:,1));

%% concatenate data across sessions
for anIdx = 1:length(animals)
    inclsess = find(indices.behaviorindex(:,1) == animals(anIdx));
    for trackIdx = 1:length(trainingoptions)
        %initialize structures and track info
        track = trainingoptions{trackIdx};
        allsessdata(animals(anIdx)).(track).perCorrect = []; allsessdata(animals(anIdx)).(track).trialDur = [];
        allsessdata(animals(anIdx)).(track).numTrials = []; allsessdata(animals(anIdx)).(track).trialDurCorrect = [];
        allsessdata(animals(anIdx)).(track).trialDurFailed = []; allsessdata(animals(anIdx)).(track).trialDurIncorrect = [];
        allsessdata(animals(anIdx)).(track).sessOutcomesAll = [];
        for sessIdx = 1:size(inclsess,1)
            sessindex = indices.behaviorindex(inclsess(sessIdx),:);
            sessdata = behaviordata.bySession{sessindex(1)}{sessindex(2)}{sessindex(3)};
            trialdata = behaviordata.byTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
            if strcmp(sessdata.trainingtype,track)
                %percent correct
                perCorrect = sessdata.numCorrect/sessdata.numTrials;
                allsessdata(animals(anIdx)).(track).perCorrect = [allsessdata(animals(anIdx)).(track).perCorrect; perCorrect];
                allsessdata(animals(anIdx)).(track).numTrials = [allsessdata(animals(anIdx)).(track).numTrials; sessdata.numTrials];
                
                %trial performance
                allsessdata(animals(anIdx)).(track).sessOutcomes{sessIdx} = sessdata.sessOutcomes;
                allsessdata(animals(anIdx)).(track).correctTrials{sessIdx} = sessdata.logicalCorrect;
                allsessdata(animals(anIdx)).(track).sessOutcomesAll = [allsessdata(animals(anIdx)).(track).sessOutcomesAll sessdata.sessOutcomes];
                
                %trial params
                allsessdata(animals(anIdx)).(track).turnDirEnc{sessIdx} = sessdata.trialTurnDirEnc; %1 = left, 2 = right
                allsessdata(animals(anIdx)).(track).turnDirChoice{sessIdx} = sessdata.trialTurnDirChoice; %1 = left, 2 = right
                allsessdata(animals(anIdx)).(track).startLoc{sessIdx} = sessdata.trialStartLoc; %1 = north, 2 = east, 3 = south, 4 = west 
                allsessdata(animals(anIdx)).(track).choiceLoc{sessIdx} = sessdata.trialChoiceLoc; %1 = north, 2 = east, 3 = south, 4 = west 
                
                %trial duration
                allsessdata(animals(anIdx)).(track).trialDur = [allsessdata(animals(anIdx)).(track).trialDur; sessdata.trialDur];
                allsessdata(animals(anIdx)).(track).trialDurCorrect = [allsessdata(animals(anIdx)).(track).trialDurCorrect; sessdata.trialDur(sessdata.logicalCorrect)];
                allsessdata(animals(anIdx)).(track).trialDurFailed = [allsessdata(animals(anIdx)).(track).trialDurFailed; sessdata.trialDur(sessdata.sessOutcomes == -1)];
                allsessdata(animals(anIdx)).(track).trialDurIncorrect = [allsessdata(animals(anIdx)).(track).trialDurIncorrect; sessdata.trialDur(sessdata.sessOutcomes == 0)];
            end
        end
    end
end
 
%% plot percent correct 
for anIdx = 1:length(animals)
    for trackIdx = 1:length(trainingoptions)
        numSessions = length(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect);
        numTrials = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).numTrials;
        figure; hold on;
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect,'b-','LineWidth',2);
        xlabel('Session'); ylabel('% Correct Trials');
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'percentcorrecttrials_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        figure; hold on;
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect.*numTrials,'g-','LineWidth',2);
        xlabel('Session'); ylabel('# Correct Trials');
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'numcorrecttrials_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        figure; hold on;
        outcomes = allsessdata(animals(anIdx)).(track).sessOutcomesAll;
        h1 = plot(find(outcomes == 1), ones(length(find(outcomes == 1))),'g*','LineWidth',2,'DisplayName','Correct');
        h2 = plot(find(outcomes == -1), zeros(length(find(outcomes == -1))),'k*','LineWidth',2,'DisplayName','Failed');
        h3 = plot(find(outcomes == 0), zeros(length(find(outcomes == 0))),'r*','LineWidth',2,'DisplayName','Incorrect');
        xlabel('Trial'); ylabel('Outcome');
        ylim([-0.5 1.5])
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track g = correct, b = failed, r = incorrect'])
        filename = [dirs.behaviorfigdir 'trialbytrialperformance_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end
end

%% plot trial duration
for anIdx = 1:length(animals)
    for trackIdx = 1:length(trainingoptions)
        track = trainingoptions{trackIdx};
        numSessions = length(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect);
        numTrials = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).numTrials;
        figure; hold on;
        plot(1:sum(numTrials),allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).trialDur,'b-','LineWidth',2);
        xlabel('Trial'); ylabel('Duration (s)');
        title(['S' num2str(animals(anIdx)) ' trial duration on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'trialdur_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        figure; hold on;
        numCorrectTrials = length(allsessdata(animals(anIdx)).(track).trialDurCorrect);
        numIncorrectTrials = length(allsessdata(animals(anIdx)).(track).trialDurIncorrect);
        numFailedTrials = length(allsessdata(animals(anIdx)).(track).trialDurFailed);
        plot(1:numCorrectTrials,allsessdata(animals(anIdx)).(track).trialDurCorrect,'g-','LineWidth',2);
        plot(1:numFailedTrials,allsessdata(animals(anIdx)).(track).trialDurFailed,'k-','LineWidth',2);
        plot(1:numIncorrectTrials,allsessdata(animals(anIdx)).(track).trialDurIncorrect,'r-','LineWidth',2);
        xlabel('Trial'); ylabel('Duration (s)'); legend('Correct', 'Failed','Incorrect');
        title(['S' num2str(animals(anIdx)) ' trial duration on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'trialdurcorrectvincorrect_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end
end

%% plot position traces
for anIdx = 1:length(animals)
    inclsess = find(indices.behaviorindex(:,1) == animals(anIdx));
    for trackIdx = 2:length(trainingoptions)
        track = trainingoptions{trackIdx};
        for sessIdx = 1:size(inclsess,1)
            sessindex = indices.behaviorindex(inclsess(sessIdx),:);
            sessdata = behaviordata.bySession{sessindex(1)}{sessindex(2)}{sessindex(3)};
            trialdata = behaviordata.byTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
            if strcmp(sessdata.trainingtype,track)
                clr = 'rmygcbkrmygcbkrmygcbkrmygcbkrmygcbkrmygcbkrygcbk';
                figure; hold on;
                for trialIdx = 1:sessdata.numTrials
                    phaseInds = trialdata{trialIdx}.phaseInds;
                    for phaseIdx = 1:size(phaseInds,1)
                        xpos = trialdata{trialIdx}.positionX(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
                        ypos = trialdata{trialIdx}.positionY(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
                        plot(xpos, ypos, [clr(trialIdx) 'o'],'MarkerSize',2);
                        xlabel('x-axis'); ylabel('y-axis');
                        ylim([0 700]); xlim([-200 200]);
                        title(['S' num2str(animals(anIdx)) ' position over trials - sess ' num2str(sessIdx)])
                    end
                end
                filename = [dirs.behaviorfigdir 'position_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx)) '_' num2str(sessindex(2))];
                saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
            end
        end
    end
end

figure; hold on; 
for i = 1:size(trialdata,1)
    phaseInds = trialdata{i}.phaseInds;
    clr = 'rgbk';
    for j = 1:size(trialdata{i}.phaseInds,1)
        plot(trialdata{i}.positionX(phaseInds(j,1):phaseInds(j,2)),trialdata{i}.positionY(phaseInds(j,1):phaseInds(j,2)),['o' clr(j)]); 
        pause; 
    end
end

figure; hold on;
for trialIdx = 1:size(trialStarts,1); 
    phaseInds = behaviorDataDiamondByTrial{trialIdx}.phaseInds;
    plot(behaviorDataDiamondByTrial{trialIdx}.positionX(phaseInds(1,1)+1:phaseInds(1,2)),behaviorDataDiamondByTrial{trialIdx}.positionY(phaseInds(1,1)+1:phaseInds(1,2)));
    pause
end

%% plot view angle traces




