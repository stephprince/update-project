function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,behaviordata);

trainingoptions = {'linear','shaping'};
animals = unique(indices.behaviorindex(:,1));

%% concatenate data across sessions
for anIdx = 1:length(animals)
    inclsess = find(indices.behaviorindex(:,1) == animals(anIdx));
    for trackIdx = 1:length(trainingoptions)
        %initialize structures and track info
        track = trainingoptions{trackIdx};
        allsessdata(animals(anIdx)).(track).perCorrect = []; allsessdata(animals(anIdx)).(track).trialdur = [];
        allsessdata(animals(anIdx)).(track).trialdurAvg = []; allsessdata(animals(anIdx)).(track).trialdurSem = [];
        allsessdata(animals(anIdx)).(track).numTrials = [];
        for sessIdx = 1:size(inclsess,1)
            sessindex = indices.behaviorindex(inclsess(sessIdx),:);
            sessdata = behaviordata.bySession{sessindex(1)}{sessindex(2)}{sessindex(3)};
            trialdata = behaviordata.byTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
            if strcmp(sessdata.trainingtype,track)
                %percent correct
                perCorrect = sessdata.numCorrect/sessdata.numTrials;
                allsessdata(animals(anIdx)).(track).perCorrect = [allsessdata(animals(anIdx)).(track).perCorrect; perCorrect];
                allsessdata(animals(anIdx)).(track).numTrials = [allsessdata(animals(anIdx)).(track).numTrials; sessdata.numTrials];
            
                %trial duration
                allsessdata(animals(anIdx)).(track).trialdur = [allsessdata(animals(anIdx)).(track).trialdur; sessdata.trialdur];
                allsessdata(animals(anIdx)).(track).trialdurAvg = [allsessdata(animals(anIdx)).(track).trialdurAvg; sessdata.trialdurAvg];
                allsessdata(animals(anIdx)).(track).trialdurSem = [allsessdata(animals(anIdx)).(track).trialdurSem; sessdata.trialdurSem];
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
    end
end

%% plot trial duration
for anIdx = 1:length(animals)
    for trackIdx = 1:length(trainingoptions)
        numSessions = length(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect);
        numTrials = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).numTrials;
        figure; hold on;
        plot(1:sum(numTrials),allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).trialdur,'b-','LineWidth',2);
        xlabel('Trial'); ylabel('Duration (s)');
        title(['S' num2str(animals(anIdx)) ' trial duration on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'trialdur_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        figure; hold on;
        trialavgs = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).trialdurAvg;
        trialsems = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).trialdurSem;
        plot(1:numSessions,trialavgs,'b-','LineWidth',2);
        errorbar2(1:numSessions, trialavgs,trialsems, 0.25) 
        xlabel('Trial'); ylabel('Duration (s)');
        title(['S' num2str(animals(anIdx)) ' trial duration on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'avgtrialdur_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
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
            if strcmp(sessdata.trainingtype,track)
                figure; hold on;
                for trialIdx = 1:sessdata.numTrials
                    phaseInds = round(sessdata.phaseInds{trialIdx});
                    clr = 'rmbk';
                    for phaseIdx = 1:size(sessdata.phase,2)
                        xpos = sessdata.phase(phaseIdx).posXNormSouth(trialIdx,:);
                        ypos = sessdata.phase(phaseIdx).posYNormSouth(trialIdx,:);
                        plot(xpos, ypos, [clr(phaseIdx)],'LineWidth',2);
                        xlabel('x-axis'); ylabel('y-axis');
                        title(['S' num2str(animals(anIdx)) ' position over North starting trials - sess ' num2str(sessIdx)])

                    end
                    pause
                end
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




