function statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,behaviordata);

trainingoptions = {'linear','shaping','choice1side','choice2side'};
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
        allsessdata(animals(anIdx)).(track).sessOutcomesAll = []; allsessdata(animals(anIdx)).(track).sessOutcomesRight = [];
        allsessdata(animals(anIdx)).(track).sessOutcomesLeft = []; allsessdata(animals(anIdx)).(track).perCorrectRight = [];
        allsessdata(animals(anIdx)).(track).perCorrectLeft = []; allsessdata(animals(anIdx)).(track).posXEnc = [];
        allsessdata(animals(anIdx)).(track).posYEnc = []; allsessdata(animals(anIdx)).(track).posXChoice = [];
        allsessdata(animals(anIdx)).(track).posYChoice = []; allsessdata(animals(anIdx)).(track).sessOutcomesRightAll = [];
        allsessdata(animals(anIdx)).(track).sessOutcomesLeftAll = []; allsessdata(animals(anIdx)).(track).rewardLocEncAll = [];
        allsessdata(animals(anIdx)).(track).rewardLocChoiceAll = []; allsessdata(animals(anIdx)).(track).sessOutcomesSameAll = [];
        allsessdata(animals(anIdx)).(track).sessOutcomesAltAll = []; allsessdata(animals(anIdx)).(track).perCorrectAlt = [];
        allsessdata(animals(anIdx)).(track).perCorrectSame = [];
        sesscounter = 1; %keep track of how many sessions on each track there are
        for sessIdx = 1:size(inclsess,1)
            sessindex = indices.behaviorindex(inclsess(sessIdx),:);
            sessdata = behaviordata.bySession{sessindex(1)}{sessindex(2)}{sessindex(3)};
            trialdata = behaviordata.byTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};
            if strcmp(sessdata.trainingtype,track)
                %percent correct
                perCorrect = sessdata.numCorrect/sessdata.numTrials;
                perCorrectRight = sessdata.numCorrectRight/sessdata.numTrialsRight;
                perCorrectLeft = sessdata.numCorrectLeft/sessdata.numTrialsLeft;
                perCorrectSame = sessdata.numCorrectSame/sessdata.numTrialsSame;
                perCorrectAlt = sessdata.numCorrectAlt/sessdata.numTrialsAlt;
                allsessdata(animals(anIdx)).(track).perCorrect = [allsessdata(animals(anIdx)).(track).perCorrect; perCorrect];
                allsessdata(animals(anIdx)).(track).perCorrectRight = [allsessdata(animals(anIdx)).(track).perCorrectRight; perCorrectRight];
                allsessdata(animals(anIdx)).(track).perCorrectLeft = [allsessdata(animals(anIdx)).(track).perCorrectLeft; perCorrectLeft];
                allsessdata(animals(anIdx)).(track).perCorrectSame = [allsessdata(animals(anIdx)).(track).perCorrectSame; perCorrectSame];
                allsessdata(animals(anIdx)).(track).perCorrectAlt = [allsessdata(animals(anIdx)).(track).perCorrectAlt; perCorrectAlt];
                allsessdata(animals(anIdx)).(track).numTrials = [allsessdata(animals(anIdx)).(track).numTrials; sessdata.numTrials];
                
                %trial params
                allsessdata(animals(anIdx)).(track).turnDirEnc{sesscounter} = sessdata.trialTurnDirEnc; %1 = left, 2 = right
                allsessdata(animals(anIdx)).(track).turnDirChoice{sesscounter} = sessdata.trialTurnDirChoice; %1 = left, 2 = right
                allsessdata(animals(anIdx)).(track).startLoc{sesscounter} = sessdata.trialStartLoc; %1 = north, 2 = east, 3 = south, 4 = west 
                allsessdata(animals(anIdx)).(track).choiceLoc{sesscounter} = sessdata.trialChoiceLoc; %1 = north, 2 = east, 3 = south, 4 = west 
                allsessdata(animals(anIdx)).(track).rewardLocEnc{sesscounter} = sessdata.trialCorrectEncLoc;
                allsessdata(animals(anIdx)).(track).rewardLocChoice{sesscounter} = sessdata.trialCorrectChoiceLoc;
                
                %trial performance
                allsessdata(animals(anIdx)).(track).sessOutcomes{sesscounter} = sessdata.sessOutcomes;
                allsessdata(animals(anIdx)).(track).sessOutcomesRight{sesscounter} = sessdata.sessOutcomesRight;
                allsessdata(animals(anIdx)).(track).sessOutcomesLeft{sesscounter} = sessdata.sessOutcomesLeft;
                allsessdata(animals(anIdx)).(track).sessOutcomesSame{sesscounter} = sessdata.sessOutcomesSame; 
                allsessdata(animals(anIdx)).(track).sessOutcomesAlt{sesscounter} = sessdata.sessOutcomesAlt;
                allsessdata(animals(anIdx)).(track).correctTrials{sesscounter} = sessdata.logicalCorrect;
                allsessdata(animals(anIdx)).(track).sessOutcomesAll = [allsessdata(animals(anIdx)).(track).sessOutcomesAll sessdata.sessOutcomes];
                allsessdata(animals(anIdx)).(track).sessOutcomesRightAll = [allsessdata(animals(anIdx)).(track).sessOutcomesRightAll sessdata.sessOutcomesRight];
                allsessdata(animals(anIdx)).(track).sessOutcomesLeftAll = [allsessdata(animals(anIdx)).(track).sessOutcomesLeftAll sessdata.sessOutcomesLeft];
                allsessdata(animals(anIdx)).(track).sessOutcomesSameAll = [allsessdata(animals(anIdx)).(track).sessOutcomesSameAll sessdata.sessOutcomesSame];
                allsessdata(animals(anIdx)).(track).sessOutcomesAltAll = [allsessdata(animals(anIdx)).(track).sessOutcomesAltAll sessdata.sessOutcomesAlt]; 
                
                %trial duration
                allsessdata(animals(anIdx)).(track).trialDur = [allsessdata(animals(anIdx)).(track).trialDur; sessdata.trialDur];
                allsessdata(animals(anIdx)).(track).trialDurCorrect = [allsessdata(animals(anIdx)).(track).trialDurCorrect; sessdata.trialDur(sessdata.logicalCorrect)];
                allsessdata(animals(anIdx)).(track).trialDurFailed = [allsessdata(animals(anIdx)).(track).trialDurFailed; sessdata.trialDur(sessdata.sessOutcomes == -1)];
                allsessdata(animals(anIdx)).(track).trialDurIncorrect = [allsessdata(animals(anIdx)).(track).trialDurIncorrect; sessdata.trialDur(sessdata.sessOutcomes == 0)];
                
                %trial position
                allsessdata(animals(anIdx)).(track).posXEnc = [allsessdata(animals(anIdx)).(track).posXEnc; sessdata.phase(1).posXNorm]; %encoding phase is phase 1
                allsessdata(animals(anIdx)).(track).posYEnc = [allsessdata(animals(anIdx)).(track).posYEnc; sessdata.phase(1).posYNorm]; %encoding phase is phase 1
                allsessdata(animals(anIdx)).(track).posXChoice = [allsessdata(animals(anIdx)).(track).posXChoice; sessdata.phase(3).posXNorm]; %encoding phase is phase 3
                allsessdata(animals(anIdx)).(track).posYChoice = [allsessdata(animals(anIdx)).(track).posYChoice; sessdata.phase(3).posYNorm]; %encoding phase is phase 3
                
                sesscounter = sesscounter + 1;
            end
        end
    end
end
 
%% plot percent correct 
for anIdx = 1:length(animals)
    for trackIdx = 2:length(trainingoptions)
        numSessions = length(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect);
        numTrials = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).numTrials;
        
        %percent correct trials
        figure; hold on;
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect,'bo-','LineWidth',2);
        xlabel('Session'); ylabel('% Correct Trials');
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'percentcorrecttrials_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        %percent correct trials separated by right vs. left
        figure; hold on;
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrectRight,'ro-','LineWidth',2);
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrectLeft,'bo-','LineWidth',2);
        xlabel('Session'); ylabel('% Correct Trials');
        legend('Right Turn','Left Turn')
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'percentcorrecttrials_rightvsleft_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        %percent correct same vs. alternate side
        figure; hold on;
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrectSame,'go-','LineWidth',2);
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrectAlt,'mo-','LineWidth',2);
        xlabel('Session'); ylabel('% Correct Trials');
        legend('Same Turn','Alt Turn')
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'percentcorrecttrials_samevsalt_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        %number correct trials
        figure; hold on;
        plot(1:numSessions,allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect.*numTrials,'go-','LineWidth',2);
        xlabel('Session'); ylabel('# Correct Trials');
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'numcorrecttrials_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        %all trials for all sessions
        figure; hold on;
        outcomes = allsessdata(animals(anIdx)).(track).sessOutcomesAll;
        plot(find(outcomes == 1), ones(length(find(outcomes == 1))),'g*','LineWidth',2,'DisplayName','Correct');
        plot(find(outcomes == -1), zeros(length(find(outcomes == -1))),'k*','LineWidth',2,'DisplayName','Failed');
        plot(find(outcomes == 0), [zeros(length(find(outcomes == 0)))-1],'r*','LineWidth',2,'DisplayName','Incorrect');
        xlabel('Trial'); ylabel('Outcome');
        ylim([-1.5 1.5])
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track g = correct, b = failed, r = incorrect'])
        filename = [dirs.behaviorfigdir 'trialbytrialperformance_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx)) 'all'];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        %all trials for each session figure; hold on;
        for sessIdx = 1:size(allsessdata(animals(anIdx)).(track).sessOutcomes,2)
            outcomes = allsessdata(animals(anIdx)).(track).sessOutcomes{sessIdx};
            figure; hold on;
            plot(find(outcomes == 1), ones(length(find(outcomes == 1))),'g*','LineWidth',2,'DisplayName','Correct');
            plot(find(outcomes == -1), zeros(length(find(outcomes == -1))),'k*','LineWidth',2,'DisplayName','Failed');
            plot(find(outcomes == 0), [zeros(length(find(outcomes == 0)))-1],'r*','LineWidth',2,'DisplayName','Incorrect');
            xlabel('Trial'); ylabel('Outcome');
            ylim([-1.5 1.5])
            title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} 'sess' num2str(sessIdx)])
            filename = [dirs.behaviorfigdir 'trialbytrialperformance_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx)) 'sess' num2str(sessIdx)];
            saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
            
            %right vs. left
            figure; hold on;
            turndirs = {'Right','Left'}; turnmarkers = {'+','o'};
            for turnIdx = 1:2
                outcomes = allsessdata(animals(anIdx)).(track).(['sessOutcomes' turndirs{turnIdx}]){sessIdx};                
                plot(find(outcomes == 1), ones(length(find(outcomes == 1))),'g*','LineWidth',2,'DisplayName','Correct');
                plot(find(outcomes == -1), zeros(length(find(outcomes == -1))),'k*','LineWidth',2,'DisplayName','Failed');
                plot(find(outcomes == 0), [zeros(length(find(outcomes == 0)))-1],'r*','LineWidth',2,'DisplayName','Incorrect');
                xlabel('Trial'); ylabel('Outcome');
                ylim([-1.5 1.5])
                title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} 'sess' num2str(sessIdx)])
                filename = [dirs.behaviorfigdir 'trialbytrialperformance_righvsleft_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx)) 'sess' num2str(sessIdx)];
                saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
            end
        end
        
        %all trials separated by right vs. left turns
        figure; hold on;
        turndirs = {'Right','Left'}; turnmarkers = {'+','o'};
        for turnIdx = 1:2
            outcomes = allsessdata(animals(anIdx)).(track).(['sessOutcomes' turndirs{turnIdx} 'All']);
            h1 = plot(find(outcomes == 1), ones(length(find(outcomes == 1))),['g' turnmarkers{turnIdx}],'LineWidth',2,'DisplayName','Correct');
            h2 = plot(find(outcomes == -1), zeros(length(find(outcomes == -1))),['k' turnmarkers{turnIdx}],'LineWidth',2,'DisplayName','Failed');
            h3 = plot(find(outcomes == 0), [zeros(length(find(outcomes == 0)))-1],['r' turnmarkers{turnIdx}],'LineWidth',2,'DisplayName','Incorrect');
        end        
        xlabel('Trial'); ylabel('Outcome');
        ylim([-1.5 1.5])
        title(['S' num2str(animals(anIdx)) ' performance on ' trainingoptions{trackIdx} ' track + = right, o = left'])
        filename = [dirs.behaviorfigdir 'trialbytrialperformance_rightvsleft' trainingoptions{trackIdx} '_S' num2str(animals(anIdx)) 'all'];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
    end
end

%% plot trial duration
for anIdx = 1:length(animals)
    for trackIdx = 1:length(trainingoptions)
        track = trainingoptions{trackIdx};
        numSessions = length(allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).perCorrect);
        numTrials = allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).numTrials;
        
        % all durations
        figure; hold on;
        plot(1:sum(numTrials),allsessdata(animals(anIdx)).(trainingoptions{trackIdx}).trialDur,'b-','LineWidth',2);
        xlabel('Trial'); ylabel('Duration (s)');
        title(['S' num2str(animals(anIdx)) ' trial duration on ' trainingoptions{trackIdx} ' track '])
        filename = [dirs.behaviorfigdir 'trialdur_' trainingoptions{trackIdx} '_S' num2str(animals(anIdx))];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        % durations separated by correct, incorrect, failed 
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
            pause
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



% figure; hold on; 
% for i = 1:size(trialdata,1)
%     phaseInds = trialdata{i}.phaseInds;
%     clr = 'rgbk';
%     for j = 1:size(trialdata{i}.phaseInds,1)
%         plot(trialdata{i}.positionX(phaseInds(j,1):phaseInds(j,2)),trialdata{i}.positionY(phaseInds(j,1):phaseInds(j,2)),['o' clr(j)]); 
%         pause; 
%     end
% end
% 
% figure; hold on;
% for trialIdx = 1:size(trialStarts,1); 
%     phaseInds = behaviorDataDiamondByTrial{trialIdx}.phaseInds;
%     plot(behaviorDataDiamondByTrial{trialIdx}.positionX(phaseInds(1,1)+1:phaseInds(1,2)),behaviorDataDiamondByTrial{trialIdx}.positionY(phaseInds(1,1)+1:phaseInds(1,2)));
%     pause
% end

%% plot view angle traces




