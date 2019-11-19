function output = plotDiamondTrackCorrectPerformance(trackdata, animal, track, dirs)
%SP 190925 this function plots correct performance of animals with
%different slices of data
%outputs figure data for combining across animals

if ~isempty(trackdata.sessInfo)
    %% initialize variables
    %get session and trial intervals from structure
    trialBlockSize = 5;
    trackInfo = trackdata.sessInfo;
    plotInfo.numSessions = trackInfo(end,4); %last value for sessions counter
    plotInfo.numTrials = sum(trackdata.numTrialsAll);
    plotInfo.daySessionInfo = trackInfo(logical(diff(trackInfo(:,4))),[2,4]); %matrix of sessions for each day
    plotInfo.dayIntervalsByTrial = [[1; find(diff(trackdata.sessInfo(:,2)))+1], [find(diff(trackdata.sessInfo(:,2))); length(trackdata.sessInfo(:,2))]];
    plotInfo.dayIntervalsBySession = [[1; find(diff(plotInfo.daySessionInfo(:,1)))+1], [find(diff(plotInfo.daySessionInfo(:,1))); length(plotInfo.daySessionInfo(:,1))]];
    
    %make background fill structures to indicate day/sessions/trial blocks
    plotInfo.backgroundInfoDays = plotInfo.dayIntervalsBySession(1:2:end,:)+[-0.5 0.5];
    plotInfo.backgroundInfoDays = [plotInfo.backgroundInfoDays, fliplr(plotInfo.backgroundInfoDays)];
    plotInfo.fillInfoDays = repmat([0 0 1.01 1.01],size(plotInfo.backgroundInfoDays,1),1);
    
    %define color mappings
    plotInfo.trialSubtype = {'All','Left','Right','Same','Alt'};
    plotInfo.colorSubtype = 'kbrmg';
    plotInfo.trialPairs = {'Left','Right'; 'Same','Alt'};
    
    %% plot percent correct trials for each session
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'perCorrect');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'perIncorrect');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numCorrect');
    plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,'numTrials');
    
    %% plot number correct trials for each session
    % fieldsToPlot = find(~cellfun(@isempty,strfind(fnames,'numTrials'))); %finds all the fieldname indices in the data structure that match
    % fieldsToPlot2 = find(~cellfun(@isempty,strfind(fnames,'numCorrect'))); %the order of all, right left same matches above so can index through at the same time
    % fillInfoDays = repmat([0 0 1.01*max(trackdata.numTrialsAll) 1.01*max(trackdata.numTrialsAll)],size(backgroundInfoDays,1),1);
    % for fieldIdx = 1:length(fieldsToPlot)
    %     figure; hold on;
    %     colorToPlot = colorSubtype(~cellfun(@isempty, cellfun(@(x) strfind(fnames{fieldsToPlot(fieldIdx)},x), trialSubtype, 'UniformOutput', 0))); %finds which trial subtype matches the fieldname and then grabs that color
    %     for i = 1:size(backgroundInfoDays); fill(backgroundInfoDays(i,:),fillInfoDays(i,:),[0.5 0 1],'LineStyle','none','FaceAlpha',0.25); end  %show background of single days performance
    %     p1 = plot(1:numSessions,trackdata.(fnames{fieldsToPlot(fieldIdx)}),'ko-','LineWidth',2);
    %     p2 = plot(1:numSessions,trackdata.(fnames{fieldsToPlot2(fieldIdx)}),[colorToPlot 'o-'],'LineWidth',2);
    %     xlabel('Session'); ylabel(fnames{fieldsToPlot(fieldIdx)});
    %     set(gca,'tickdir','out'); ylim([0 1.01*max(trackdata.numTrialsAll)]);
    %     legend([p1 p2], 'numTrials','numCorrectTrials')
    %     title(['S' num2str(animal) ' number of trials on ' track ' track - ' fnames{fieldsToPlot(fieldIdx)}]);
    %     filename = [dirs.behaviorfigdir fnames{fieldsToPlot(fieldIdx)} '_' track  '_S' num2str(animal)];
    %     saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    % end
end
    
end