function output = plotContinuousAltPerformance(trackdata, animal, track, dirs)
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

    %% plot trials since last correct for each session
    sessions = unique(trackdata.sessInfo(:,2:3),'rows');
    for sessIdx = 1:length(sessions)
      trials2avg = ismember(trackdata.trialsSinceCorrectIDAll(:,2:3),sessions(sessIdx,:),'rows');
      avgTrialsSinceLastCorrect(sessIdx) = nanmean(trackdata.trialsSinceLastAll(trials2avg));
    end
    m = max(avgTrialsSinceLastCorrect);
    plotInfo.fillInfoDays = repmat([0 0 1.01*max(m) 1.01*max(m)],size(plotInfo.backgroundInfoDays,1),1);
    plotInfo.trialSubtype = 'All';

    % plot individually
    figure; hold on;
    colorToPlot = 'k';
    for i = 1:size(plotInfo.backgroundInfoDays)
      fill(plotInfo.backgroundInfoDays(i,:),plotInfo.fillInfoDays(i,:),[0.5 0 1],'LineStyle','none','FaceAlpha',0.25); end  %show background of single days performance
      plot(1:plotInfo.numSessions,avgTrialsSinceLastCorrect,[colorToPlot 'o-'],'LineWidth',2);
      plot(1:plotInfo.numSessions,repmat(0.5,plotInfo.numSessions,1),'k--','LineWidth',2);
      xlabel('Session'); ylabel('avgTrialsSinceCorrect');
      set(gca,'tickdir','out'); ylim([0 1.01*max(m)]);
      title(['S' num2str(animal) ' performance on ' track ' track - avgTrialsSinceLastCorrect']);
      filename = [dirs.behaviorfigdir 'avgTrialsSinceLastCorrect_' track  '_S' num2str(animal)];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end

end
