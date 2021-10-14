function output = plotContinuousAltPerformance(trackdata, animal, track, dirs)
%SP 190925 this function plots correct performance of animals with
%different slices of data
%outputs figure data for combining across animals

if ~isempty(trackdata.sessInfo)
    %% initialize variables
    %get session and trial intervals from structure
    plotInfo = getDiamondTrackPlotInfo(trackdata, animal, track, dirs);
    trackInfo = trackdata.sessInfo;

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
      plot(1:length(sessions),avgTrialsSinceLastCorrect,[colorToPlot 'o-'],'LineWidth',2);
      plot(1:length(sessions),repmat(0.5,length(sessions),1),'k--','LineWidth',2);
      xlabel('Session'); ylabel('avgTrialsSinceCorrect');
      set(gca,'tickdir','out'); ylim([0 1.01*max(m)]);
      title(['S' num2str(animal) ' performance on ' track ' track - avgTrialsSinceLastCorrect']);
      filename = [dirs.behaviorfigdir 'avgTrialsSinceLastCorrect_' track  '_S' num2str(animal)];
      saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end

end
