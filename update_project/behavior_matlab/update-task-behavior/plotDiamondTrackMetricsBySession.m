function plotDiamondTrackMetricsBySession(trackdata,animal,track,dirs,plotInfo,metricToPlot)
%SP 190929

%% get fieldnames to plot
fnames = fieldnames(trackdata);
fieldsToPlot = find(~cellfun(@isempty,strfind(fnames,metricToPlot))); %finds all the fieldname indices in the data structure that match perCorrect
m = [1];
for fIdx = 1:length(fieldsToPlot);
  if ~iscell(trackdata.(fnames{fieldsToPlot(fIdx)}))
    m = [m; max(trackdata.(fnames{fieldsToPlot(fIdx)}))];
  end
end
plotInfo.fillInfoDays = repmat([0 0 1.01*max(m) 1.01*max(m)],size(plotInfo.backgroundInfoDays,1),1);

%% loop through fieldnames to plot different figures
for fieldIdx = 1:length(fieldsToPlot)
  if ~iscell(trackdata.(fnames{fieldsToPlot(fieldIdx)}))
    % plot individually
    figure; hold on;
    colorToPlot = plotInfo.colorSubtype(~cellfun(@isempty, cellfun(@(x) strfind(fnames{fieldsToPlot(fieldIdx)},x), plotInfo.trialSubtype, 'UniformOutput', 0))); %finds which trial subtype matches the fieldname and then grabs that color
    for i = 1:size(plotInfo.backgroundInfoDays); fill(plotInfo.backgroundInfoDays(i,:),plotInfo.fillInfoDays(i,:),[0.5 0 1],'LineStyle','none','FaceAlpha',0.25); end  %show background of single days performance
    plot(1:plotInfo.numSessions,trackdata.(fnames{fieldsToPlot(fieldIdx)}),[colorToPlot 'o-'],'LineWidth',2);
    plot(1:plotInfo.numSessions,repmat(0.5,plotInfo.numSessions,1),'k--','LineWidth',2);
    xlabel('Session'); ylabel(fnames{fieldsToPlot(fieldIdx)});
    set(gca,'tickdir','out'); ylim([0 1.01*max(m)]);
    title(['S' num2str(animal) ' performance on ' track ' track - ' fnames{fieldsToPlot(fieldIdx)}]);
    filename = [dirs.behaviorfigdir fnames{fieldsToPlot(fieldIdx)} '_' track  '_S' num2str(animal)];
    saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

    % plot trial pairs
    trialPairsMatch = ~cellfun(@isempty, cellfun(@(x) strfind(fnames{fieldsToPlot(fieldIdx)},x), plotInfo.trialPairs, 'UniformOutput', 0));
    if sum(sum(trialPairsMatch))
        subtype1 = plotInfo.trialPairs{trialPairsMatch}; subtype2 = plotInfo.trialPairs{fliplr(trialPairsMatch)};
        subtype1Idx = find(~cellfun(@isempty,strfind(fnames(fieldsToPlot),subtype1)));
        subtype2Idx = find(~cellfun(@isempty,strfind(fnames(fieldsToPlot),subtype2)));

        figure; hold on;
        colorToPlot1 = plotInfo.colorSubtype(~cellfun(@isempty, cellfun(@(x) strfind(fnames{fieldsToPlot(subtype1Idx)},x), plotInfo.trialSubtype, 'UniformOutput', 0))); %finds which trial subtype matches the fieldname and then grabs that color
        colorToPlot2 = plotInfo.colorSubtype(~cellfun(@isempty, cellfun(@(x) strfind(fnames{fieldsToPlot(subtype2Idx)},x), plotInfo.trialSubtype, 'UniformOutput', 0))); %finds which trial subtype matches the fieldname and then grabs that color
        for i = 1:size(plotInfo.backgroundInfoDays); fill(plotInfo.backgroundInfoDays(i,:),plotInfo.fillInfoDays(i,:),[0.5 0 1],'LineStyle','none','FaceAlpha',0.25); end  %show background of single days performance
        plot(1:plotInfo.numSessions,trackdata.(fnames{fieldsToPlot(subtype1Idx)}),[colorToPlot1 'o-'],'LineWidth',2);
        plot(1:plotInfo.numSessions,trackdata.(fnames{fieldsToPlot(subtype2Idx)}),[colorToPlot2 'o-'],'LineWidth',2);
        plot(1:plotInfo.numSessions,repmat(0.5,plotInfo.numSessions,1),'k--','LineWidth',2);
        xlabel('Session'); ylabel(fnames{fieldsToPlot(fieldIdx)});
        set(gca,'tickdir','out'); ylim([0 1.01*max(m)]);
        title(['S' num2str(animal) ' performance on ' track ' track - ' fnames{fieldsToPlot(subtype1Idx)} 'vs' fnames{fieldsToPlot(subtype2Idx)}]);
        filename = [dirs.behaviorfigdir fnames{fieldsToPlot(subtype1Idx)} 'vs' fnames{fieldsToPlot(subtype2Idx)} '_' track  '_S' num2str(animal)];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
    end

    close all;
  end
end
