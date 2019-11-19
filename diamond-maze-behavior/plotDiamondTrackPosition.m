function plotDiamondTrackPosition(trackdata, animal, track, dirs);

%% initialize variables
trialBlockSize = 5;
trackInfo = trackdata.sessInfo;
plotInfo.trialSubtype = {'All','Left','Right','Same','Alt'};
plotInfo.colorSubtype = 'kbrmg';
plotInfo.trialPairs = {'Left','Right'; 'Same','Alt'};
    
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
end