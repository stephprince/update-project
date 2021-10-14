function plotDiamondTrackMetricsByTrial(trackdata,animal,track,dirs,plotInfo,metricToPlot)
%SP 191130

metricToPlot = 'sessOutcomes';
params.trialBlockSize = [5, 10, 15, 20];
stepsize = 0.2;

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

for blockIdx = 1:length(params.trialBlockSize)
  blockSize = params.trialBlockSize(blockIdx);

  %% loop through fieldnames to bin the data
  %concatenate all the binned data into trial blocks
  perCorrectPerBlock = []; perCorrectPerBlockRight = []; perCorrectPerBlockLeft = []; perCorrectTrialBlockDistBySess = []; sessGroupVect = [];
  for sessIdx = 1:max(trackdata.sessInfo(:,4))
    trials2Bin = trackdata.sessOutcomesAll(find(trackdata.sessInfo(:,4) == sessIdx));
    trialSubType = trackdata.turnDirEncAll(find(trackdata.sessInfo(:,4) == sessIdx));
    trials2BinLeft = trials2Bin(trialSubType == 1);
    trials2BinRight = trials2Bin(trialSubType == 2);
    if length(find(trackdata.sessInfo(:,4) == sessIdx)) >= blockSize
      trialBins = 1:blockSize:length(find(trackdata.sessInfo(:,4) == sessIdx));
      trialBinsLeft = 1:blockSize:length(trials2BinLeft);
      trialBinsRight = 1:blockSize:length(trials2BinRight);

      %loop through bins to calc percent correct
      perCorrectPerSess{sessIdx} = []; perCorrectPerSessRight{sessIdx} = []; perCorrectPerSessLeft{sessIdx} = [];
      for binIdx = 1:length(trialBins)-1
        trialBlockOutcomes = trials2Bin(trialBins(binIdx):trialBins(binIdx+1));
        perCorrectTemp = sum(trialBlockOutcomes == 1)/length(trialBlockOutcomes);
        perCorrectPerBlock = [perCorrectPerBlock; perCorrectTemp];
        perCorrectPerSess{sessIdx} = [perCorrectPerSess{sessIdx}; perCorrectTemp];
      end
      for binIdx = 1:length(trialBinsRight)-1
        trialBlockOutcomesRight = trials2BinRight(trialBinsRight(binIdx):trialBinsRight(binIdx+1));
        perCorrectTempRight = sum(trialBlockOutcomesRight == 1)/length(trialBlockOutcomesRight);
        perCorrectPerBlockRight = [perCorrectPerBlockRight; perCorrectTempRight];
        perCorrectPerSessRight{sessIdx} = [perCorrectPerSessRight{sessIdx}; perCorrectTemp];
      end
      for binIdx = 1:length(trialBinsLeft)-1
        trialBlockOutcomesLeft = trials2BinLeft(trialBinsLeft(binIdx):trialBinsLeft(binIdx+1));
        perCorrectTempLeft = sum(trialBlockOutcomesLeft == 1)/length(trialBlockOutcomesLeft);
        perCorrectPerBlockLeft = [perCorrectPerBlockLeft; perCorrectTempLeft];
        perCorrectPerSessLeft{sessIdx} = [perCorrectPerSessLeft{sessIdx}; perCorrectTemp];
      end

      %get distribution of trial block percent correct for the session
      perCorrectTrialBlockDistBySess = [perCorrectTrialBlockDistBySess; perCorrectPerSess{sessIdx}];
      sessGroupVect = [sessGroupVect; ones(size(perCorrectPerSess{sessIdx}))*sessIdx];
    else
      perCorrectTrialBlockDistBySess = [perCorrectTrialBlockDistBySess; nan];
      sessGroupVect = [sessGroupVect; sessIdx];
    end
  end

  %plot boxplot distribution
  % if sum(~ismember([1:max(trackdata.sessInfo(:,4))],sessGroupVect))
  %     missingval = find(~ismember([1:max(trackdata.sessInfo(:,4))],sessGroupVect));
  %     trialDurs = [trialDurs; nan(size(missingval))']; groupVect = [groupVect; missingval'];
  % end
  figure; hold on;
  boxplot(perCorrectTrialBlockDistBySess,sessGroupVect); hold on;
  ylabel('percent correct trial blocks'); set(gca,'tickdir','out');
  xlabel('Session'); ylim([0 1.01])
  title(['S' num2str(animal) ' performance on ' track ' track']);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialblocks_boxplotdist'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %get the distribution of percentages
  edges = 0:stepsize:1; plotedges = edges(2:end)-stepsize*0.5;
  perCorrectPerBlockHist = histcounts(perCorrectPerBlock, edges);
  perCorrectPerBlockHistNorm = perCorrectPerBlockHist/sum(perCorrectPerBlockHist);
  perCorrectPerBlockHistLeft = histcounts(perCorrectPerBlockLeft, edges);
  perCorrectPerBlockHistLeftNorm = perCorrectPerBlockHistLeft/sum(perCorrectPerBlockHistLeft);
  perCorrectPerBlockHistRight = histcounts(perCorrectPerBlockRight, edges);
  perCorrectPerBlockHistRightNorm = perCorrectPerBlockHistRight/sum(perCorrectPerBlockHistRight);

  %plot the trial block performance over times
  figure; hold on;
  colorToPlot = 'k';
  for i = 1:size(plotInfo.backgroundInfoTrials,1);
    backgroundInfoTrialBlocks = plotInfo.backgroundInfoTrials(i,:)/blockSize;
    fill(backgroundInfoTrialBlocks,plotInfo.fillInfoTrials(i,:),[0.5 0 1],'LineStyle','none','FaceAlpha',0.25);
  end  %show background of single days performance
  plot(1:length(perCorrectPerBlock),perCorrectPerBlock,[colorToPlot '-'],'LineWidth',2);
  xlabel('Trial Block'); ylabel(['Percent Trial Blocks block  = ' num2str(blockSize)]);
  set(gca,'tickdir','out'); ylim([0 1])
  title(['S' num2str(animal) ' trial block performance on ' track ' track - n = ' num2str(size(perCorrectPerBlock,1))]);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialblocks'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %plot the distribution of trial blocks perfcent correctTrials
  figure; hold on;
  colorToPlot = 'k';
  plot(plotedges,perCorrectPerBlockHistNorm,[colorToPlot '-'],'LineWidth',2);
  xlabel('Percent Correct'); ylabel(['Percent Trial Blocks block  = ' num2str(blockSize)]);
  set(gca,'tickdir','out'); ylim([0 1]); xlim([0 1]);
  title(['S' num2str(animal) ' trial block performance on ' track ' track - n = ' num2str(size(perCorrectPerBlock,1))]);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialblocksdist'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %plot the data for right vs. left
  figure; hold on;
  plot(plotedges,perCorrectPerBlockHistRightNorm,'r-','LineWidth',2);
  plot(plotedges,perCorrectPerBlockHistLeftNorm,'b-','LineWidth',2);
  xlabel('Percent Correct'); ylabel(['Percent Trial Blocks block = ' num2str(blockSize)]);
  set(gca,'tickdir','out'); ylim([0 1]); xlim([0 1]);
  title(['S' num2str(animal) ' trial block performance on ' track ' track - n = ' num2str(size(perCorrectPerBlockLeft,1)) ' ' num2str(size(perCorrectPerBlockLeft,1))]);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_'  '_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialblocksdist_rightvsleft'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end
