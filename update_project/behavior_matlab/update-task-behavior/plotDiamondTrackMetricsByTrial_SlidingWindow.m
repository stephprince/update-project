function plotDiamondTrackMetricsByTrial_SlidingWindow(trackdata,animal,track,dirs,plotInfo,metricToPlot)
%SP 191130

metricToPlot = 'sessOutcomes';
params.trialBlockSize = [20, 30, 40, 50, 60];
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
    numTrialsAll =  length(find(trackdata.sessInfo(:,4) == sessIdx));
    numTrialsLeft = length(trials2BinLeft); numTrialsRight = length(trials2BinRight);
    if length(find(trackdata.sessInfo(:,4) == sessIdx)) >= blockSize
      %loop through bins to calc percent correct
      perCorrectPerSess{sessIdx} = []; perCorrectPerSessRight{sessIdx} = []; perCorrectPerSessLeft{sessIdx} = [];
      for iterIdx = 1:(numTrialsAll-blockSize+1)
        trialBins = 1*iterIdx:1*iterIdx+blockSize-1;
        trialBlockOutcomes = trials2Bin(trialBins);
        perCorrectTemp = sum(trialBlockOutcomes == 1)/length(trialBlockOutcomes);
        perCorrectPerBlock = [perCorrectPerBlock; perCorrectTemp];
        perCorrectPerSess{sessIdx} = [perCorrectPerSess{sessIdx}; perCorrectTemp];
      end
      for iterIdx = 1:(numTrialsRight-blockSize+1)
        trialBins = 1*iterIdx:1*iterIdx+blockSize-1;
        trialBlockOutcomesRight = trials2BinRight(trialBins);
        perCorrectTempRight = sum(trialBlockOutcomesRight == 1)/length(trialBlockOutcomesRight);
        perCorrectPerBlockRight = [perCorrectPerBlockRight; perCorrectTempRight];
        perCorrectPerSessRight{sessIdx} = [perCorrectPerSessRight{sessIdx}; perCorrectTemp];
      end
      for iterIdx = 1:(numTrialsLeft-blockSize+1)
        trialBins = 1*iterIdx:1*iterIdx+blockSize-1;
        trialBlockOutcomesLeft = trials2BinLeft(trialBins);
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

  %for each session plot performance over time
  perCorrectSlidingWindowAll = []; windowIntervals = [];
  subplotSize = 5; numSessions = max(trackdata.sessInfo(:,4));
  iter = 1;
  figure('units','normalized','outerposition',[0 0 1 1]); hold on;
  for sessIdx = 1:max(trackdata.sessInfo(:,4))
    %resets to new figure if more subplots than max
    if iter > subplotSize*subplotSize
      figure; hold on;
      iter = 0;
    end

    %plots the data throughout the sessions
    subplot(subplotSize,subplotSize,iter)
    plot([1 length(perCorrectPerSess{sessIdx})], [0.5 0.5], 'k--'); hold on;
    plot(perCorrectPerSess{sessIdx},'k','LineWidth',2);
    ylabel('percent correct'); set(gca,'tickdir','out');
    xlabel('Window'); ylim([0 1.01])
    title(['Session ' num2str(sessIdx)])
    iter = iter + 1;

    %concatenate the data across numSessions
    windows = ones(length(perCorrectPerSess{sessIdx}),1)+(iter-1);
    perCorrectSlidingWindowAll = [perCorrectSlidingWindowAll; perCorrectPerSess{sessIdx}];
    windowIntervals = [windowIntervals; windows];
  end
  sgtitle(['S' num2str(animal) ' performance on ' track ' track']);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialslidingwindow'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %plot session performance in sliding window for all
  windowIntervalsAll = [[1; find(diff(windowIntervals(:,1)))+1], [find(diff(windowIntervals(:,1))); length(windowIntervals(:,1))]];
  backgroundInfoDays = windowIntervalsAll(1:2:end,:)+[-0.5 0.5];
  backgroundInfoDays = [backgroundInfoDays, fliplr(backgroundInfoDays)];
  fillInfoDays = repmat([0 0 1.01 1.01],size(backgroundInfoDays,1),1);
  figure('units','normalized','outerposition',[0 0 1 1]); hold on;
  for i = 1:size(fillInfoDays);
    fill(backgroundInfoDays(i,:),fillInfoDays(i,:),[0.5 0 1],'LineStyle','none','FaceAlpha',0.25);
  end %show background of single days performance
  perCorrectSlidingWindowAllSmooth = perCorrectSlidingWindowAll;
  plot([1 length(perCorrectSlidingWindowAll)], [0.5 0.5], 'k--'); hold on;
  %plot(perCorrectSlidingWindowAll, 'k', 'LineWidth', 2); hold on;
  plot(perCorrectSlidingWindowAllSmooth, 'k', 'LineWidth', 2);
  ylabel('percent correct'); set(gca,'tickdir','out');
  xlabel('Window'); ylim([0 1.01])
  title(['S' num2str(animal) ' performance on ' track ' track - all sessions with sliding window']);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_ALLtrials'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %plot boxplot distribution
  figure; hold on;
  boxplot(perCorrectTrialBlockDistBySess,sessGroupVect); hold on;
  ylabel('percent correct trial blocks'); set(gca,'tickdir','out');
  xlabel('Session'); ylim([0 1.01])
  title(['S' num2str(animal) ' performance on ' track ' track']);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialslidingwindow_boxplotdist'];
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
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialslidingwindow'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %plot the distribution of trial blocks perfcent correctTrials
  figure; hold on;
  colorToPlot = 'k';
  plot(plotedges,perCorrectPerBlockHistNorm,[colorToPlot '-'],'LineWidth',2);
  xlabel('Percent Correct'); ylabel(['Percent Trial Blocks block  = ' num2str(blockSize)]);
  set(gca,'tickdir','out'); ylim([0 1]); xlim([0 1]);
  title(['S' num2str(animal) ' trial block performance on ' track ' track - n = ' num2str(size(perCorrectPerBlock,1))]);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialslidingwindowdist'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');

  %plot the data for right vs. left
  figure; hold on;
  plot(plotedges,perCorrectPerBlockHistRightNorm,'r-','LineWidth',2);
  plot(plotedges,perCorrectPerBlockHistLeftNorm,'b-','LineWidth',2);
  xlabel('Percent Correct'); ylabel(['Percent Trial Blocks block = ' num2str(blockSize)]);
  set(gca,'tickdir','out'); ylim([0 1]); xlim([0 1]);
  title(['S' num2str(animal) ' trial block performance on ' track ' track - n = ' num2str(size(perCorrectPerBlockLeft,1)) ' ' num2str(size(perCorrectPerBlockLeft,1))]);
  filename = [dirs.behaviorfigdir 'sessPerformanceAll_'  '_' track  '_S' num2str(animal) '_blocksize' num2str(blockSize) '_trialslidingwindowdist_rightvsleft'];
  saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
end
