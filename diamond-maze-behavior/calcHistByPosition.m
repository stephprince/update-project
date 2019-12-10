function output = calcHistByPosition(trackDataClean, metric);

  for trialIdx = 1:size(trackDataClean.time,1)
    [n edg binsUsedX] = histcounts(trackDataClean.positionXClean{trialIdx},trackDataClean.posXbins);
    [n edg binsUsedY] = histcounts(trackDataClean.positionYClean{trialIdx},trackDataClean.posYbins);
    uniquebinsX = unique(binsUsedX); uniquebinsY = unique(binsUsedY);
    output.histX(trialIdx,:) = nan(size(trackDataClean.posXbins));
    output.histY(trialIdx,:) = nan(size(trackDataClean.posYbins));
    output.posXbins = trackDataClean.posXbins;
    output.posYbins = trackDataClean.posYbins;
    
    for binIdx = 1:size(uniquebinsX,2)
      bins2avg = find(binsUsedX == uniquebinsX(binIdx));
      metricXAvg = nanmean(trackDataClean.([metric 'Clean']){trialIdx}(bins2avg));
      output.histX(trialIdx,uniquebinsX(binIdx)) = metricXAvg;
    end

    for binIdx = 1:size(uniquebinsY,2)
      bins2avg = find(binsUsedY == uniquebinsY(binIdx));
      metricYAvg = nanmean(trackDataClean.([metric 'Clean']){trialIdx}(bins2avg));
      output.histY(trialIdx,uniquebinsY(binIdx)) = metricYAvg;
    end
  end
end
