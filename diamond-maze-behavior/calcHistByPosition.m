function output = calcHistByPosition(trackDataClean, metric);

  for trialIdx = 1:size(trackDataClean.time,1)
    %get position histograms
    [n edg binsUsedX] = histcounts(trackDataClean.positionXClean{trialIdx},trackDataClean.posXbins);
    [n edg binsUsedY] = histcounts(trackDataClean.positionYClean{trialIdx},trackDataClean.posYbins);
    uniquebinsX = unique(binsUsedX); uniquebinsY = unique(binsUsedY);
    output.posXbins = trackDataClean.posXbins;
    output.posYbins = trackDataClean.posYbins;

    %initialize data structures
    output.histX(trialIdx,:) = nan(size(trackDataClean.posXbins));
    output.histY(trialIdx,:) = nan(size(trackDataClean.posYbins));
    for binIdx = 1:size(trackDataClean.posXbins,2)
      output.allvalsX{trialIdx}{binIdx} = nan;
    end
    for binIdx = 1:size(trackDataClean.posYbins,2)
      output.allvalsY{trialIdx}{binIdx} = nan;
    end

    %loop through x bins to get histograms by X position
    for binIdx = 1:size(uniquebinsX,2)
      bins2avg = find(binsUsedX == uniquebinsX(binIdx));
      metricXAvg = nanmean(trackDataClean.([metric 'Clean']){trialIdx}(bins2avg));
      output.histX(trialIdx,uniquebinsX(binIdx)) = metricXAvg;
      output.allvalsX{trialIdx}{uniquebinsX(binIdx)} = trackDataClean.([metric 'Clean']){trialIdx}(bins2avg);
    end

    %loop through y bins to get histograms by X position
    for binIdx = 1:size(uniquebinsY,2)
      bins2avg = find(binsUsedY == uniquebinsY(binIdx));
      metricYAvg = nanmean(trackDataClean.([metric 'Clean']){trialIdx}(bins2avg));
      output.histY(trialIdx,uniquebinsY(binIdx)) = metricYAvg;
      output.allvalsY{trialIdx}{uniquebinsY(binIdx)} = trackDataClean.([metric 'Clean']){trialIdx}(bins2avg);
    end
  end
end
