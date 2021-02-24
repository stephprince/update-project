function [histXTable, histYTable] = calcPositionHists(positionData, binsTable)

for trialIdx = 1:size(positionData,1)
    %get position histograms
    trialTrajectoryData = positionData(trialIdx,:).trajectoryData{1};
    if istable(trialTrajectoryData)
        [n edgX binsUsedX] = histcounts(trialTrajectoryData.xPos,binsTable.xPos);
        [n edg binsUsedY] = histcounts(trialTrajectoryData.yPos,binsTable.yPos);
        uniquebinsX = unique(binsUsedX); uniquebinsY = unique(binsUsedY);
        histX = cell(size(binsTable,1),size(trialTrajectoryData,2)); histY = cell(size(binsTable,1),size(trialTrajectoryData,2));
        histX(:) = {nan}; histY(:) = {nan};
        
        %loop through x and y bins to get all values for each position bin
        for binIdx = 1:size(uniquebinsX,1)
            bins2use = find(binsUsedX == uniquebinsX(binIdx));
            for varIdx = 1:size(trialTrajectoryData,2)
                vals2add = trialTrajectoryData(bins2use,varIdx);
                histX{uniquebinsX(binIdx),varIdx} = table2array(vals2add);
            end
        end
        histOutput.histX{trialIdx} = array2table(histX,'VariableNames',trialTrajectoryData.Properties.VariableNames);
        meanValsX = varfun(@(x) cellfun(@(y) nanmean(y), x, 'UniformOutput', 0), histOutput.histX{trialIdx});
        meanValsX.Properties.VariableNames = regexprep(meanValsX.Properties.VariableNames, 'Fun_', 'mean');
        histOutput.histX{trialIdx} = [histOutput.histX{trialIdx}, meanValsX];
        
        for binIdx = 1:size(uniquebinsY,1)
            bins2use = find(binsUsedY == uniquebinsY(binIdx));
            for varIdx = 1:size(trialTrajectoryData,2)
                vals2add = trialTrajectoryData(bins2use,varIdx);
                histY{uniquebinsY(binIdx),varIdx} = table2array(vals2add);
            end
        end
        histOutput.histY{trialIdx} = array2table(histY,'VariableNames',trialTrajectoryData.Properties.VariableNames);
        meanValsY = varfun(@(x) cellfun(@(y) nanmean(y), x, 'UniformOutput', 0), histOutput.histY{trialIdx});
        meanValsY.Properties.VariableNames = regexprep(meanValsY.Properties.VariableNames, 'Fun_', 'mean');
        histOutput.histY{trialIdx} = [histOutput.histY{trialIdx}, meanValsY];
    else
        histOutput.histY{trialIdx} = nan;
        histOutput.histX{trialIdx} = nan;
    end
end
histYTable = array2table(histOutput.histY','VariableNames',{'histY'});
histXTable = array2table(histOutput.histX','VariableNames',{'histX'});
