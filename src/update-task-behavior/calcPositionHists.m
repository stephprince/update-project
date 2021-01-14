function output = calcPositionHists(trackdata, indices, dirs, params)

%clean the data to get rid of teleportation events for plotting
for anIdx = 1:numel(indices.animals)
    animaldata = trackdata(trackdata.Animal == indices.animals(anIdx),:);
    for worldIdx = 1:4
        %compile all trials for each track across days
        trialRows = cellfun(@(x) find(x.trialWorld == worldIdx), animaldata.trialTable,'UniformOutput',0);
        trialdata = [];
        for trialIdx = 1:numel(trialRows)
            trialdata = [trialdata; animaldata.trialTable{trialIdx,:}(trialRows{trialIdx},:)];
        end
        
        %extract trial periods of interest
        positionData = [];
        for trialIdx = 1:size(trialdata,1)
            virmenData = trialdata(trialIdx,:).resampledTrialData{1};
            patternToFind = ['[^5]5']; %look for any number of intertrial periods and the start of a new trial
            teleportToInterTrial = regexp(sprintf('%i', virmenData.currentWorld), patternToFind)'; %teleport happens at first intertrial phase
            
            if ~isempty(teleportToInterTrial)
                temp.trajectoryData = virmenData(1:teleportToInterTrial,2:6); %all the position related variables
                temp.minVals = varfun(@(x) min(x), temp.trajectoryData);
                temp.maxVals = varfun(@(x) max(x), temp.trajectoryData);
            else
                temp.trajectoryData = nan; %use the old min/max vals
            end
            positionData = [positionData; struct2table(temp,'AsArray',1)];
        end
        
        %get position bins
        minVals = varfun(@(x) min(x), positionData.minVals);
        maxVals = varfun(@(x) max(x), positionData.maxVals);
        newNames = regexprep(minVals.Properties.VariableNames, 'Fun_Fun_', '');
        for valIdx = 1:size(minVals,2)
            binsTemp.(newNames{valIdx}) = linspace(table2array(minVals(1,valIdx)),table2array(maxVals(1,valIdx)),50)';
        end
        binsTable = struct2table(binsTemp);
        
        %concatenate data across all the trials
        for trialIdx = 1:size(positionData,1)
            %get position histograms
            trialTrajectoryData = positionData(trialIdx,:).trajectoryData{1};
            if istable(trialTrajectoryData)
                [n edg binsUsedX] = histcounts(trialTrajectoryData.xPos,binsTable.xPos);
                [n edg binsUsedY] = histcounts(trialTrajectoryData.yPos,binsTable.yPos);
                uniquebinsX = unique(binsUsedX); uniquebinsY = unique(binsUsedY);

                %loop through x and y bins to get all values for each position bin
                for binIdx = 1:size(uniquebinsX,1)
                    bins2use = find(binsUsedX == uniquebinsX(binIdx));
                    for varIdx = 1:size(trialTrajectoryData,2)
                        vals2add = trialTrajectoryData{bins2use,varIdx};
                        histX{uniquebinsX(binIdx),varIdx} = vals2add;
                    end
                end
                histOutput.histX{trialIdx} = array2table(histX,'VariableNames',trialTrajectoryData.Properties.VariableNames);
                
                for binIdx = 1:size(uniquebinsY,1)
                    bins2use = find(binsUsedY == uniquebinsY(binIdx));
                    for varIdx = 1:size(trialTrajectoryData,2)
                        vals2add = trialTrajectoryData{bins2use,varIdx};
                        histY{uniquebinsY(binIdx),varIdx} = vals2add;
                    end
                end
                histOutput.histY{trialIdx} = array2table(histY,'VariableNames',trialTrajectoryData.Properties.VariableNames);
            else
                histOutput.histY{trialIdx} = nan;
                histOutput.histX{trialIdx} = nan;
            end
        end
        histYTable = array2table(histOutput.histY','VariableNames',{'histY'});
        histXTable = array2table(histOutput.histY','VariableNames',{'histX'});
        positionData = [positionData, histXTable, histYTable];
        
        %compile data for each world/animal
        trialOutcome = {'correct','incorrect'}; trialTurn = {'right','left'}; 
        for turnIdx = 1:2
            for outIdx = 1:2
                
                
                  trialsFromDelayType = cellfun(@(x) find(round(x.trialDelayLocation) == params.plotCategories(paramIdx,2)), animaldata.trialTable,'UniformOutput',0);
        trialsFromUpdateType = cellfun(@(x) find(round(x.trialTypesUpdate) == params.plotCategories(paramIdx,3)), animaldata.trialTable,'UniformOutput',0);
        
        
                rightTrialOutcomes = trialdata.trialOutcomes; leftTrialOutcomes = trialdata.trialOutcomes;
        rightTrialOutcomes(trialdata.trialTypesLeftRight == params.trialTypeMap('left')) = nan; %replace with nans so lengths of vectors are equal
        leftTrialOutcomes(trialdata.trialTypesLeftRight == params.trialTypeMap('right')) = nan;
        perCorrect{anIdx}{paramIdx} = movmean(trialdata.trialOutcomes,params.trialBlockSize,'omitnan');
    
            end
        end
    end
end


fnames = fieldnames(hists2plot);

for metricIdx = 1:size(metrics,2)
  for axIdx = 1:2
    for turnIdx = 1:2
      for outIdx = 1:2
        %get data and trial types to look at
        metric2plot = metrics{metricIdx};
        trialtype1 = trialtypes.([trialOutcome{outIdx} 'Trials']);
        trialtype2 = trialtypes.([trialTurn{turnIdx} 'Trials']);
        metricHist = hists2plot.([metric2plot 'Hists']).(['hist' trialPlottype{axIdx}]);
        metricVals = hists2plot.([metric2plot 'Hists']).(['allvals' trialPlottype{axIdx}]);

        %get histogram info for specific trial subtypes
        metricHistTrialType = metricHist(intersect(trialtype1, trialtype2),:);
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).all = metricHistTrialType;
        posBinsAll = hists2plot.([metric2plot 'Hists']).(['pos' trialPlottype{axIdx} 'bins']);
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posBinsAll = posBinsAll;
        posBinsRaw = posBinsAll;

        %get average, sem for the different trial types
        tempmean = nanmean(metricHistTrialType);
        tempsem = nanstd(metricHistTrialType)/sqrt(size(metricHistTrialType,1));
        numTrials = size(metricHistTrialType,1);

        %get all values for specific trial subtypes
        trials2use = intersect(trialtype1, trialtype2);
        counter = 1;
        for trialIdx = 1:length(trials2use)
          metricAllValsTrialType{counter} = metricVals{trials2use(trialIdx)};
          counter = counter + 1;
        end
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).allvalsperbin = metricAllValsTrialType;

        %clean the data from nans and fill structure
        posBinsAll(isnan(tempmean)) = []; tempmean(isnan(tempmean)) = []; tempsem(isnan(tempsem)) = [];
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).avg =  tempmean;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).sem = tempsem;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posBinsRaw = posBinsRaw;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).posBins = posBinsAll;
        output.(metric2plot).(trialPlottype{axIdx}).(trialTurn{turnIdx}).(trialOutcome{outIdx}).numTrials = numTrials;
      end
    end
  end
end

