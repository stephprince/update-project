function plotDiamondTrackBoxplotDist(trackdata,animal,track,dirs,metricToPlot)
%SP 190929

if ~isempty(trackdata.durAll)
    
    %define color mappings
    trackInfo = trackdata.sessInfo;
    plotInfo.colors = 'grk';
    plotInfo.trialType = {'Correct','Incorrect','Failed'};
    plotInfo.trialSubtype = {'All','Left','Right','Same','Alt'};
    plotInfo.colorSubtype = 'kbrmg';
    plotInfo.trialPairs = {'Left','Right'; 'Same','Alt'};
    
    %% get fieldnames to plot
    fnames = fieldnames(trackdata);
    fieldsToPlot = find(~cellfun(@isempty,strfind(fnames,metricToPlot))); %finds all the fieldname indices in the data structure that match perCorrect
    
    %% loop through fieldnames to plot different figures
    fieldsToPlot = fieldsToPlot(2:end);
    fieldGroups = 1:3:length(fieldsToPlot);
    for fIdx = 1:length(fieldGroups)
        %compile data in groups of 3 (need to fix later to be better)
        fieldIdx = fieldGroups(fIdx);
        trialDurs = [trackdata.(fnames{fieldsToPlot(fieldIdx)}); trackdata.(fnames{fieldsToPlot(fieldIdx+1)}); trackdata.(fnames{fieldsToPlot(fieldIdx+2)})];
        groupVect = [ones(size(trackdata.(fnames{fieldsToPlot(fieldIdx)}))); ones(size(trackdata.(fnames{fieldsToPlot(fieldIdx+1)})))+1; ones(size(trackdata.(fnames{fieldsToPlot(fieldIdx+2)})))+2];
        if sum(~ismember([1 2 3],groupVect))
            missingval = find(~ismember([1 2 3],groupVect));
            trialDurs = [trialDurs; nan(size(missingval))']; groupVect = [groupVect; missingval'];
        end
        labels = {fnames{fieldsToPlot(fieldIdx)},fnames{fieldsToPlot(fieldIdx+1)},fnames{fieldsToPlot(fieldIdx+2)}};
        
        %plot data
        figure; hold on;
        boxplot(trialDurs,groupVect,'colors','grk','labels',labels);
        ylabel('Duration (s)'); set(gca,'tickdir','out');
        title(['S' num2str(animal) ' performance on ' track ' track - ' fnames{fieldsToPlot(fieldIdx)}]);
        filename = [dirs.behaviorfigdir fnames{fieldsToPlot(fieldIdx)} '_' track  '_S' num2str(animal)];
        saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        
        % plot trial pairs
        trialPairsMatch = ~cellfun(@isempty, cellfun(@(x) strfind(fnames{fieldsToPlot(fieldIdx)},x), plotInfo.trialPairs, 'UniformOutput', 0));
        if sum(sum(trialPairsMatch))
            subtype1 = plotInfo.trialPairs{trialPairsMatch}; subtype2 = plotInfo.trialPairs{fliplr(trialPairsMatch)};
            subtype1Idx = find(~cellfun(@isempty,strfind(fnames(fieldsToPlot),subtype1)));
            subtype2Idx = find(~cellfun(@isempty,strfind(fnames(fieldsToPlot),subtype2)));
            
            %compile data in groups of 3 (need to fix later to be better)
            trialDursType1 = [trackdata.(fnames{fieldsToPlot(subtype1Idx(1))}); trackdata.(fnames{fieldsToPlot(subtype1Idx(2))}); trackdata.(fnames{fieldsToPlot(subtype1Idx(3))})];
            groupVectType1 = [ones(size(trackdata.(fnames{fieldsToPlot(subtype1Idx(1))}))); ones(size(trackdata.(fnames{fieldsToPlot(subtype1Idx(2))})))+1; ones(size(trackdata.(fnames{fieldsToPlot(subtype1Idx(3))})))+2];
            trialDursType2 = [trackdata.(fnames{fieldsToPlot(subtype2Idx(1))}); trackdata.(fnames{fieldsToPlot(subtype2Idx(2))}); trackdata.(fnames{fieldsToPlot(subtype2Idx(3))})];
            groupVectType2 = [ones(size(trackdata.(fnames{fieldsToPlot(subtype2Idx(1))}))); ones(size(trackdata.(fnames{fieldsToPlot(subtype2Idx(2))})))+1; ones(size(trackdata.(fnames{fieldsToPlot(subtype2Idx(3))})))+2];
            trialDursBoth = [trialDursType1; trialDursType2];
            groupVectBoth = [groupVectType1; groupVectType2+3];
            
            if sum(~ismember([1 2 3 4 5 6],groupVectBoth))
                missingval = find(~ismember([1 2 3 4 5 6],groupVectBoth));
                trialDursBoth = [trialDursBoth; nan(size(missingval))']; groupVectBoth = [groupVectBoth; missingval'];
            end
            labelsBoth = {fnames{fieldsToPlot(subtype1Idx(1))},fnames{fieldsToPlot(subtype1Idx(2))},fnames{fieldsToPlot(subtype1Idx(3))},fnames{fieldsToPlot(subtype2Idx(1))},fnames{fieldsToPlot(subtype2Idx(2))},fnames{fieldsToPlot(subtype2Idx(3))}};
            
            %plot the data
            figure('units','normalized','outerposition',[0 0 1 1]); hold on;
            boxplot(trialDursBoth,groupVectBoth,'colors','grkgrk','labels',labelsBoth);
            ylabel('Duration (s)'); set(gca,'tickdir','out');
            title(['S' num2str(animal) ' performance on ' track ' track - ' fnames{fieldsToPlot(subtype1Idx)} 'vs' fnames{fieldsToPlot(subtype2Idx)}]);
            filename = [dirs.behaviorfigdir fnames{fieldsToPlot(subtype1Idx)} 'vs' fnames{fieldsToPlot(subtype2Idx)} '_' track  '_S' num2str(animal)];
            saveas(gcf,filename,'png'); saveas(gcf,filename,'fig');
        end
    end
end