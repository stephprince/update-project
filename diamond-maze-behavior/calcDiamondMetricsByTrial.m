function behaviorDataDiamondByTrial = calcDiamondMetricsByTrial(sessdata, params, dirs, index, animalID, makenewfiles)
%this function splits up diamond track behavioral data into individual trials

%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

% it perfoms the following actions in this order
%       gets trial start/ends based on reward/punishment phase or large position changes
%       creates a new data structure where each trial is a cell in a cell array
%       gets trialdur
%       resamples vectors so constant time sampling rate (corrects stepfunction vectors to remain step functions as well)
%       gets incorrect/correct, left/right, north/south trial info


filename = [dirs.savedatadir 'behaviorDataDiamondTrial_' animalID num2str(index(1)) '_' num2str(index(2)) '_' num2str(index(3))];
disp(['Calculating trial metrics for ' animalID num2str(index(1)) ' ' num2str(index(2)) ' ' num2str(index(3))]);

if ~exist(filename) || makenewfiles
    %% get trial times
    trialStarts = [1; find(diff(ismember(sessdata.currentPhase,[3 4])) == -1) + 1]; %looks for when reward or punishment phase happened and finds the end
    trialEnds = [find(diff(ismember(sessdata.currentPhase,[3 4])) == -1); length(sessdata.currentPhase)];
    if ~ismember(sessdata.currentPhase,[3,4]) %failed trial or linear track, in this case look for large changes in position
        trialStarts = [1; find(abs(diff(sessdata.positionY)) > 10) + 1]; %10 is arbitraty number but seems to work for catching the teleportation events
        trialEnds = [find(abs(diff(sessdata.positionY)) > 10); length(sessdata.positionY)];
    end

    %% make new trial data structure
    fnames = fieldnames(sessdata);
    fields = fnames(4:end); %get rid of the trackname, params, session info fields that weren't matrices
    for trialIdx = 1:size(trialStarts,1)
        for fieldIdx = 1:size(fields,1)
            trialInds = [trialStarts(trialIdx) trialEnds(trialIdx)];
            if ~isnan(sessdata.(fields{fieldIdx}))
                behaviorDataDiamondByTrial{trialIdx}.(fields{fieldIdx}) = sessdata.(fields{fieldIdx})(trialInds(1):trialInds(2));
            else
                behaviorDataDiamondByTrial{trialIdx}.(fields{fieldIdx}) = nan;
            end
        end
    end

    %% get trial duration
    for trialIdx = 1:size(trialStarts,1)
        timeVect = behaviorDataDiamondByTrial{trialIdx}.time;
        behaviorDataDiamondByTrial{trialIdx}.trialdur = timeVect(end)-timeVect(1); %duration in sec
    end

    %% resample trial data so constant time sampling rate
    for trialIdx = 1:size(trialStarts,1)
        %gets fields to resample (vectors of behavioral data)
        numSamples = length(behaviorDataDiamondByTrial{trialIdx}.time);
        fnames = fieldnames(behaviorDataDiamondByTrial{trialIdx});
        counter = 1;
        for i = 1:length(fnames)
            if size(behaviorDataDiamondByTrial{trialIdx}.(fnames{i}),1) == numSamples
                fnames2resample{counter} = fnames{i};
                counter = counter + 1;
            end
        end

        %sets time window for resampling to constant sampling rate (in time)
        newTrialSampSize = round(behaviorDataDiamondByTrial{trialIdx}.trialdur/params.constSampRateTime); %find out how many samples should be in new vect
        newTimes = behaviorDataDiamondByTrial{trialIdx}.time(1):params.constSampRateTime:((newTrialSampSize*params.constSampRateTime)+behaviorDataDiamondByTrial{trialIdx}.time(1)); %get times from start to end with const time window
        for i = 1:length(fnames2resample)
            resampledVect = interp1(behaviorDataDiamondByTrial{trialIdx}.time,behaviorDataDiamondByTrial{trialIdx}.(fnames2resample{i}),newTimes,'linear','extrap'); %added extrap to get rid of accidental nan values in the data
            behaviorDataDiamondByTrial{trialIdx}.([fnames2resample{i} 'ConstTime']) = resampledVect;
        end

        %for incremental vectors (where values go up in steps need to resample to know where these steps are)
        fnames2findsteppoints = {'numRewards','numLicks','currentZone','correctZone','currentWorld','currentPhase''numTrials'};
        for i = 1:length(fnames2findsteppoints) %replace resampled vectors with 0 and resubstitute switch indices as ones
              %use old indices/switch times to find new ones
              oldInds = find(diff(behaviorDataDiamondByTrial{trialIdx}.(fnames2findsteppoints{i})))+1;
              oldTimes = behaviorDataDiamondByTrial{trialIdx}.time(oldInds);
              newInds = lookup2(oldTimes,behaviorDataDiamondByTrial{trialIdx}.timeConstTime);
              %make new vector where new Inds are indicated with ones and the rest of the vector is zeros
              behaviorDataDiamondByTrial{trialIdx}.([fnames2findsteppoints{i} 'ConstTime']) = zeros(size(behaviorDataDiamondByTrial{trialIdx}.([fnames2findsteppoints{i} 'ConstTime'])));
              behaviorDataDiamondByTrial{trialIdx}.([fnames2findsteppoints{i} 'ConstTime'])(newInds) = 1; %replace switch indices time points with 1
              behaviorDataDiamondByTrial{trialIdx}.([fnames2findsteppoints{i} 'IndsConstTime']) = newInds;
        end

        %for current phase, get start and end indices
        phaseStarts = [1; behaviorDataDiamondByTrial{trialIdx}.currentPhaseIndsConstTime)];
        phaseEnds = [behaviorDataDiamondByTrial{trialIdx}.currentPhaseIndsConstTime)-1; length(behaviorDataDiamondByTrial{trialIdx}.currentPhaseConstTime)];
        phaseStartsEnds = [phaseStarts phaseEnds];
        behaviorDataDiamondByTrial{trialIdx}.phaseStartsEndsConstTime = phaseStartsEnds;
        behaviorDataDiamondByTrial{trialIdx}.worldByPhaseConstTime = behaviorDataDiamondByTrial{trialIdx}.currentWorldConstTime(phaseStartsEnds);
        behaviorDataDiamondByTrial{trialIdx}.phaseTypeConstTime = behaviorDataDiamondByTrial{trialIdx}.currentPhaseConstTime(phaseStartsEnds);
    end

    %% get incorrect/correct trial %0 = incorrect, 1 = correct, -1 = failed
    for trialIdx = 1:size(trialStarts,1)
        behaviorDataDiamondByTrial{trialIdx}.outcome = getTrialOutcomes(sessdata, behaviorDataDiamondByTrial{trialIdx});
    end

    %% get north or south trial
    for trialIdx = 1:size(trialStarts,1)
        [startLoc choiceLoc] = getTrialStartLoc(behaviorDataDiamondByTrial{trialIdx});
        behaviorDataDiamondByTrial{trialIdx}.trialStartLoc = startLoc;
        behaviorDataDiamondByTrial{trialIdx}.trialChoiceLoc = choiceLoc;
    end

    %% get right or left trial
    for trialIdx = 1:size(trialStarts,1)
        %note the first correct zone value will be correct side for encoding only (would switch for choice)
        if behaviorDataDiamondByTrial{trialIdx}.correctZoneConstTime(1) == 2
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectEncLoc = 'east';
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectChoiceLoc = 'west';
        elseif behaviorDataDiamondByTrial{trialIdx}.correctZoneConstTime(1) == 1
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectEncLoc = 'west';
            behaviorDataDiamondByTrial{trialIdx}.trialCorrectChoiceLoc = 'east';
        end
    end

    %% save file
    save(filename, 'behaviorDataDiamondByTrial');

else
    load(filename);
end
