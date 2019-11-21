function output = resampleDiamondMazeTrials(trialdata,params,tracktype);
% SP 190924
% SP changed on 191119 to move this function to the trial section of the code

%gets fields to resample (vectors of behavioral data)
numSamples = length(trialdata.timeRaw);
fnames = fieldnames(trialdata);
counter = 1;
for i = 1:length(fnames)
  if size(trialdata.(fnames{i}),1) == numSamples
    fnames2resample{counter} = fnames{i}(1:end-3); %get rid of the raw part for renaming the names
    counter = counter + 1;
  end
end

%sets time window for resampling to constant sampling rate (in time)
newTrialSampSize = round(trialdata.trialdurRaw/params.constSampRateTime); %find out how many samples should be in new vect
newTimes = trialdata.timeRaw(1):params.constSampRateTime:((newTrialSampSize*params.constSampRateTime)+trialdata.timeRaw(1)); %get times from start to end with const time window
for i = 1:length(fnames2resample)
  resampledVect = interp1(trialdata.timeRaw,trialdata.([fnames2resample{i} 'Raw']),newTimes,'linear','extrap'); %added extrap to get rid of accidental nan values in the data
  output.([fnames2resample{i}]) = resampledVect;
end

%for incremental vectors (where values go up in steps need to resample to know where these steps are)
fnames2findsteppoints = {'numRewards','numLicks','currentZone','currentWorld','currentPhase'};
for i = 1:length(fnames2findsteppoints) %replace resampled vectors with 0 and resubstitute switch indices as ones

  %use old indices/switch times to find new one
  oldInds = find(diff(trialdata.([fnames2findsteppoints{i} 'Raw'])))+1;
  oldTimes = trialdata.timeRaw(oldInds);
  newInds = lookup2(oldTimes,output.time);
  output.([fnames2findsteppoints{i} 'Inds']) = newInds;
  startingValue = trialdata.([fnames2findsteppoints{i} 'Raw'])(1);
  endingValue = trialdata.([fnames2findsteppoints{i} 'Raw'])(end);
  oldValues = [startingValue; trialdata.([fnames2findsteppoints{i} 'Raw'])(oldInds); endingValue]; %gets values at each of the indices for stepping through

  %make new vector with step functions with step occurring at each ind
  output.(fnames2findsteppoints{i}) = zeros(size(output.(fnames2findsteppoints{i})))+startingValue;
  for stepIdx = 1:length(oldValues(2:end-1))    %only include value where switch occurred, starting value was already initialized

    if stepIdx == length(oldValues(2:end-1))
      inds2change = newInds(stepIdx):length(output.(fnames2findsteppoints{i})); %if last indics then go to end with final value
    else
      inds2change = newInds(stepIdx):newInds(stepIdx+1);
    end
    output.(fnames2findsteppoints{i})(inds2change) = oldValues(stepIdx+1); %replace switch indices time points with 1
  end
end

%get when the phases start/end and what the worlds are at that time
if length(output.currentPhaseInds) >= 1 %if there is more than one phase in a trial (bc otherwise phase inds will just be 0)
  phaseStarts = [1; output.currentPhaseInds];
  phaseEnds = [output.currentPhaseInds-1; length(output.time)];
else
  phaseStarts = 1;
  phaseEnds = length(output.time);
end
phaseStartsEnds = [phaseStarts phaseEnds];
output.phaseStartsEnds = phaseStartsEnds;
output.worldByPhase = output.currentWorld(phaseStartsEnds(:,1));
output.phaseType = output.currentPhase(phaseStartsEnds(:,1));

%get complete and incomplete phases
possiblePhaseTypes = [0 1 2 3 4]; %encoding (0), delay (1), choice (2), reward (3) punish (4) on the forced alt task
incompletePhases = possiblePhaseTypes(~ismember(possiblePhaseTypes, output.phaseType));
completePhases = output.phaseType;
output.completePhases = completePhases+1;

end
