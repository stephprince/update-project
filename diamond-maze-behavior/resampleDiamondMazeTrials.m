function output = resampleDiamondMazeTrials(trialdata,params);
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
newTrialSampSize = round(trialdata.trialdur/params.constSampRateTime); %find out how many samples should be in new vect
newTimes = trialdata.time(1):params.constSampRateTime:((newTrialSampSize*params.constSampRateTime)+trialdata.time(1)); %get times from start to end with const time window
for i = 1:length(fnames2resample)
  resampledVect = interp1(trialdata.time,trialdata.(fnames2resample{i}),newTimes,'linear','extrap'); %added extrap to get rid of accidental nan values in the data
  output.([fnames2resample{i}]) = resampledVect;
end

%for incremental vectors (where values go up in steps need to resample to know where these steps are)
fnames2findsteppoints = {'numRewards','numLicks','currentZone','correctZone','currentWorld','currentPhase''numTrials'};
for i = 1:length(fnames2findsteppoints) %replace resampled vectors with 0 and resubstitute switch indices as ones
  %use old indices/switch times to find new ones
  oldInds = find(diff(trialdata.(fnames2findsteppoints{i})))+1;
  oldTimes = trialdata.time(oldInds);
  newInds = lookup2(oldTimes,output.time);
  %make new vector where new Inds are indicated with ones and the rest of the vector is zeros
  output.(fnames2findsteppoints{i}) = zeros(size(output.(fnames2findsteppoints{i})));
  output.(fnames2findsteppoints{i})(newInds) = 1; %replace switch indices time points with 1
  output.([fnames2findsteppoints{i} 'Inds']) = newInds;
end

%get when the phases start/end and what the worlds are at that time
phaseStarts = [1; output.currentPhaseInds];
phaseEnds = [output.currentPhaseInds-1; length(output.currentPhase)];
phaseStartsEnds = [phaseStarts phaseEnds];
output.phaseStartsEnds = phaseStartsEnds;
output.worldByPhase = output.currentWorld(phaseStartsEnds);
output.phaseType = output.currentPhase(phaseStartsEnds);

%get complete and incomplete phases
possiblePhaseTypes = [0 1 2 3 4]; %encoding (0), delay (1), choice (2), reward (3) punish (4) usually
incompletePhases = possiblePhaseTypes(~ismember(possiblePhaseTypes, output.phaseType));
completePhases = output.phaseType;
output.completePhases = completePhases+1;

end
