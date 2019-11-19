function output = resampleDiamondMazeTrials(trialdata,params);
% SP 190924
% SP changed on 191119 to move this function to the trial section of the code

%gets fields to resample (vectors of behavioral data)
numSamples = length(trialdata.time);
fnames = fieldnames();
counter = 1;
for i = 1:length(fnames)
  if size(trialdata.(fnames{i}),1) == numSamples
    fnames2resample{counter} = fnames{i};
    counter = counter + 1;
  end
end

%sets time window for resampling to constant sampling rate (in time)
newTrialSampSize = round(trialdata.trialdur/params.constSampRateTime); %find out how many samples should be in new vect
newTimes = trialdata.time(1):params.constSampRateTime:((newTrialSampSize*params.constSampRateTime)+trialdata.time(1)); %get times from start to end with const time window
for i = 1:length(fnames2resample)
  resampledVect = interp1(trialdata.time,trialdata.(fnames2resample{i}),newTimes,'linear','extrap'); %added extrap to get rid of accidental nan values in the data
  output.([fnames2resample{i} 'ConstTime']) = resampledVect;
end

%for incremental vectors (where values go up in steps need to resample to know where these steps are)
fnames2findsteppoints = {'numRewards','numLicks','currentZone','correctZone','currentWorld','currentPhase''numTrials'};
for i = 1:length(fnames2findsteppoints) %replace resampled vectors with 0 and resubstitute switch indices as ones
  %use old indices/switch times to find new ones
  oldInds = find(diff(trialdata.(fnames2findsteppoints{i})))+1;
  oldTimes = trialdata.time(oldInds);
  newInds = lookup2(oldTimes,output.timeConstTime);
  %make new vector where new Inds are indicated with ones and the rest of the vector is zeros
  output.([fnames2findsteppoints{i} 'ConstTime']) = zeros(size(output.([fnames2findsteppoints{i} 'ConstTime'])));
  output.([fnames2findsteppoints{i} 'ConstTime'])(newInds) = 1; %replace switch indices time points with 1
  output.([fnames2findsteppoints{i} 'IndsConstTime']) = newInds;
end

%get when the phases start/end and what the worlds are at that time
phaseStarts = [1; output.currentPhaseIndsConstTime)];
phaseEnds = [output.currentPhaseIndsConstTime)-1; length(output.currentPhaseConstTime)];
phaseStartsEnds = [phaseStarts phaseEnds];
output.phaseStartsEndsConstTime = phaseStartsEnds;
output.worldByPhaseConstTime = output.currentWorldConstTime(phaseStartsEnds);
output.phaseTypeConstTime = output.currentPhaseConstTime(phaseStartsEnds);

%get complete and incomplete phases
possiblePhaseTypes = [0 1 2 3 4]; %encoding (0), delay (1), choice (2), reward (3) punish (4) usually
incompletePhases = possiblePhaseTypes(~ismember(possiblePhaseTypes, output.phaseTypeConstTime));
completePhases = trialdata.phaseType;
output.completePhases = completePhases+1;

end
