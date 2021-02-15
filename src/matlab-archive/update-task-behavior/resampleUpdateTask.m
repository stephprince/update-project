function outputTable = resampleDiamondMazeTrials(trialData, params)
% SP 190924
% SP changed on 191119 to move this function to the trial section of the code

%gets fields to resample (vectors of behavioral data)
numSamples = size(trialData,1);
continuousVars = {'time','transVeloc','rotVeloc'}; %can be normally interpolated and resampled
teleportVars = {'xPos','yPos','viewAngle'}; %influenced by teleportation, need to be treated specially
discreteVars = trialData.Properties.VariableNames(7:end); %step functions, need to find new inds to resample

%resample continuous variables
newSampSize = round(trialData.time(end)/params.constSampRateTime); %find out how many samples should be in new vect
newTimes = 0:params.constSampRateTime:newSampSize*params.constSampRateTime;
resampledContinuous = varfun(@(x) interp1(trialData.time, x, newTimes,'linear','extrap')',  trialData(:,continuousVars));

%resample teleport-influenced variables
resampledTeleport = varfun(@(x) resampleTeleportInfluencedVars(x, resampledContinuous.Fun_time, trialData, params),  trialData(:,teleportVars));
   
%resample step function variables
resampledDiscrete = varfun(@(x) resampleIncrementingVars(x, resampledContinuous.Fun_time, trialData, params),  trialData(:,discreteVars));

%combind them into output table
outputTable = [resampledContinuous, resampledTeleport, resampledDiscrete];
outputTable.Properties.VariableNames = regexprep(outputTable.Properties.VariableNames, 'Fun_', '');

end
