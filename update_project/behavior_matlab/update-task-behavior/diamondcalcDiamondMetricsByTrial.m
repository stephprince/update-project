function behaviorDataDiamondByTrial = calcDiamondMetricsByTrial(sessdata, dirs, index, animalID)
%this function splits up diamond track behavioral data into individual
%trials

%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

%% get trial start/end times

%% make new trial data structure
