function  behaviorDataDiamondBySess = calcDiamondMetricsBySess(sessdata, trialdata, params, dirs, index, animalID, makenewfiles);
%this function calculates session metrics using the raw and trial data
%input:
%       sessdata - raw behavior structure, output of loadRawDiamondVirmenFile
%       trialdata - behavior broken up into trials, output of calcDiamondMetricsByTrial
%       dirs - directory structure with all the file path info
%       sessindex - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'

filename = [dirs.savedatadir 'behaviorDataDiamondSess_' animalID num2str(index(1)) '_' num2str(index(2)) '_' num2str(index(3))];
disp(['Calculating session metrics for ' animalID num2str(index(1)) ' ' num2str(index(2)) ' ' num2str(index(3))]);

if ~exist(filename) || makenewfiles
    %% resample vectors so can combine across trials
    for trialIdx = 1:size(trialdata,2)
        numSamples = length(trialdata{trialIdx}.time); %initial sample number
        
        %resample velocity, time, etc.
        behaviorDataDiamondBySess.timeNorm(trialIdx,:) = resample(trialdata{trialIdx}.time, params.newSampSize, numSamples);
        behaviorDataDiamondBySess.velocTransNorm(trialIdx,:) = resample(trialdata{trialIdx}.velocTrans, params.newSampSize, numSamples);
        behaviorDataDiamondBySess.velocRotNorm(trialIdx,:) = resample(trialdata{trialIdx}.velocRot, params.newSampSize, numSamples);
        if ~isnan(trialdata{trialIdx}.viewAngle)
            behaviorDataDiamondBySess.viewAngle(trialIdx,:) = resample(trialdata{trialIdx}.viewAngle, params.newSampSize, numSamples);
        end
        
        %adjust phase/licks/reward times for resampled data
        %(get fraction of initial trial time that event occurred and multiply by new sample size)
        behaviorDataDiamondBySess.phaseInds{trialIdx} = trialdata{trialIdx}.phaseInds*params.newSampSize/numSamples;
        behaviorDataDiamondBySess.rewardInds{trialIdx} = trialdata{trialIdx}.rewardInds*params.newSampSize/numSamples;
        behaviorDataDiamondBySess.lickInds{trialIdx} = trialdata{trialIdx}.lickInds*params.newSampSize/numSamples;
        
        %adjust position
        %caveat here, need to separate phases bc otherwise you will smooth over the really big jumps in position
        %so for the adjusted position vectors, each phase will be numSamples x 1
        possiblePhaseTypes = [0 1 2 3 4]; %encoding (0), delay (1), choice (2), reward (3) punish (4) usually
        incompletePhases = possiblePhaseTypes(~ismember(possiblePhaseTypes, trialdata{trialIdx}.phaseType));
        completePhases = trialdata{trialIdx}.phaseType;
        phaseInds = trialdata{trialIdx}.phaseInds;
        behaviorDataDiamondBySess.completePhases = completePhases+1;
        
        %fill in incomplete phases with nans
        for phaseIdx = 1:length(incompletePhases)
            phaseType = incompletePhases(phaseIdx);
            behaviorDataDiamondBySess.phase(phaseType+1).posXNorm(trialIdx,:) = nan(1, params.newSampSize); %since encoding is 0, have to shift phases by 1
            behaviorDataDiamondBySess.phase(phaseType+1).posYNorm(trialIdx,:) = nan(1, params.newSampSize); %so now enc = 1, del = 2, choice = 3, reward = 4, punish = 5;
        end
        
        %fill in complete phases with data
        for phaseIdx = 1:length(completePhases)
            phaseType = completePhases(phaseIdx);
            posX = trialdata{trialIdx}.positionX(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
            posY = trialdata{trialIdx}.positionY(phaseInds(phaseIdx,1):phaseInds(phaseIdx,2));
            padFactor = 100;
            padLength = ceil(length(posX)/params.newSampSize)*padFactor; %get ratio of downsampling and multiply by padFactor to add pad on either side
            posXpad = [repmat(posX(1),padLength,1); posX; repmat(posX(end),padLength,1)];
            posYpad = [repmat(posY(1),padLength,1); posY; repmat(posY(end),padLength,1)];
            numSamplesPhase = length(posXpad);
            
            posXtemp = resample(posXpad, params.newSampSize+padFactor*2, numSamplesPhase);
            posYtemp = resample(posYpad, params.newSampSize+padFactor*2, numSamplesPhase);
            if length(posY) == 1
                posYtemp = repmat(posY,params.newSampSize+padFactor*2,1);
                posXtemp = repmat(posX,params.newSampSize+padFactor*2,1);
            end
            behaviorDataDiamondBySess.phase(phaseType+1).posXNorm(trialIdx,:) = posXtemp(padFactor:end-padFactor-1);
            behaviorDataDiamondBySess.phase(phaseType+1).posYNorm(trialIdx,:) = posYtemp(padFactor:end-padFactor-1);
        end
    end
    
    %% concatenate trial outputs
    %its a little confusing but in this scenario any variables that start
    %with 'trial' are the concatenation of all the trials and any variables
    %that don't are the single outcome/parameter for that trial 
    trialDur = []; trialStartLoc = []; trialChoiceLoc = []; trialCorrectEncLoc = []; trialCorrectChoiceLoc = [];
    trialTurnDirEnc = []; trialTurnDirChoice = [];
    for trialIdx = 1:size(trialdata,2)
        %the easy stuff, duration, starting locations in string format
        trialDur = [trialDur; trialdata{trialIdx}.trialdur]; %trial duration
        
        %finds what the directions/locations are and map to number (ie,1 = north, 2 = east, 3 = south, 4 = west (like a compass lol))
        mappingLoc = {'north', 'east', 'south', 'west','nan'}; mappingNum = [1,2,3,4,5];
        startLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialStartLoc)))); %trial startpoint (encoding)
        choiceLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialChoiceLoc)))); %trial choice phase startpoint
        correctEncLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialCorrectEncLoc)))); %trial encoding side
        correctChoiceLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialCorrectChoiceLoc)))); %trial choice side
        
        %concatenate coded directions across trials
        trialStartLoc = [trialStartLoc; startLoc];
        trialChoiceLoc = [trialChoiceLoc; choiceLoc];
        trialCorrectEncLoc = [trialCorrectEncLoc; correctEncLoc];
        trialCorrectChoiceLoc = [trialCorrectChoiceLoc; correctChoiceLoc];
        
        %separate coded directions by phase (bc enc and choice have different options and potential starting points)
        % phases are encoding (1), delay (2), choice (3), intertrial interval (4)
        behaviorDataDiamondBySess.phase(1).startLoc(trialIdx) = startLoc; %encoding phase start point
        behaviorDataDiamondBySess.phase(1).correctLoc(trialIdx) = correctEncLoc; %encoding correct side
        behaviorDataDiamondBySess.phase(3).startLoc(trialIdx) = choiceLoc; %choice phase start point
        behaviorDataDiamondBySess.phase(3).correctLoc(trialIdx) = correctChoiceLoc; %choice correct side
        
        %classify phases as right or left turns (left = 1, right = 2)
        mappingTurns = [1, 2, 1; %north start, east open, left turn
            1, 4, 2; %north start, west open, right turn
            3, 2, 2; %south start, east open, right turn
            3, 4, 1];  %south start, west open, left turn
        turnDirEnc = mappingTurns(ismember(mappingTurns(:,1:2),[startLoc, correctEncLoc],'rows'),3);
        behaviorDataDiamondBySess.phase(1).turnDir(trialIdx,:) = turnDirEnc;
        turnDirChoice = mappingTurns(ismember(mappingTurns(:,1:2),[choiceLoc, correctChoiceLoc],'rows'),3);
        if ~isempty(turnDirChoice)
            behaviorDataDiamondBySess.phase(3).turnDir(trialIdx,:) = turnDirChoice;
        else
            behaviorDataDiamondBySess.phase(3).turnDir(trialIdx,:) = nan;
        end
        trialTurnDirEnc = [trialTurnDirEnc; turnDirEnc];
        trialTurnDirChoice = [trialTurnDirChoice;  turnDirChoice];
    end
    
    %save output data to data structure
    behaviorDataDiamondBySess.trialStartLoc = trialStartLoc;
    behaviorDataDiamondBySess.trialChoiceLoc = trialChoiceLoc;
    behaviorDataDiamondBySess.trialCorrectEncLoc = trialCorrectEncLoc;
    behaviorDataDiamondBySess.trialCorrectChoiceLoc = trialCorrectChoiceLoc;
    behaviorDataDiamondBySess.trialDur = trialDur;
    behaviorDataDiamondBySess.trialTurnDirEnc = trialTurnDirEnc;
    behaviorDataDiamondBySess.trialTurnDirChoice = trialTurnDirChoice;
    
    %% get percent correct
    %get outcomes for all trials
    temp = []; temp = cellfun(@(x) [temp; x.outcome], trialdata, 'UniformOutput', 0);
    behaviorDataDiamondBySess.sessOutcomes = cell2mat(temp);
    behaviorDataDiamondBySess.logicalCorrect = behaviorDataDiamondBySess.sessOutcomes == 1;
    behaviorDataDiamondBySess.numCorrect = length(find(behaviorDataDiamondBySess.sessOutcomes == 1));
    behaviorDataDiamondBySess.numFailed = length(find(behaviorDataDiamondBySess.sessOutcomes == -1));
    behaviorDataDiamondBySess.numIncorrect = length(find(behaviorDataDiamondBySess.sessOutcomes == 0));
    behaviorDataDiamondBySess.numTrials = size(trialdata,2);
    behaviorDataDiamondBySess.trainingtype = sessdata.params.trainingtype;
    
    %get outcomes for right vs. left turn trials (right or left turns during choice phase
    behaviorDataDiamondBySess.sessOutcomesLeft = behaviorDataDiamondBySess.sessOutcomes(trialTurnDirChoice == 1); %left turns
    behaviorDataDiamondBySess.sessOutcomesRight = behaviorDataDiamondBySess.sessOutcomes(trialTurnDirChoice == 2); %right turns
    behaviorDataDiamondBySess.numCorrectLeft = length(find(behaviorDataDiamondBySess.sessOutcomesLeft == 1));
    behaviorDataDiamondBySess.numCorrectRight = length(find(behaviorDataDiamondBySess.sessOutcomesRight == 1));
    behaviorDataDiamondBySess.numTrialsLeft = length(behaviorDataDiamondBySess.sessOutcomesLeft);
    behaviorDataDiamondBySess.numTrialsRight = length(behaviorDataDiamondBySess.sessOutcomesRight);
    
    %% save filename
    save(filename, 'behaviorDataDiamondBySess');
    
else
    load(filename);
end
