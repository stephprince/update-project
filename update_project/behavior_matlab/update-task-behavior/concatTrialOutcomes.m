function output = concatTrialOutcomes(trialdata,tracktype)

%% initialize variables
fnames = {'dur','startLoc','choiceLoc','correctEncLoc','correctChoiceLoc','turnDirEnc','turnDirChoice','sameTurn'};
for fieldIdx = 1:length(fnames)
    output.([fnames{fieldIdx} 'All']) = [];
end

%% loop through trials to concatenate data
lastTrialTurn = 0; %initialize for first trial of continuous alt lookback
for trialIdx = 1:size(trialdata,2)
    %% get info from trialdata structure
    %trial duration
    dur = trialdata{trialIdx}.trialdurRaw;

    % get start and reward locations
    %finds what the directions/locations are and map to number (ie,1 = north, 2 = east, 3 = south, 4 = west (like a compass lol))
    mappingLoc = {'north', 'east', 'south', 'west','nan'};
    mappingNum = [1,2,3,4,5];
    startLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialStartLoc)))); %trial startpoint (encoding)
    choiceLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialChoiceLoc)))); %trial choice phase startpoint
    correctEncLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialCorrectEncLoc)))); %trial encoding side
    correctChoiceLoc = mappingNum(find(~cellfun(@isempty, strfind(mappingLoc,trialdata{trialIdx}.trialCorrectChoiceLoc)))); %trial choice side

    %classify phases as right or left turns (left = 1, right = 2)
    mappingTurns = [1, 2, 1; %north start, east open, left turn
        1, 4, 2; %north start, west open, right turn
        3, 2, 2; %south start, east open, right turn
        3, 4, 1];  %south start, west open, left turn
    turnDirEnc = mappingTurns(ismember(mappingTurns(:,1:2),[startLoc, correctEncLoc],'rows'),3);
    turnDirChoice = mappingTurns(ismember(mappingTurns(:,1:2),[choiceLoc, correctChoiceLoc],'rows'),3);

    %classify trials as same or alternate turning direction for correct choice
    if strcmp(tracktype,'continuousalt') %same or alt turn with different meanings for continuous alt (previous trial the same or alternate turn)
      thisTrialTurn = turnDirEnc;
      sameoralt = thisTrialTurn - lastTrialTurn;
      lastTrialTurn = thisTrialTurn; %reset last trial value to this trial
    else
      sameoralt = turnDirEnc - turnDirChoice; %if encoding and choice are the same then result is 0, otherwise +/- 2 since it's 4 and 2 reward locations
    end

    if sameoralt
        sameTurn = 0; %if theres a positive or negative value, encoding and choice turning values were different
    else
        sameTurn = 1; %if 0, turn and choice were the same
    end

    %% concatenate across trials
    for fieldIdx = 1:length(fnames)
        output.([fnames{fieldIdx} 'All']) = [output.([fnames{fieldIdx} 'All']); eval(fnames{fieldIdx})];
    end
end
