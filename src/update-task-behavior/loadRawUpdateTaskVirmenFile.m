function behaviorDataTableRaw = loadRawUpdateTaskVirmenFile(dirs, index, animalID, makenewfiles)
%this function converts raw virmen data from the Update track to a table
%200930

%inputs:
%       dirs - directory structure with all the file path info
%       index - single animal index in format [animal# date session# genotype]
%       animalID - animal identifier, ie 'S','F'
%outputs:
%       behaviorTable - contains all the info from the virmen file organized into
%       a table

%% check if file already exists or if makenewfile flag is set
savedatadir = [dirs.behaviorfigdir 'data\'];
if ~exist(savedatadir); mkdir(savedatadir); end;
filename = [savedatadir 'updateTaskBehaviorDataRaw_' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)];

if ~exist(filename) || makenewfiles
    %% load mat file
    try
        sessionfile = load([dirs.virmendatadir, animalID, num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session),'\virmenDataRaw.mat']);
    catch ME
        if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
            fprintf(['File could not be found for ' animalID, num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session) ' . Skipping analysis... \n']);
            behaviorData = nan;
            return
        end
    end

    %% make corrections to raw data    
    %catch for when I was missing the licks header
    if numel(sessionfile.virmenData.dataHeaders) < size(sessionfile.virmenData.data,2)
        sessionfile.virmenData.dataHeaders = [sessionfile.virmenData.dataHeaders(1:11), {'numLicks'}, sessionfile.virmenData.dataHeaders(12:end)];
    end

    %catch for when I was missing the surpriseCue locations
    if ~isfield(sessionfile.virmenData.params, 'surpriseCueStarts')
        sessionfile.virmenData.params.surpriseCueStarts = nan(1,5);
    end
    
    %catch for old data structure when many less fields
    if size(sessionfile.virmenData.dataHeaders,2) < 23
        sessionfile.virmenData.dataHeaders(16:18) = {'trialTypeUpdate','updateCue','updateOccurred'};
        sessionfile.virmenData.dataHeaders = [sessionfile.virmenData.dataHeaders, {'delayUpdateOccurred','teleportOccurred','syncVoltage','syncTrigger','syncPulse'}];
        sessionfile.virmenData.data(:,19:23) = nan(size(sessionfile.virmenData.data,1),5);
    end

    %adjust for computer time
    sessionfile.virmenData.data(:,1) = (sessionfile.virmenData.data(:,1) - sessionfile.virmenData.data(1,1))*24*60*60;

    %% make data structure
    behaviorData.ProtocolLog{1,:} = struct2table(sessionfile.virmenData.protocolLog);
    behaviorData.Params{1,:} = struct2table(sessionfile.virmenData.params);
    behaviorData.RawData = array2table(sessionfile.virmenData.data,'VariableNames',sessionfile.virmenData.dataHeaders);
    behaviorDataTableRaw = struct2table(behaviorData,'AsArray',1);
    behaviorDataTableRaw = [index(:,1:4), behaviorDataTableRaw];

    %% save structure for single animal
    save(filename,'behaviorDataTableRaw');

else
    load(filename);
end

disp(['Loading file for ' animalID num2str(index.Animal) '_' num2str(index.Date) '_' num2str(index.Session)])

end
