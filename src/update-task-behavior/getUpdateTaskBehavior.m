function getUpdateTaskBehavior(dirs, indices, makenewfiles)
%SP 190813
%this function generates and plots the behavioral data for the Update Task

%set parameters for all behavior analysis
params.constSampNum = 10000; %set number of samples for resampling to constant number (ie. each trial is 10000 samples)
params.constSampRateTime = 1/500; %sets time window for resampling to constant sampling rate (in time), 0.01 s samples means 2000 Hz samp rate
params.taskStatesMap = containers.Map({'startOfTrial', 'initialCue', 'updateCue', 'delayCue', 'choiceMade', 'duringReward', 'endOfTrial', 'interTrial'},[1:8]);
params.trackTypeMap = containers.Map({'linear','ymazeShort','ymazeLong','ymazeSurprise'},[1:4]);
params.trialTypeMap = containers.Map({'left','right','linear'},[1 2 3]); %this is what virmen sees, the actual environment is flipped
params.updateTypeMap = containers.Map({'stay','update'},[1 2]);
params.choiceMap = containers.Map({'incorrect','correct','terminated'},[0 1 2]);

%set directories
if makenewfiles
  dirs.behaviorfigdir = [dirs.savedfiguresdir 'behavior\' num2str(yyyymmdd(datetime('now'))) '\'];
else
  folders = dir([dirs.savedfiguresdir 'behavior\']);
  dirs.behaviorfigdir = [dirs.savedfiguresdir 'behavior\' folders(end).name '\']; %get most recent folder
  dirs.behaviorfigdir = '\\neuro-cloud\labs\singer\Steph\Code\hpc-pfc-interactions\results\behavior\20201129\';
end
dirs.savedatadir = [dirs.behaviorfigdir 'data\'];

if ~exist(dirs.savedatadir); mkdir(dirs.savedatadir); end;

%% generate the data structures
behaviordata = getUpdateTaskBehaviorMetrics(dirs,indices,params,makenewfiles);

%% plot the data
plotUpdateTaskBehaviorMetrics(dirs,indices,params, behaviordata);

end
