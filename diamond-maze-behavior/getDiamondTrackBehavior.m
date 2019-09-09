function getDiamondTrackBehavior(dirs, indices, makenewfiles)
%SP 190813
%this function generates and plots the behavioral data for the Diamond
%Track behavioral paradigm

params.newSampSize = 10000; %set number of samples for behavioral outputs
dirs.behaviorfigdir = [dirs.savedfiguresdir 'behavior\' num2str(yyyymmdd(datetime('now'))) '\'];
dirs.savedatadir = [dirs.behaviorfigdir 'data\'];
if ~exist(dirs.savedatadir); mkdir(dirs.savedatadir); end;

%% generate the data structures
behaviordata = getDiamondTrackBehaviorMetrics(dirs,indices,params,makenewfiles);

%% plot the data
statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,behaviordata);

%% write the stats output file
%load filename
filename = [dirs.behaviorfigdir 'behavioralstatsoutput.txt'];
writestatsoutputfile(filename,statsoutput)

end