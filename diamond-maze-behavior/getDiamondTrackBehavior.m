function getDiamondTrackBehavior(dirs, indices, makenewfiles)
%SP 190813
%this function generates and plots the behavioral data for the Diamond
%Track behavioral paradigm

params.binsize = 2; %binsize for licking and velocity data
dirs.behaviorfigdir = [dirs.savedfiguresdir 'behavior\'];
if ~exist(dirs.behaviorfigdir); mkdir(dirs.behaviorfigdir); end;

%% generate the data structures
behaviordata = getDiamondTrackBehaviorMetrics(dirs,indices,params,makenewfiles);

%% plot the data
statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,behaviordata);

%% write the stats output file
%load filename
filename = [dirs.behaviorfigdir 'behavioralstatsoutput.txt'];
writestatsoutputfile(filename,statsoutput)

end