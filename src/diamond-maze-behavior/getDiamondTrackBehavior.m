function getDiamondTrackBehavior(dirs, indices, makenewfiles)
%SP 190813
%this function generates and plots the behavioral data for the Diamond
%Track behavioral paradigm

params.constSampNum = 10000; %set number of samples for resampling to constant number (ie. each trial is 10000 samples)
params.constSampRateTime = 0.01; %sets time window for resampling to constant sampling rate (in time), 0.01 s samples means 100 Hz samp rate
dirs.behaviorfigdir = [dirs.savedfiguresdir 'behavior\' num2str(yyyymmdd(datetime('now'))) '\'];
dirs.savedatadir = [dirs.behaviorfigdir 'data\'];
if ~exist(dirs.savedatadir); mkdir(dirs.savedatadir); end;

%% generate the data structures
behaviordata = getDiamondTrackBehaviorMetrics(dirs,indices,params,makenewfiles);

%% plot the data
statsoutput = plotDiamondTrackBehaviorMetrics(dirs,indices,params, behaviordata);

%% write the stats output file
%load filename
filename = [dirs.behaviorfigdir 'behavioralstatsoutput.txt'];
writestatsoutputfile(filename,statsoutput)

end
