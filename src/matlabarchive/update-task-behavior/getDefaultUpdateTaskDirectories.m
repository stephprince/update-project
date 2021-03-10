function dirs = getDefaultUpdateTaskDirectories
%this function sets all the default directories used for monitoring
%behavioral perfomrance

%% directory to save all the data into
dirs.projectdir = 'C:\\Desktop\UpdateTaskPerformance\';

%% directories to get and save data to
dirs.virmendatadir = 'C:\\Desktop\VirmenData\UpdateTask';
dirs.savedfiguresdir = [dirs.projectdir];

%% files to use
dirs.behaviorspreadsheetfname = [dirs.virmendatadir 'doc\VRUpdateTaskBehaviorSummary.csv'];
