function dirs = getDefaultUpdateTaskDirectories
%this function sets all the default directories used for monitoring
%behavioral perfomrance

%% directory to save all the data into
dirs.projectdir = 'C:\Users\singerlab\Desktop\UpdateTaskPerformance\';

%% directories to get and save data to
dirs.virmendatadir = 'C:\Users\singerlab\Desktop\Virmen Data\UpdateMaze\Data\';
dirs.savedfiguresdir = [dirs.projectdir];

%% files to use
dirs.behaviorspreadsheetfname = [dirs.virmendatadir(1:end-5) 'VRUpdateTaskBehaviorSummary.csv'];
