function dirs = getdefaultdirectorieshpcpfcinteractions
%SP 3.12.19
%this function sets all the default directories used for the analyses

%% directory to save all the data into
dirs.baseserverdir = '\\ad.gatech.edu\bme\labs\singer\';
dirs.projectdir = [dirs.baseserverdir 'Steph\Code\update-project\'];

%% directories to get and save data to
dirs.virmendatadir = [dirs.baseserverdir 'Virmen Logs\UpdateTask\'];
dirs.processeddatadir = [dirs.baseserverdir  'ProcessedData\UpdateTask\'];
dirs.savedfiguresdir = [dirs.projectdir 'results\'];

%% files to use
dirs.ephysspreadsheetfname = [dirs.projectdir 'docs\metadata-summaries\VRUpdateTaskEphysSummary.xlsx'];
dirs.behaviorspreadsheetfname = [dirs.projectdir 'docs\metadata-summaries\VRUpdateTaskBehaviorSummary.csv'];
dirs.virmendatafname = 'virmenDataRaw';
dirs.virmenephysdatafname = 'virmenDataSynced';