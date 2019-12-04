function dirs = getdefaultdirectorieshpcpfcinteractions
%SP 3.12.19
%this function sets all the default directories used for the analyses

%% directory to save all the data into
dirs.savedfiguresdir = ['\\neuro-cloud\labs\singer\Steph\Figures\hpcpfcinteractions\'];
if ~exist(dirs.savedfiguresdir); mkdir(dirs.savedfiguresdir); end;

%% directories to get data from
dirs.virmenephysdatadir = '\\neuro-cloud\labs\singer\Steph\VirmenEphysData\';
dirs.virmendatadir = '\\neuro-cloud\labs\singer\Virmen Logs\DiamondMaze\';
dirs.processeddatadir = '\\neuro-cloud\labs\singer\ProcessedData\VR_AnnularTrack\';
dirs.spreadsheetdir = '\\neuro-cloud.ad.gatech.edu\labs\singer\Steph\Code\spreadsheets\VRDiamondTrack.xlsx';
dirs.behaviorspreadsheetdir = '\\neuro-cloud.ad.gatech.edu\labs\singer\Steph\Code\spreadsheets\ContinuousAlternationTrackSummary.xlsx';
%dirs.behaviorspreadsheetdir = '\\neuro-cloud.ad.gatech.edu\labs\singer\Steph\Code\spreadsheets\DiamondTrackSummary.xlsx';
