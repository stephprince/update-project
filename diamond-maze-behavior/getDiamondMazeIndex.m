function out = getDiamondMazeIndex(animal, spreadsheetfilename)
%This function gets the indices and best sessions of the behavioral data
%for a given animal and track. This is for the specific spreadsheet that is
%defined in the function 'DiamondTrackSummary.xlsx'. 
%
%Inputs:    animal - number, without a letter
%           genotype - 1 for WT, 2 for 5XFAD
%           track - track number, see spreadsheet for key
%           spreadsheetdir - the directory of the spreadsheet
%
%Outputs:   out - a four column vector of indices
%               [animal# date session# genotype]

data = xlsread(strcat(spreadsheetfilename)); 

%Column 1 is the Animal ID Number
%Column 2 is the date
%Column 3 is the session number
%Column 4 is the include/don't include flag
%Column 5 is the genotype 

includerows = ismember(data(:,1),animal) & ismember(data(:,4),1);

out = data(includerows, [1 2 3 5]);
end