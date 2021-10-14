function out = getUpdateTaskIndex(animal, spreadsheetfilename, animalID)
%This function gets the indices and best sessions of the behavioral data
%for a given animal and track. This is for the specific spreadsheet that is
%defined in the function 'DiamondTrackSummary.xlsx'.
%
%Inputs:    animal - number, without a letter
%           track - track number, see spreadsheet for key
%           spreadsheetdir - the directory of the spreadsheet
%
%Outputs:   out - a four column vector of indices
%               [animal# date session# genotype]

data = readtable(spreadsheetfilename,'Delimiter',',');

%Column 1 is the Animal ID Number
%Column 2 is the date
%Column 3 is the session number
%Column 5 is the include/don't include flag (called Exclude_ bc question mark in CSV)
%Column 6 is the ephys/not ephys flag (called Ephys_ bc question mark in CSV)

newAnimalIDs = cellfun(@(x) convertAnimalIDToNum(x), data.Animal);
data.Animal = newAnimalIDs;

out = data(ismember(data.Animal,animal) & ~data.Exclude_, {'Animal','Date','Session','Ephys_','FirstWorld','LastWorld'});

function out = convertAnimalIDToNum(str)
  if ~isempty(str)
    out = str2num(str(2:end));
  end
end

end
