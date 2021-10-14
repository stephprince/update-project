function output = concatTrialBehavior(trialData)
%SP 191121
% this function concatenate data from trials like position, view angle, etc.

%% get variables to concatenate
fnames = fieldnames(trialdata{1});
counter = 1;
for fieldIdx = 1:length(fnames)
    if size(trialdata{1}.(fnames{fieldIdx}),2) == size(trialdata{1}.time,2) %checking which fieldnames are behavior data vectors
      fnames2concat{counter} = fnames{fieldIdx};
      counter = counter + 1;
    end
end

%% loop through trials to concatenate data
trialcounter = 1; trialInfo = [];
for trialIdx = 1:size(trialdata,2)
    for fieldIdx = 1:length(fnames2concat)
      output.(fnames2concat{fieldIdx}){trialcounter,1} = trialdata{trialIdx}.(fnames2concat{fieldIdx}); %change data structure so will be able to concat sessions
    end
    trialcounter = trialcounter + 1;
end
