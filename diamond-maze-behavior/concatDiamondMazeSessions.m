function allsessdata = concatDiamondMazeSessions(animals, indices, behaviordata, trainingoptions)
%SP 190925
%this function concatenates data across session for visualization purposes

%% initialize fieldnames for data structure
fnames = fieldnames(behaviordata.bySession{end}{end}{end});
fnamesPos = {'posXEnc'; 'posYEnc'; 'posXChoice'; 'posYChoice';'viewAngle';'sessInfo'};
fnamesAll = [fnames; fnamesPos];

%% concatenate data across sessions
for anIdx = 1:length(animals)
    inclsess = find(indices.behaviorindex(:,1) == animals(anIdx));
    for trackIdx = 1:length(trainingoptions)
        %initialize structures and track info
        track = trainingoptions{trackIdx};
        for fieldIdx = 1:length(fnamesAll)
            allsessdata(animals(anIdx)).(track).(fnamesAll{fieldIdx}) = [];
        end

        %loop through sessions and concatenate data
        sesscounter = 1; %keep track of how many sessions on each track there are
        for sessIdx = 1:size(inclsess,1)
            %initialize variabls
            sessindex = indices.behaviorindex(inclsess(sessIdx),:);
            sessdata = behaviordata.bySession{sessindex(1)}{sessindex(2)}{sessindex(3)};
            trialdata = behaviordata.byTrial{sessindex(1)}{sessindex(2)}{sessindex(3)};

            %hacky way to get choice 1 side for now bc I was dumb and forgot to include it in the raw data structure
            if sessindex(2) > 190922 && sessindex(2) < 191030
                sessdata.trainingtype = 'choice1side_short';
            end
                
            if strcmp(sessdata.trainingtype,track)

                %combine all the fields where there is one value per trial/session
                for fieldIdx = 1:length(fnames)
                    if isfield(sessdata, fnamesAll{fieldIdx})
                        allsessdata(animals(anIdx)).(track).(fnamesAll{fieldIdx}) = [allsessdata(animals(anIdx)).(track).(fnamesAll{fieldIdx}); sessdata.(fnamesAll{fieldIdx})];
                    end
                end

                %combine fields where there are multiple values per trial (ex. viewAngle, positionX, positionY)



                %sessinfo structure so you can index into trials easily
                sessInfo = [repmat([sessindex(1:3), sesscounter],sessdata.numTrialsAll,1), [1:sessdata.numTrialsAll]'];
                allsessdata(animals(anIdx)).(track).sessInfo = [allsessdata(animals(anIdx)).(track).sessInfo; sessInfo];
                sesscounter = sesscounter + 1;
            end
        end
    end
end
