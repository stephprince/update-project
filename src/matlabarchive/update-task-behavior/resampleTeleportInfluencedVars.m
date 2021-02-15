function out = resampleTeleportInfluencedVars(vect2convert,newTimeVect,data, params)
    %get new and old inds for teleport events
    patternToFind = ['[' num2str(params.taskStatesMap('interTrial')) ']+' num2str(params.taskStatesMap('startOfTrial'))]; %look for any number of intertrial periods and the start of a new trial
    teleportToInterTrial = regexp(sprintf('%i', data.taskState), patternToFind)'; %teleport happens at first intertrial phase
    teleportFromInterTrial = find(data.taskState == params.taskStatesMap('startOfTrial'))+1; %teleport happens right after start of new trial
    oldTeleportInds = [1; sort([teleportToInterTrial; teleportFromInterTrial]); numel(data.taskState)];
    oldTeleportTimes = data.time(oldTeleportInds);
    newTeleportInds = lookup2(oldTeleportTimes, newTimeVect);
    
    %create new vectors that interpolate only in the same environment (exclude teleports)
    out = nan(length(newTimeVect),1);
    for eventIdx = 1:length(oldTeleportInds)-1
        inds2interpover = oldTeleportInds(eventIdx):oldTeleportInds(eventIdx+1)-1;
        oldtimestointerpover = data.time(inds2interpover);
        newtimestointerpover = newTimeVect(newTeleportInds(eventIdx):newTeleportInds(eventIdx+1)-1);
        resampledVect = interp1(oldtimestointerpover, vect2convert(inds2interpover',:), newtimestointerpover,'linear','extrap');
        out(newTeleportInds(eventIdx):newTeleportInds(eventIdx+1)-1) = resampledVect;
    end
end