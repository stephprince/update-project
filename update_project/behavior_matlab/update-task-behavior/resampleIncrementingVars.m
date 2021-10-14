function out = resampleIncrementingVars(vect2convert,newTimeVect, data, params)
    %get new and old inds for step changes and their values
    oldStepInds = find(diff(vect2convert)) + 1;
    oldStepTimes = data.time(oldStepInds);
    newStepInds = [lookup2(oldStepTimes, newTimeVect); length(vect2convert)];
    stepVals = [vect2convert(1); vect2convert(oldStepInds); vect2convert(end)];

    %create new vectors with values in the right places
    out = zeros(length(newTimeVect),1)+vect2convert(1);
    for stepIdx = 1:length(oldStepInds(1:end-1))
        inds2change = newStepInds(stepIdx):newStepInds(stepIdx+1);
        out(inds2change) = stepVals(stepIdx+1);
    end
end