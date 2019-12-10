function output = cleanTeleportEvents(trackdata)

%clean the data to get rid of teleportation events for plotting
minY = []; minX = []; maxX = []; maxY = [];
output.time = trackdata.time;
for trialIdx = 1:size(trackdata.time,1)
  %find when animal is in main part of track and large position jumps
  altTrackInds = find(trackdata.currentWorld{trialIdx} == 1);
  teleportEvents = find(abs(diff(trackdata.positionY{trialIdx})) > 10);
  if isempty(teleportEvents);
    bins2keep = altTrackInds;
  else
    bins2throw = find(ismember(altTrackInds,teleportEvents));
    bins2keep = altTrackInds(1):altTrackInds(bins2throw);
  end

  %clean the vectors for each trial
  output.positionXClean{trialIdx} = trackdata.positionX{trialIdx}(bins2keep);
  output.positionYClean{trialIdx} = trackdata.positionY{trialIdx}(bins2keep);
  output.velocTransClean{trialIdx} = trackdata.velocTrans{trialIdx}(bins2keep);
  output.velocRotClean{trialIdx} = trackdata.velocRot{trialIdx}(bins2keep);
  output.viewAngleClean{trialIdx} = trackdata.viewAngle{trialIdx}(bins2keep);

  %get bins for averaging
  minY = [minY; min(output.positionYClean{trialIdx})];
  maxY = [maxY; max(output.positionYClean{trialIdx})];
  minX = [minX; min(output.positionXClean{trialIdx})];
  maxX = [maxX; max(output.positionXClean{trialIdx})];
end

%get averages for right and left turns by binning by position
output.minYpos = min(minY); output.minXpos = min(minX);
output.maxYpos = max(maxY); output.maxXpos = max(maxX);
output.posYbins = linspace(output.minYpos,output.maxYpos,50);
output.posXbins = linspace(output.minXpos,output.maxXpos,50);
