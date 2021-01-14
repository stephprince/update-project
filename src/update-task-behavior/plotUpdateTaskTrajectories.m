function plotUpdateTaskTrajectories(trackdata, indices, dirs, params)

savedfiguresdir = [dirs.behaviorfigdir 'trajectories\'];
if ~exist(savedfiguresdir); mkdir(savedfiguresdir); end;

%get histogram of position for the vector and apply the indices to other metrics
positionHists = calcPositionHists(trackdata, indices, dirs, params);

% plot averages and individual traces
plotHistByPosition(hists2plotConcat, trialtypes, metrics2plot, animal, track, dirs);