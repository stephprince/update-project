function velocity = move_diamondmaze_2mice_exponentialgain(vr)
%   the code below was taken from Pinto et al., 2018 paper and adapted by SP
%   for the Singer Lab
%
%   This motion rule assumes that the sensor measures the displacement of the ball at the mouse's
%   feet as [dX, dY] where dX is lateral and dY is anterior-posterior displacement. The displacement
%   is taken literally as the velocity of the mouse in the virtual world, where dX is along the
%   world x-axis and dY is along the world y-axis. The orientation of this displacement vector is
%   interpreted as the new view angle of the animal in the virtual world, with the view angle
%   velocity computed such that this change in view angle is over the course of 1 second.
%
%   A qualitatively determined exponential gain is used. This is
%   motivated by the fact that it is physically much more difficult for a mouse to orient its body
%   along large angles while head-fixed on top of a ball, and is therefore unable to make sharp
%   turns when a linear gain is applied (unless the gain is very large, which then makes it
%   impossible for the mouse to reliably walk in a straight line). The following code can be used to
%   visualize the exponential gain function:
%
%     figure; hold on; grid on; xlabel('Orientation (radians)'); ylabel('View angle velocity (rad/s)');
%     ori = -pi/2:0.01:pi/2; plot(ori,ori); plot(ori,sign(ori).*min( exp(1.4*abs(ori).^1.2) - 1, pi/2 ))

%% get data from NIDAQ and remove nans 
%initialize variables and scaling calibration
velocity = [0 0 0 0];
numSamples = 15; %to look at at a time with the NIDAQ
ballCircumference = 63.8; % in cm
dotsPerRevFront = 1341/10; %integral of velocity per 10 revolutions
dotsPerRevSide = 1341/10; %integral of velocity per 10 revolutions
virmenDisplacementPerCM = 1; %gain of 1 
vr.scaleX = virmenDisplacementPerCM * ballCircumference/dotsPerRevSide;
vr.scaleY = virmenDisplacementPerCM * ballCircumference/dotsPerRevFront;

% Read data from NIDAQ
data = peekdata(vr.ai,numSamples);

% Remove NaN's from the data (these occur after NIDAQ has stopped)
f = isnan(mean(data,2));
data(f,:) = [];
data = mean(data,1)';
data(isnan(data)) = 0;

%dy/dt (front/back translational motion) is data(2)
%dx/dt (side to side motion) is data(1)

% --------------------------------------------------------------------------------%
% THE FOLLOWING SCALING IS USED TO MANIPULATE ROTATIONAL (first number) VS.
% TRANSLATIONAL (second number) MOVEMENT. Rotation is EXTREMELY sensitive. 
% --------------------------------------------------------------------------------%

% Obtain the displacement recorded by sensor (here the second sensor)
vr.orientation  = atan2(-dX*sign(dY), abs(dY));   % Zero along AP axis, counterclockwise is positive

velocity(1:2) = [cos(vr.position(4)) -sin(vr.position(4)); sin(vr.position(4)) cos(vr.position(4))]*velocity(1:2)'; 
%sets position/movement relative to actual location, instead of (0,0) so move in correct direction

% Rotate displacement vector into animal's current coordinate frame
R               = R2(vr.position(4));             % vr.position(4) is the current view angle in the virtual world
%dF              = sqrt(dX^2 + dY^2) * sign(dY);   % Length (including sign for forward vs. backward movement) of displacement vector
dF              = sqrt(velocity(1)^2 + velocity(2)^2) * sign(velocity(2));   % Length (including sign for forward vs. backward movement) of displacement vector
temp            = R * [0; dF];
dX              = temp(1);
dY              = temp(2);

% Apply scale factors to translate optical mouse velocity into virtual world units
velocity(1)     = vr.scaleX * data(1);
velocity(2)     = vr.scaleY * data(2);
velocity(3)     = 0;

% Apply exponential gain function for view angle velocity if vr.scaleA is NaN
vr.orientation= sign(vr.orientation)*min( exp(1.4*abs(vr.orientation)^1.2) - 1, pi/2 );
velocity(4)   = (vr.orientation - heading);

% The following should never happen but just in case
velocity(~isfinite(velocity)) = 0;

%% 2D rotation matrix counter-clockwise.
function R = R2(x)
  R = [cos(x) -sin(x); sin(x) cos(x)];
end