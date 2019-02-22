clc;clear;close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This swarm controller is based on 
% Distributed UAV Swarm Formation Control via
% Object-Focused, Multi-Objective SARSA
%
% This work was started by David Stier on 2/20/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%number of agents (swarm size)
numAgents = 10;
agentRange = 500;

%number of obstacles
numObstacles = 2;

%bounds on initial position
initBounds = [-50 50;
               -50 50;
               -50 50];

%where the target should be placed
targetPos = [40;40;10];
         

%create target object
target = Target(targetPos);

%randomly place obstacles
for i=1:numObstacles
        
        position = zeros(size(initBounds,1),1);
    
        %set the position to be normally distributed within the bounds
        %with a standard deviation 1/6th of the initialization region in
        %each axis
        for j=1:size(initBounds,1)
            center = (initBounds(j,2)+initBounds(j,1))/2;
            sdt_dev = (initBounds(j,2)-initBounds(j,1))/6;
            position(j) = normrnd(center,sdt_dev);
        end

        obstacle(i) = Obstacle(position); 
    
end


%randomly place the swarm agents within the 1st quarter of the space
for i=1:numAgents
        
        position = zeros(size(initBounds,1),1);
    
        %set the position to be normally distributed within the bounds
        %with a standard deviation 1/12th of the initialization region in
        %each axis
        for j=1:size(initBounds,1)
            center = initBounds(j,1) + (initBounds(j,2)-initBounds(j,1))/4;
            sdt_dev = (initBounds(j,2)-initBounds(j,1))/12;
            position(j) = normrnd(center,sdt_dev);
        end

        agent(i) = Agent(position,agentRange); 
        
        %upate the target state while we are here
        %this function (for now) automatically accounts for the target
        %being within range
        agent(i) = agent(i).updateTargetState(target);
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    