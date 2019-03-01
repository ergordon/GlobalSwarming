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
numAgents = 1;
agentRange = 10;

%number of obstacles
numObstacles = 2;

%bounds on initial position
initBounds = [30 50;
               30 50];

%where the target should be placed
targetPos = [40 40];
         
%start epsilon at almost zero for each particle
%encourages exploration
%then linearly(for now) increase epsilon to 1 over the traning period
epsilon = 0.001;

%create target object
target = Target(targetPos);

%randomly place obstacles
for i=1:numObstacles
        
        position = zeros(1,size(initBounds,1));
    
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
        
        position = zeros(1,size(initBounds,1));
    
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
        agent(i) = agent(i).UpdateTargetState(target);
        
        %set the initial epsilon value
        agent(i) = agent(i).SetEpsilon(epsilon);
end


%initialize plot
vid_writer = VideoWriter('SARSA.avi');
vid_writer.FrameRate = 60;
open(vid_writer);
figure(); 

% plotHandlePos = zeros(numAgents + 1);
% %plot the initial positions and velocities
% for i=1:numAgents
% %        plotHandlePos(i) = plot3(agent(i).position(1),agent(i).position(2),agent(i).position(3),'o');
%        plotHandlePos(i) = plot(agent(i).position(1),agent(i).position(2),'o');
%        hold on;
% end
% % plotHandlePos(end) = plot3(target.position(1),target.position(2),target.position(3),'*');
% plotHandlePos(end) = plot(target.position(1),target.position(2),'*');
% frame = getframe(gcf);
% writeVideo(vid_writer,frame);
% delete(plotHandlePos);
% axis([initBounds(1,:) initBounds(2,:)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    


% continue_learning = true;
% learning_counter = 1;
% max_learning_counter = 100;
maxEpisodes = 1000;
episodeCounter = 1;
%each iteration of this while loop is a single learning epsisode.
%terminate based on a max number of iterations or a convergence of the Q
%matrices
while(episodeCounter < maxEpisodes)
    
    %TODO reset positions to random placement
    %randomly place the swarm agents within the 1st quarter of the space
    for i=1:numAgents

            position = zeros(1,size(initBounds,1));

            %set the position to be normally distributed within the bounds
            %with a standard deviation 1/12th of the initialization region in
            %each axis
            for j=1:size(initBounds,1)
                center = initBounds(j,1) + (initBounds(j,2)-initBounds(j,1))/4;
                sdt_dev = (initBounds(j,2)-initBounds(j,1))/12;
                position(j) = normrnd(center,sdt_dev);
            end

            %agent(i) = Agent(position,agentRange); 
            %reset the initial position
            agent(i) = agent(i).SetPosition(position);
            
            %upate the target state while we are here
            %this function (for now) automatically accounts for the target
            %being within range
            agent(i) = agent(i).UpdateTargetState(target);
            
            %update epsilon
%             epsilon = 1/maxEpisodes*episodeCounter;
            epsilon = exp(episodeCounter-100);
            if(epsilon>1)
                epsilon = 1;
            end
            agent(i) = agent(i).SetEpsilon(epsilon);
    end
    
    
    continue_episode = true;
    max_iterations = 100;
    counter = 1;
    %each iteration of this while loop is a single time step.
    %terminate based on a max amount of time or acheiving the goal or some
    %other failure criteria (collision?)
    while(continue_episode)
        
%         %plot it
%         %plot the initial positions and velocities
%         for i=1:numAgents
% %            plotHandlePos(i) = plot3(agent(i).position(1),agent(i).position(2),agent(i).position(3),'o');
%            plotHandlePos(i) = plot(agent(i).position(1),agent(i).position(2),'o');
%         end
% %         plotHandlePos(end) = plot3(target.position(1),target.position(2),target.position(3),'*');
%         plotHandlePos(end) = plot(target.position(1),target.position(2),'*');
%         frame = getframe(gcf);
%         writeVideo(vid_writer,frame);
%         delete(plotHandlePos);

        %execute the Q-learning algorithm on each agent to update their Q
        %matrices and select their next action
        for i=1:numAgents
           
            %(initialized action or action from previou step) (maybe or may not use this scheme)
            %store the position result that would occur if the agent took its
            %currently queued action.
            agent(i) = agent(i).SoftPerformAction();

%             delta_pos =  agent(i).position_prime - agent(i).position
            %store the target_state result that would occur if the agent took its
            %currently queued action.
            %TODO handle mutliple targets???
            agent(i) = agent(i).SoftUpdateTargetState(target);
            
            %collect the reward for the new state
            agent(i) = agent(i).CollectTargetReward();
           
            
            %select the next action (a') to take by softmax exploratoin
            agent(i) = agent(i).SelectNextAction();
            
            
            agent(i) = agent(i).updateQ();
            
        end
        
        %perform the actions selected from the previous time step. we do
        %this after each agent has selected its action because a
        %distributed swarm controller would operate simultaneously on each
        %agent.
        for i=1:numAgents
            
            agent(i).action_current = agent(i).action_next;
            agent(i) = agent(i).SetPosition(agent(i).position_prime);
            agent(i) = agent(i).UpdateTargetState(target);

        end
        
        counter = counter+1;
        if(counter == max_iterations)
            continue_episode = false;
        end
    end
    
    %here for debugging
%     updated_Q = agent(1).Q

    
    episodeCounter = episodeCounter + 1;
   
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Q Table Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

close all;

%initialize plot
vid_writer = VideoWriter('SARSA.avi');
vid_writer.FrameRate = 60;
open(vid_writer);
figure(); 
hold on;
axis([30 50 30 50])
agent(1).SetPosition([45 45]);

max_test_iterations = 1000;
for j=0:max_test_iterations

    for i=1:numAgents
  
        %plot the initial positions and velocities
        for i=1:numAgents
%            plotHandlePos(i) = plot3(agent(i).position(1),agent(i).position(2),agent(i).position(3),'o');
           plotHandlePos(i) = plot(agent(i).position(1),agent(i).position(2),'o');
        end
%         plotHandlePos(end) = plot3(target.position(1),target.position(2),target.position(3),'*');
        plotHandlePos(end) = plot(target.position(1),target.position(2),'*');
        frame = getframe(gcf);
        writeVideo(vid_writer,frame);
        delete(plotHandlePos);
        
        
        %select the next action (a') to take by softmax exploratoin
        agent(i) = agent(i).SelectNextAction();
            
        agent(i).action_current = agent(i).action_next;
        agent(i) = agent(i).PerformAction();
        
        agent(i) = agent(i).SetPosition(agent(i).position_prime);
        agent(i) = agent(i).UpdateTargetState(target);
        
    end
    
    
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Q Table Test
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    



