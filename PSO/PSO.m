%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Emilio and David
%this is our first attempt at implementing a swarm intellegence algorithm
%this work was started on 2/11/2019
%we are implementing the basic particle swarm algorithm found in
%Evolutionary and Swarm Intelligence Algorithms.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is a basic particle swarm optimization implementation based on 
% Evolutionary and Swarm Intellegence Algorithms published by Springer in 
% Studies of Computational Intelligence Volume 779
%
% ::Inputs::
% -objectiveFunction-
% this is the function that your your are trying to minimize. It should be 
% declared as something like obfunc = @(x) x(1)^2 + x(2)^2 + x(3)^2;
% -dimension-
% this is a matrix of the bounds on the search space for each variable. it
% should be declared as something like domain = [-1 1; -2 4; -8 8];
% each row represents the search space for each deimension. so in the
% example above, x(1) is restriced between [-1 1]. Also, the number of rows
% must match the number variables being optimized.
% -numParticles-
% this is the number of particles that belong to the swarm. Their initial
% positions are normally distribued about each dimension.
% -c1-
% cognative scaling parameter. determines the speed at which each particle 
% moves towards its previous best position.
% -c2-
% social scaling parameter. determines the speed at which each particle 
% moves towards the global best position
% (note about scaling parameters c1 and c2)
% In basic version of PSO, c1 = c2 = 2 were chosen. With this choice, 
% particle’s speed increases without control which is good for faster 
% convergence rate but harmful for better exploitation of the search space.
% If we set c1 = c2 > 0 then particles will attract towards the average of 
% pbest and gbest. c1 > c2 setting will be beneficial for multimodal 
% problems while c2 > c1 will be beneficial for unimodal problems. Small 
% values of c1 and c2 will provide smooth particle trajectories during the 
% search procedure while larger values of c1 and c2 will be responsible for 
% abrupt movements with more acceleration.
% -maxIterations-
% the number of iterations to execute before stopping the search
% -v_max_percent-
% a scale on the upper velocity bound for each particle in each dimension.
% This just scales the length of the dimension by the value passed in. You
% want to keep this below 0.5 and probably even much lower than that.
%
% ::outputs::
% -gbest_position-
% the set of arguments(x1, x2, ... , xn) that optimize the objective funtion
% -gbest_value-
% the value returned when evaluating the objective function with
% gbest_position
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gbest_position, gbest_value] = PSO(objectiveFunction, dimension, numParticles, c1, c2, iterations, v_max_percent)
  
   

      

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
%number of dimensions in the search space    
D = size(dimension,1); 
%initialize the position matrix
x = zeros(numParticles,D);
%initialize the velocity matrix
v = zeros(numParticles,D);
%initialize the max velicty matrices
v_max = zeros(D,1);
%initialize the pbest position matrix
pbest_position = zeros(numParticles,D);
%initialize the pbest value matrix
pbest_value = zeros(numParticles,1);
%initialize the gbest position matrix
gbest_position = zeros(1,D);
%initialize the gbest value
gbest_value = inf;


%initialize the initial posisions
%use a random distribution bounded by the length of each dimension

%for each dimension
for d=1:D
    
    %get the length of the dimension
    dimension_size = dimension(d,end) - dimension(d,1);
    dimension_center = (dimension(d,end) + dimension(d,1))/2
    
    %for each particle
    for i=1:numParticles

        %normally distribute each particle about each dimension center with
        %a standard deviation 1/6th of the dimension length
        x(i,d) = normrnd(dimension_center,dimension_size/6);

    end
end

%visualize swarm if in two dimensions
if(D == 2)
    figure();
    title('PSO Visualization in 2D');
    xlabel('x-position (m)');
    ylabel('y-position (m)');
    vid_writer = VideoWriter('PSO_2D.avi');
    vid_writer.FrameRate = 5;
    open(vid_writer);
    axis([dimension(1,1) dimension(1,2) dimension(2,1) dimension(2,2)]);
    hold on;
    plotHandlePos = zeros(numParticles+1,1); % one more for gbest
    plotHandleVel = zeros(numParticles,1);
end
%visualize swarm if in three dimensions
if(D == 3)
    figure();
    title('PSO Visualization in 3D');
    xlabel('x-position (m)');
    ylabel('y-position (m)');
    xlabel('z-position');
    vid_writer = VideoWriter('PSO_3D.avi');
    vid_writer.FrameRate = 5;
    open(vid_writer);
    axis([dimension(1,1) dimension(1,2) dimension(2,1) dimension(2,2) dimension(3,1) dimension(3,2)]);
    hold on;
    plotHandlePos = zeros(numParticles+1,1); % one more for gbest
    plotHandleVel = zeros(numParticles,1);
end

%initialize the maximum velocity matrix
%this will set the maximum velocity to be the given percent of the length
%of each dimension

%for each dimension
for d=1:D
    
    %find vmax by dividing the length of the range by 2
    v_max(d) = abs((dimension(d,end) - dimension(d,1))*v_max_percent);
    
end

%initialize the velocity matrix
%use a normal distribution that keeps it within the range [-v_max, v_max]
for d=1:D
   
    %for each particle
    for i=1:numParticles
        
        %set the velocity to be noise scaled with a standard deviation 
        %1/3th of the max veliocity
        v(i,d) = normrnd(0,v_max(d)/3);
        
    end
end

%visualize swarm if in two dimensions
if(D == 2)
    %plot the initial positions and velocities
    for i=1:numParticles
           plotHandlePos(i) = plot(x(i,1),x(i,2),'o');
           plotHandleVel(i) = plot([x(i,1) x(i,1)+v(i,1)],[x(i,2) x(i,2)+v(i,2)]);
    end
    plotHandlePos(end) = plot(gbest_position(1),gbest_position(2),'o');
    frame = getframe(gcf);
    writeVideo(vid_writer,frame);
    delete(plotHandlePos);
    delete(plotHandleVel);
end
%visualize swarm if in three dimensions
if(D == 3)
    %plot the initial positions and velocities
    
    for i=1:numParticles
           plotHandlePos(i) = plot3(x(i,1),x(i,2),x(i,3),'o');
           plotHandleVel(i) = plot3([x(i,1) x(i,1)+v(i,1)],[x(i,2) x(i,2)+v(i,2)],[x(i,3) x(i,3)+v(i,3)]);
    end
    plotHandlePos(end) = plot3(gbest_position(1),gbest_position(2),gbest_position(3),'o');
    frame = getframe(gcf);
    writeVideo(vid_writer,frame);
    delete(plotHandlePos);
    delete(plotHandleVel);
end

%initialize pbest position and value
%evaluate the objective function for each particle position.
for i=1:numParticles
    
   pbest_position(i,:) = x(i,:);
   pbest_value(i) = objectiveFunction(pbest_position(i,:));

   %find the initial gbest parameters
   if(pbest_value(i) < gbest_value)
        gbest_position = pbest_position(i,:);
        gbest_value = pbest_value(i);
   end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%for each time step
for t=1:iterations

    %for each particle
    for i=1:numParticles
        %for each dimension of the search space
        for d=1:D
            
            %calculate the random scalars [0,1] to be applied to the
            %cognitive and social scaling parameters
            r1 = normrnd(0.5,1/6);
            r2 = normrnd(0.5,1/6);
            
            %calculate the velocity
            v(i,d) = v(i,d) + c1*r1*(pbest_position(i,d) - x(i,d)) + c2*r2*(gbest_position(d) - x(i,d));

            %update the position by the velocity
            x(i,d) = x(i,d) + v(i,d);
            
            %keep the position within the searchspace.
            [pos,vel] = boundPositionVelocity(x(i,d), v(i,d), dimension(d,:), v_max(d));
            
            x(i,d) = pos;
            v(i,d) = vel;
        end
        
        %evaluate objective function 
        obj_value = objectiveFunction(x(i,:));
       
        %update pbest if better
        if(obj_value < pbest_value(i))
           pbest_position(i,:) = x(i,:);
           pbest_value(i) = obj_value;
        end
        
        %update gbest if better
        if(obj_value < gbest_value)
           gbest_position = x(i,:);
           gbest_value = obj_value;
        end    
    end
    
    %visualize swarm if in two dimensions
    if(D == 2)
        %plot the positions
        for i=1:numParticles
               plotHandlePos(i) = plot(x(i,1),x(i,2),'o');
               plotHandleVel(i) = plot([x(i,1) x(i,1)+v(i,1)],[x(i,2) x(i,2)+v(i,2)]);
        end
        plotHandlePos(end) = plot(gbest_position(1),gbest_position(2),'o');
        frame = getframe(gcf);
        writeVideo(vid_writer,frame);
        delete(plotHandlePos)
        delete(plotHandleVel)
    end
    %visualize swarm if in three dimensions
    if(D == 3)
        %plot the positions
        for i=1:numParticles
               plotHandlePos(i) = plot3(x(i,1),x(i,2),x(i,3),'o');
               plotHandleVel(i) = plot3([x(i,1) x(i,1)+v(i,1)],[x(i,2) x(i,2)+v(i,2)],[x(i,3) x(i,3)+v(i,3)]);
        end
        plotHandlePos(end) = plot3(gbest_position(1),gbest_position(2),gbest_position(3),'o');
        frame = getframe(gcf);
        writeVideo(vid_writer,frame);
        delete(plotHandlePos)
        delete(plotHandleVel)
    end
end

%visualize swarm if in two dimensions
if(D == 2)
    close(vid_writer);
end
%visualize swarm if in three dimensions
if(D == 3)
    close(vid_writer);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end



%keep each particle within the bounds of the search space. If the particle
%goes outside of the search space, set it to the boundary value and set the
%velocity to zero.
function [boundedPosition,boundedVelocity] = boundPositionVelocity(position, velocity, bounds, max_velocity)

    maxbound = max(bounds);
    minbound = min(bounds);

    %enforce upper bound on position
    if(position > maxbound)
       position = maxbound;
       velocity = 0;
    end
    %enforce lower bound on position
    if(position < minbound)
       position = minbound;
       velocity = 0;
    end
    
    %enforce upper bound on velocity
    if(velocity > max_velocity)
       velocity = max_velocity;
    end
    %enforce lower bound on velocity
    if(velocity < -max_velocity)
       velocity = -max_velocity;
    end
    
    %return the bounded position and velocity
    boundedPosition = position;
    boundedVelocity = velocity;
    
end

