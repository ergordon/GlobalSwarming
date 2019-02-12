%%
%Emilio and David
%this is our first attempt at implementing a swarm intellegence algorithm
%this work was started on 2/11/2019
%we are implementing the basic particle swarm algorithm found in
%Evolutionary and Swarm Intelligence Algorithms.
%we will also optimize the example given in that pdf.

clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Input Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


maxIterationsPerImprovement = 100; %the maximum number of iterations after finding a new global best. This is the stopping criteria. Note: A different stopping criteria could be defined.
g = 0; %the index of the best particle in the swarm
c1 = .2; %cognative scaling parameter. determines the speed at which each particle moves towards its previous best position
c2 = .2; %social scaling parameter. determines the speed at which each particle moves towards the global best position
%{
note about these scaling parameters
In basic version of PSO, c1 = c2 = 2 were
chosen. With this choice, particle’s speed increases without control which is good
for faster convergence rate but harmful for better exploitation of the search space.
If we set c1 = c2 > 0 then particles will attract towards the average of pbest and
gbest. c1 > c2 setting will be beneficial for multimodal problems while c2 > c1 will
be beneficial for unimodal problems. Small values of c1 and c2 will provide smooth
particle trajectories during the search procedure while larger values of c1 and c2 will
be responsible for abrupt movements with more acceleration.
%}


%example from book.
%Minf(x1,x2) = x1^2 + x2^2; where x1,x2 are in the range [-5,5]
D = 2;
SbyDim = [5,4]; % particles along the 1st dimension, 4 along the second dimension. total of 20 particles
S = 1;  
for i=1:D
    S = S*SbyDim(i);
end
ranges = [-5 5;  %search space for first dimension
          -4 4]; %search space for second dimension

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Input Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      
%initialize the position matrix
x = zeros(S,D);
%initialize the velocity matrix
v = zeros(S,D);
%initialize the max velicty matrices
v_max = zeros(D,1);
%initialize the pbest position matrix
pbest_position = zeros(S,D);
%initialize the pbest value matrix
pbest_value = zeros(S,1);
%initialize the gbest position matrix
gbest_position = zeros(1,D);
%initialize the gbest value
gbest_value = nan;


%TODO actually implement this
%{
%initialize the initial positions
%lets equally space them in a grid and then add a bit of noise to randomize
%it a bit. Scale the noise by the grid size in some fasion.

%for each dimension
for d1=1:D 

    %determine how far to space each particle by dividing the length of the
    %range by the dimension of particles
    step_size = (ranges(d1,end) - ranges(d1,1))/(SbyDim(d1)+1);
    
    %TODO make this work beyond the 2D case....
    %for each 'column' of particles
    for s1=1:SbyDim(d1)
        
        %for each other dimension
        for d2=1:D
           if(d2~=d1)
               %for each 'row' of particles
                for s2=1:SbyDim(d2)
                    if(s2~=s1)
                    
                        noise = normrnd(0,step_size/6);
                        x(i,d) = step_size*i + noise;
                        
                    end
                end
           end
        end
        
        
        
%         %create noise scaled with a standard deviation 1/6th of the
%         %stepsize size and apply it to the equally spaced position
%         noise = normrnd(0,step_size/6);
%         x(i,d) = step_size*i + noise;
        
    end
end
%}

%initialize the initial posisions
%use a random distribution bounded by the range of each dimension



%for each dimension
for d=1:D
    
    %get the length of the dimension
    range_size = ranges(d,end) - ranges(d,1);

    %for each particle
    for i=1:S

        %create noise scaled with a standard deviation 1/6th of the
        %dimension range size and apply
        x(i,d) = normrnd(0,range_size/6);

    end
end

figure();
title('PSO for minf(x1,x2)=x1^{2}+x2^{2}');
xlabel('x-position (m)');
ylabel('y-position (m)');
vid_writer = VideoWriter('PSO.avi');
vid_writer.FrameRate = 5;
open(vid_writer);
axis([ranges(1,1) ranges(1,2) ranges(2,1) ranges(2,2)]);
hold on;
plotHandlePos = zeros(S,1);
plotHandleVel = zeros(S,1);




%initialize the maximum velocity matrix
%this will set the maximum velocity to be half of the lenght of each
%dimension

%for each dimension
for d=1:D
    
    %find vmax by dividing the length of the range by 2
    v_max(d) = (ranges(d,end) - ranges(d,1))/2;
    
end

%initialize the velocity matix
%use a normal distribution that keeps it within the range [-v_max, v_max]
for d=1:D
   
    %for each particle
    for i=1:S
        
        %set the velocity to be noise scaled with a standard deviation 
        %1/3th of the max veliocity
        v(i,d) = normrnd(0,v_max(d)/3);
        
    end
end


%plot the initial positions and velocities
for i=1:S
       plotHandlePos(i) = plot(x(i,1),x(i,2),'o');
       plotHandleVel(i) = plot([x(i,1) x(i,1)+v(i,1)],[x(i,2) x(i,2)+v(i,2)]);
end

frame = getframe(gcf);
writeVideo(vid_writer,frame);
    
delete(plotHandlePos);
delete(plotHandleVel);

    
%initializing the gbest postion and value 
%use the first particle to start
gbest_position = x(1,1:end);
gbest_value = objectiveFunction(gbest_position)

%initialize pbest position and value
%evaluate the objective function for each particle position.
for i=1:S
    
   pbest_position(i,:) = x(i,:)
   pbest_value(i) = objectiveFunction(pbest_position(i,:))
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

search = true;
counter = 0;
%for each time step
% for t=1:maxIterations %TODO replace with while and convergence criteria

%for each iteration
while(search)
    
    %increase the counter each update
    counter = counter + 1;
    
    %if we haven't found a new best solution before the counter gets to the
    %set max, stop the search
    if(counter >= maxIterationsPerImprovement)
        search = false;
    end
    
    
    %for each particle
    for i=1:S 
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
            [pos,vel] = boundPositionVelocity(x(i,d), v(i,d), ranges(d,:));
            
            x(i,d) = pos;
            v(i,d) = vel;
        end
        
        %update fitness
        
        %evaluate objective function 
        obj_value = objectiveFunction(x(i,:));
       
        note = 'check'
        ov = obj_value
        pbv = pbest_value(i)
        %update pbest
        if(obj_value < pbest_value(i))
           note = 'update'
           pbest_position(i,:) = x(i,:);
           pbest_value(i) = obj_value;
        end
        
        %update gbest
        if(obj_value < gbest_value)
          
           %we found a new best, reset the iteration coutner;
           counter = 0;
            
           %store the result
           gbest_position = x(i,:);
           gbest_value = obj_value
        end    
    end
    
    
    %plot the positions
    for i=1:S
           plotHandlePos(i) = plot(x(i,1),x(i,2),'o');
           plotHandleVel(i) = plot([x(i,1) x(i,1)+v(i,1)],[x(i,2) x(i,2)+v(i,2)]);
    end

    frame = getframe(gcf);
    writeVideo(vid_writer,frame);
    
    delete(plotHandlePos)
    delete(plotHandleVel)
end

close(vid_writer);

gbest_position
gbest_value

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Main Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Objective Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%this is the function that we are trying to optimize.
%in our case we are using the example funciton from book.
%Minf(x1,x2) = x1^2 + x2^2;
function result = objectiveFunction(position)%TODO consider a better name than position...

    %evaluate the objective function at this position
    result = position(1)^2 + position(2)^2;     
        
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Objective Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin Bounding Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%keep each particle within the bounds of the search space. If the particle
%goes outside of the search space, set it to the boundary value and set the
%velocity to zero.
% function [boundedPosition,boundedVelocity] = boundPositionVelocity(position, velocity, bounds)
% 
%     %for each particle
%     for i=1:size(position,1)
%         %for each dimension
%         for d=1:size(bounds,1)
%             
%             %enforce upper bound
%             if(position(i,d) > bounds(d))
%                position(i,d) = bounds(d);
%                velocity(i,d) = 0;
%             end
%             %enforce lower bound
%             if(position(i,d) < bounds(d))
%                position(i,d) = bounds(d);
%                velocity(i,d) = 0;
%             end
%             
%         end
%         
%     end
% 
%     %return the bounded position and velocity
%     boundedPosition = position;
%     boundedVelocity = velocity;
%     
% end

function [boundedPosition,boundedVelocity] = boundPositionVelocity(position, velocity, bounds)

    maxbound = max(bounds);
    minbound = min(bounds);

    %enforce upper bound
    if(position > maxbound)
       position = maxbound;
       velocity = 0;
    end
    %enforce lower bound
    if(position < minbound)
       position = minbound;
       velocity = 0;
    end
    
    %return the bounded position and velocity
    boundedPosition = position;
    boundedVelocity = velocity;
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Bounding Function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


