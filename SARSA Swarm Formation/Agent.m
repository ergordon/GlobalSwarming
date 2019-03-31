%this class represents the autonomous agent
classdef Agent
    
    properties
        
        %these states make up the set s as seen in equation (1)
        
       
        %TODO the target state might need to be its own object with an
        %equality operator. Probably not a big deal for just target
        %tracking, but will matter more for the other modules
        state_target; %position vector from the agent to the target [x,y,z]
        
        %temporary vector used to store what the target state would be if a
        %given action were to be performed. This is being used in
        %SoftPerformAction at the moment
        state_target_prime; 
        
        state_cohesion
        state_collision;
        state_obstacle;        
        
        
        action_scale = 1; %the number of units to move for each given action
        
        action_current = Action.stay; %initial action is to do nothing (it's the safest)
        action_next = Action.stay; %need a place to temporarily store the next action that will be taken
        
        
        %the sum of rewards over the entire learning episode
        %this is a metric that will be tracked and plotted.
        total_reward = 0;
        
        %this is the reward after performing a single action.
        %this will change with each action/iteration and is used to update 
        %the Q-table
        instant_reward = 0;
        
        %moved to enum for now. Using uniform action for all states and
        %modules for now...
        %action_target = ['move_left','move_forward','move_backward','move_down','stay','move_up','move_down'];
        %action_collision;
        %action_obstacle;
        %action_cohesion;
        
        observation_range;
        
        %this position is here for simulation purposes. It is only used to
        %find relative positions and is never given as part of any markov
        %decition process state.
        position;
        position_prime;
        
        %the q matrix that maps states to actions
        %for us the actions will always be the same, but lets do it this
        %way for the sake of convention
        Q;
        
        %a list of states corresponding to the rows in the Q matrix
        Qstates;
        
        %these could be different for each module
        alpha = 0.3; %learning factor (alpha > 0)
        gamma = 0.3; %discount factor (0<gamma<1)
        epsilon = .01; %exploration factor. 0-> greedy(exploration) inf -> exploitation.
        %TODO experiment with epsilon some more
        
    end %properties
    
    methods
        %Object constructor
        %position: location to initialize the agent
        %range: how far the agent can 'see'
        function obj = Agent(position,range)
            obj = obj.SetPosition(position);
            obj = obj.SetPositionPrime(position);
%             obj.position_prime = position;
            obj.observation_range = range;
        end
        
        %use this to keep positions to whole numbers
        function obj = SetPosition(obj,position)
            obj.position = round(position);
        end
        
        function obj = SetPositionPrime(obj,position_prime)
            obj.position_prime = round(position_prime);
        end
        
        %TODO consider moving this sort of thing into its own class (module)
        %TODO dont forget to quantize this...
        %paper suggests finer quanitzation at closer ranges
        function obj = SoftUpdateTargetState(obj,target)
            if(obj.CheckRange(target.position))
                obj.state_target_prime = round(target.position - obj.position_prime);
            else
                %set to infinity to denote no target being in range.
                %this is kinda hackish and could be handled better.
                obj.state_target_prime = ones(size(obj.position_prime)).*inf;
            end
        end
        
        %TODO dont forget to quantize this...
        %paper suggests finer quanitzation at closer ranges
        function obj = UpdateTargetState(obj,target)
            if(CheckRange(obj,target.position))
                obj.state_target = round(target.position - obj.position);
            else
                %set to infinity to denote no target being in range.
                %this is kinda hackish and could be handled better.
                obj.state_target = ones(size(obj.position)).*inf;
            end
        end
        
        %set the epsilon for this agent
        %clamp the passed in epsilon to the range (0,1]
        function obj = SetEpsilon(obj, epsilon)
            if(epsilon <=0)
                epsilon = .001;
            end
            if(epsilon >1)
               epsilon = 1; 
            end
            
            obj.epsilon = epsilon;
        end
        
        
        %check if the position vector passed in is within the observational
        %range of the agent.
        function inRange = CheckRange(obj, position)
            
            dist_squared = 0;
            
            for i=1:length(position)
               dist_squared = dist_squared + (position(i) - obj.position(i))^2;
            end
            
            
            if(dist_squared <= obj.observation_range*obj.observation_range)
                inRange = true;
            else
                inRange = false;
            end
        end
        
        %perform the action passed in. Only handles changing position for
        %now
        %
        %Example:
        %agent(1) = PerformAction(agent(1),Action.move_plusX);
        function obj = PerformAction(obj)
            
            %initialize an empty vector to modify the position by
            positionModifier = 0.*obj.position;

            %consider all possible actions
            switch obj.action_current
                case Action.move_plusX
                    positionModifier = [1;0;0];
                case Action.move_minusX
                    positionModifier = [-1;0;0];
                case Action.move_plusY
                    positionModifier = [0;1;0];
                case Action.move_minusY
                    positionModifier = [0;-1;0];
%                 case Action.move_plusZ
%                     positionModifier = [0;0;1];
%                 case Action.move_minusZ
%                     positionModifier = [0;0;-1];
                 case Action.stay
                    %do nothing 
                otherwise
                    %do nothing
            end
            
            %scale the modifying vector by a predefinied parameter.
            positionModifier = positionModifier*obj.action_scale;
            
            %update the agents position.
            obj = obj.SetPosition(obj.position + positionModifier);
        end
        
        
        %perform the action passed in. Only handles changing position for
        %now
        %
        %Example:
        %agent(1) = SoftPerformAction(agent(1),Action.move_plusX);
        function obj = SoftPerformAction(obj)
            
            %initialize an empty vector to modify the position by
            positionModifier = 0.*obj.position;
            %TODO sometimes this gave a 3x3 oh no!!!

            %consider all possible actions
            switch obj.action_current
                case Action.move_plusX
                    positionModifier = [1 0];
                case Action.move_minusX
                    positionModifier = [-1 0];
                case Action.move_plusY
                    positionModifier = [0 1];
                case Action.move_minusY
                    positionModifier = [0 -1];
%                 case Action.move_plusZ
%                     positionModifier = [0 0];
%                 case Action.move_minusZ
%                     positionModifier = [0 0];
                 case Action.stay
                    %do nothing 
                otherwise
                    %do nothing
            end
            
            %scale the modifying vector by a predefinied parameter.
            positionModifier = positionModifier*obj.action_scale;
            
            %update the agents position.
            obj = obj.SetPositionPrime(obj.position + positionModifier);
            
        end
        
        %accumulate the reward assosiated with the next state
        %
        %Example:
        %agent(1) = CollectReward();
        function obj = CollectTargetReward(obj)
            
            %the amount by which the agents total reward will be modified.
            %a negative value is equivalent to a punishment
            incremental_reward  = 0;
            
            if(obj.state_target_prime(1)==inf) %target not visible by agent
                incremental_reward = incremental_reward - 1;
            else %taget visible by agent
                
                
                %max reward
                max_reward = 20;
                %make the reward proportional to the distance but up to a
                %max.
                
                
                %lets try different tiers of rewards for how close they are
                %to drive the agents closer to the target
                dist_to_target_squared = 0;
                for i=1:length(obj.state_target_prime)
                    dist_to_target_squared = dist_to_target_squared + obj.state_target_prime(i)^2;
                    
                end
                
%                 incremental_reward = incremental_reward + (1-dist_to_target_squared/obj.observation_range^2)*max_reward;
                incremental_reward = incremental_reward + (1-sqrt(dist_to_target_squared)/obj.observation_range)*max_reward;
            end
         
            
            %add the incremental reward to the agetns total reward
            obj.total_reward = obj.total_reward + incremental_reward;
            
            %store the reward for this state->action pair.
            obj.instant_reward = incremental_reward;
        end
        
        function [obj,index] = GetQStateIndex(obj,state)
           
            %first check if the current state exists within the Qstates list
           index = -1;
           for i=1:size(obj.Qstates,1) %TODO do i need to check if initialized first?

                state_i = obj.Qstates(i,:);
                if(isequal(state_i, state))
                    index = i;
                    break;%TODO make sure this only exits the for loop
                end
           end
           
           %next insert the state into the Q-table if not already present
           if(index == -1)
               obj.Qstates = [obj.Qstates; state];
               index = size(obj.Qstates,1);
               obj.Q = [obj.Q; zeros(1,length(enumeration('Action')))];
           end
            
        end
        
        
        %select the next action to take using epsilon greedy softmax
        function obj = SelectNextAction(obj)
          
           [obj,state_index] = GetQStateIndex(obj,obj.state_target_prime);
           
           %next use the boltzman probablity mass function
           %calculate the numerator for each action
           actionProbabilities = zeros(1,length(enumeration('Action'))); 
           for i=1:length(actionProbabilities)
               
              %TODO make epsilon change over time
              %TODO is epsilon normally in the denominator or numerator?
              %the paper uses T and has denominator. I think i saw epsilon
              %from other sources in numerator.
              actionProbabilities(i) = exp(obj.Q(state_index,i)/obj.epsilon); 
               
           end
           
           %divide each value by the sum of the numerators (is this really
           %necessary??? maybe later when we use continous output...)
           %TODO implement this later if it is needed
           

           %select the action that returns the highest yield
           [max_value, index] = max(actionProbabilities);
           index = find(actionProbabilities == max_value);
           if(length(index) > 1)
                index = randsample(index,1);
           end
           a = enumeration('Action');
           obj.action_next = a(index); 
           
        end
        
        %update the Q matrix
        function obj = updateQ(obj)
            
            %need current state index
            [obj,state_index] = GetQStateIndex(obj,obj.state_target);
           
            %need next state index
            [obj,next_state_index] = GetQStateIndex(obj,obj.state_target_prime);
           
            a = enumeration('Action');
            [sv,action_index] = intersect(a,obj.action_current);
            [sv,next_action_index] = intersect(a,obj.action_next);
            
            obj.Q(state_index,next_action_index) = obj.Q(state_index,action_index) ...
                + obj.alpha*(obj.instant_reward ...
                + obj.gamma*obj.Q(next_state_index,action_index) ...
                - obj.Q(state_index,action_index));
            
            %TODO there needs to be another entry in the Q function here
            %somehere. why isnt it getting updated???
%             obj.Q
%             obj.Qstates
            
        end
        
    end %methods
    
end