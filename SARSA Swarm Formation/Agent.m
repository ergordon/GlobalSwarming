%this class represents the autonomous agent
classdef Agent
    
    properties
        
        %these states make up the set s as seen in equation (1)
        state_cohesion;
        state_target; %position vector from the agent to the target [x,y,z]
        state_collision;
        state_obstacle;        
        
        action_scale = 1; %the number of units to move for each given action
        
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
        
        %the q matrix that maps states to actions
        %for us the actions will always be the same, but lets do it this
        %way for the sake of convention
        Q;
        
    end %properties
    
    methods
        function obj = Agent(position,range)
            obj.position = position;
            obj.observation_range = range;
        end
        
        %TODO dont forget to quantize this...
        %paper suggests finer quanitzation at closer ranges
        function obj = updateTargetState(obj,target)
            if(CheckRange(obj,target.position))
                obj.state_target = target.position - obj.position;
            else
                %set to infinity to denote no target being in range.
                %this is kinda hackish and could be handled better.
                obj.state_target = ones(size(obj.position)).*inf;
            end
        end
        
        %check if the position vector passed in is within the observational
        %range of the agent.
        function inRange = CheckRange(obj,position)
            dist_squared = 0;
            for i=1:length(position)
               dist_squared = dist_squared + position(i)^2;
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
        function obj = PerformAction(obj,action)
            
            %initialize an empty vector to modify the position by
            positionModifier = 0.*obj.position;

            %consider all possible actions
            switch action
                case Action.move_plusX
                    positionModifier = [1;0;0];
                case Action.move_minusX
                    positionModifier = [-1;0;0];
                case Action.move_plusY
                    positionModifier = [0;1;0];
                case Action.move_minusY
                    positionModifier = [0;-1;0];
                case Action.move_plusZ
                    positionModifier = [0;0;1];
                case Action.move_minusZ
                    positionModifier = [0;0;-1];
                 case Action.stay
                    %do nothing 
                otherwise
                    %do nothing
            end
            
            %scale the modifying vector by a predefinied parameter.
            positionModifier = positionModifier*obj.action_scale;
            
            %update the agents position.
            obj.position = obj.position + positionModifier;
        end
        
    end %methods
    
end