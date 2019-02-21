classdef Agent
    
    properties
        
        %these states make up the set s as seen in equation (1)
        state_cohesion;
        state_target; %position vector from the agent to the target [x,y,z]
        state_collision;
        state_obstacle;
        
        %for now these wont change based on the state, but they probably
        %should
        action_cohesion;
        action_target = ['move_left','move_forward','move_backward','move_down','stay','move_up','move_down'];
        action_collision;
        action_obstacle;
        
        observation_range;
        position;
        
        myCoolProperty;
    end %properties
    
    methods
        function obj = Agent(position,range)
            obj.position = position;
            obj.observation_range = range;
        end
        
        %TODO dont forget to quantize this...
        %paper suggests finer quanitzation at closer ranges
        function obj = updateTargetState(obj,target)
            obj.state_target = target.position - obj.position;
        end
        
        
    end %methods
    
end