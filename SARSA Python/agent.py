import action
import numpy as np
import module as module

##############################################################################
#   Agent class
##############################################################################

#the objects that will be trained on and carry out the distributed policy 
class Agent:
    
    module_weights = [1] #these are the weights for each module. they should sum to 1
    
    #class constructor
    def __init__(self,pos):
        self.position = pos         #the positon of the agent
        self.total_reward = 0       #running reward received by the agent
        self.modules = []           #a list of modules that the agent carries out
        self.modules.append(module.CohesionModule(self)) #cohesion module makes the agents stay together as a swarm
        #self.modules.append(module.CollisionModule(self))

    #change the agent's position based on the action passed in
    def take_action(self,act):
        step_size = 2
        if act == action.Action.MOVE_PLUS_X :
            self.position = self.position + [step_size,0]
        elif  act == action.Action.MOVE_MINUS_X :
            self.position = self.position + [-step_size,0]
        elif  act == action.Action.MOVE_PLUS_Y :
            self.position = self.position + [0,step_size]
        elif  act == action.Action.MOVE_MINUS_Y :
            self.position = self.position + [0,-step_size]
        else: #act == action.Action.STAY
            self.position = self.position + [0,0]

    #add the passed in incremental reward to the agents total reward
    def add_total_reward(self,incremental_reward):
        self.total_reward = self.total_reward + incremental_reward

    # #select the next action to preform based on a softmax of each module
    # def select_next_action(self):
    #     print("select the next action please!!!")

##############################################################################
#   Agent Class
##############################################################################
    