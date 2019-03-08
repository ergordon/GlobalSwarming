import action
import numpy as np
import module as module

class Agent:
    
    module_weights = [1] #these are the weights for each module. they should sum to 1
    total_reward = 0 #the total reward accumulated by the agent 

    def __init__(self,pos):
        self.position = pos
        self.total_reward = 0 #running reward received by the agent
        self.modules = []
        self.modules.append(module.CohesionModule(self))
        #self.modules.append(module.CollisionModule(self))
        
    def change_position(self,pos):
        self.position = pos

    def take_action(self,act):
        if act == action.Action.MOVE_PLUS_X :
            self.position = self.position + [1,0]
        elif  act == action.Action.MOVE_MINUS_X :
            self.position = self.position + [-1,0]
        elif  act == action.Action.MOVE_PLUS_Y :
            self.position = self.position + [0,1]
        elif  act == action.Action.MOVE_MINUS_Y :
            self.position = self.position + [0,-1]
        else: #act == action.Action.STAY
            self.position = self.position + [0,0]

    #TODO keep a history of the reward over time
    def add_total_reward(self,reward):
        self.total_reward = self.total_reward + reward

    #select the next action to preform based on a softmax of each module
    def select_next_action(self):
        print("select the next action please!!!")