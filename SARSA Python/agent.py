import action
import numpy as np
import module as module
from simulation import Simulation
import sys
import random
from scipy.stats import rv_discrete

##############################################################################
#   Agent class
##############################################################################

#the objects that will be trained on and carry out the distributed policy 
class Agent:

    #these are the weights for each module. they should sum to 1. 
    #If they don't, they will be scaled accordingly during initialization
    #also, there should be a weight entry for each module
    module_weights = [1]#[0.25,0.75] 
    
    #class constructor
    #TODO consider a class that houses the main simulation logic and provides access for upper level simulation variables such as the seach space.
    #search space here allows for per-agent seach spaces, but is less 'elegant'
    def __init__(self,pos): 
        self.position = pos         #the positon of the agent
        self.total_reward = 0       #running reward received by the agent
        self.modules = []           #a list of modules that the agent carries out
        
        # self.modules.append(module.CohesionModule(self)) #cohesion module makes the agents stay together as a swarm
        self.modules.append(module.CollisionModule(self)) #collision module prevents the agents from hitting each other
        # self.modules.append(module.BoundaryModule(self)) #boundary module prevents the agents from leaving the search space

        #make sure there is a module weight for each module
        if(len(self.modules) != len(self.module_weights)):
            sys.exit('number of module weights and number of modules must be the same. Fix these definitions in the Agent class')

        #make sure the module weight list sums to 1
        if(sum(self.module_weights) != 1):
            weight_sum = sum(self.module_weights)
            for i in range(len(self.module_weights)):   
                self.module_weights[i] = self.module_weights[i]/weight_sum

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

    #select the next action to preform based on a softmax of each module
    def select_next_action(self):
        # print("selecting the next action!!!")
        action_weights = np.zeros(len(action.Action))
        for i in range(0,len(self.modules)):
            action_weights = action_weights + self.module_weights[i]*self.modules[i].get_action_weights()

        # print(action_weights)
        #then select another action here.....
         #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(action.Action))/len(action.Action)

        #use a discrete random variable distribution to select the next action
        x=list(map(int,action.Action))
        px=action_weights
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        #set state_prime to be the selected next action
        action_prime = action.Action(sample) 

        for mod in self.modules:
            mod.action_prime = action_prime  

##############################################################################
#   Agent Class
##############################################################################
    