import numpy as np
import module as module
from action import Action
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
    module_weights = Simulation.module_weights
    

    #class constructor
    #TODO consider a class that houses the main simulation logic and provides access for upper level simulation variables such as the seach space.
    #search space here allows for per-agent seach spaces, but is less 'elegant'
    def __init__(self,pos): 
        self.position = pos         #the positon of the agent
        self.total_reward = 0       #running reward received by the agent
        self.modules = []           #a list of modules that the agent carries out
        
        #TODO consider a priority system in addition to weighting functions. 
        # Also impose a restriciton on weightin functions to be in the range [0,1]
        # Then if module returns >0.95 for the weight, add a bias to its module weights
        # could maybe add bias only to select weights such as largest two
         
        ## Activate Modules
        if (Simulation.CohesionModule):
            self.modules.append(module.CohesionModule(self))
        if (Simulation.CollisionAvoidanceModule):
            self.modules.append(module.CollisionModule(self)) #collision module prevents the agents from hitting each other
        if (Simulation.OutOfBoundsModule):
            self.modules.append(module.BoundaryModule(self)) 
        if (Simulation.TargetSeekingModule):
            self.modules.append(module.TargetSeekModule(self)) #collision module prevents the agents from hitting each other
        if (Simulation.ObstacleAvoidanceModule):
            self.modules.append(module.ObstacleAvoidanceModule(self))


        #make sure there is a module weight for each module
        if(len(self.modules) != len(Agent.module_weights)):
            sys.exit('number of module weights and number of modules must be the same. Fix these definitions in the Agent class')

        #make sure the module weight list sums to 1
        if(sum(self.module_weights) != 1):
            weight_sum = sum(Agent.module_weights)
            for i in range(len(Agent.module_weights)):   
                Agent.module_weights[i] = Agent.module_weights[i]/weight_sum

    #change the agent's position based on the action passed in
    def take_action(self,act):
        step_size = 2
        if act == Action.MOVE_PLUS_X :
            self.position = self.position + [step_size,0]
        elif  act == Action.MOVE_MINUS_X :
            self.position = self.position + [-step_size,0]
        elif  act == Action.MOVE_PLUS_Y :
            self.position = self.position + [0,step_size]
        elif  act == Action.MOVE_MINUS_Y :
            self.position = self.position + [0,-step_size]
        else: #act == Action.STAY
            self.position = self.position + [0,0]

    #add the passed in incremental reward to the agents total reward
    def add_total_reward(self,incremental_reward):
        self.total_reward = self.total_reward + incremental_reward
        

    #select the next action to preform based on a softmax of each module
    def select_next_action(self):

        action_weights = np.zeros(len(Action))
        for i in range(0,len(self.modules)):
            mod_action_weights = self.modules[i].get_action_weights()
            #normalize the weights to create probabilities
            if(np.sum(mod_action_weights) != 0):
                mod_action_weights = mod_action_weights / np.sum(mod_action_weights)
            else:
                mod_action_weights = np.ones(len(Action))/len(Action)

            if (Simulation.ControllerType == 0): # Steve+Bucci Approach
                action_weights = action_weights + Agent.module_weights[i]*self.modules[i].get_action_weights()      
            elif (Simulation.ControllerType == 1): # Importance Function Approach
                action_weights = action_weights + self.modules[i].get_module_weight()*mod_action_weights
            else:
                print("Level Not Yet Unlocked")
                pass            
                

        #then select another action here.....
         #normalize the weights to create probabilities
        if(np.sum(action_weights) == float('inf')):
            sum_action_weights = 1.7976931348623157e+308
            action_weights = action_weights / sum_action_weights
        else:
            sum_action_weights = np.sum(action_weights)

        if(sum_action_weights != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)
        
        # Hack Fix. Implement when other options are depleted
        # if(np.sum(action_weights) == 0 or np.sum(action_weights) == np.inf):
        #     action_weights = np.ones(len(Action))/len(Action)
        #     print("Action Weights If "+str(action_weights))
        # else:
        #     action_weights = action_weights / np.sum(action_weights)
        #     print("Action Weights Else "+str(action_weights))

        #use a discrete random variable distribution to select the next action
        x=list(map(int,Action))
        px=action_weights
        
        #print(px)
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        #set state_prime to be the selected next action
        # action_prime = action.Action(sample) 

        #set state_prime to be the selected next action
        if(Simulation.take_best_action):
            #take the action with the highest Q value
            indices = np.argwhere(action_weights == np.amax(action_weights))
            if(len(indices) == 1):
                action_prime = Action(np.argmax(action_weights))
            else:
                #if multiple entries in the Q table row are tied for highest, randomly select one of them
                index = random.randint(0,len(indices)-1)
                action_prime = Action(indices[index])
        else:
            #use a discrete random variable distribution to select the next action
            x=list(map(int,Action))
            px=action_weights
            sample=rv_discrete(values=(x,px)).rvs(size=1)
            action_prime = Action(sample)


        for mod in self.modules:
            mod.action_prime = action_prime  


##############################################################################
#   Agent Class
##############################################################################
    