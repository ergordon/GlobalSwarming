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

# The objects that will be trained on and carry out the distributed policy 
class Agent:

    # These are the weights for each module. they should sum to 1. 
    #  If they don't, they will be scaled accordingly during initialization
    #  Also, there should be a weight entry for each module
    module_weights = Simulation.module_weights
    
    #class constructor
    def __init__(self,pos): 
        self.position = pos         # The positon of the agent
        self.total_reward = 0       # Running reward received by the agent
        self.modules = []           # A list of modules that the agent carries out
        
        # TODO: consider a priority system in addition to weighting functions. 
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


        # Make sure there is a module weight for each module
        if(len(self.modules) != len(Agent.module_weights)):
            sys.exit('number of module weights and number of modules must be the same. Fix these definitions in the Agent class')

        # Make sure the module weight list sums to 1
        if(sum(self.module_weights) != 1):
            weight_sum = sum(Agent.module_weights)
            for i in range(len(Agent.module_weights)):   
                Agent.module_weights[i] = Agent.module_weights[i]/weight_sum

    # Change the agent's position based on the action passed in
    def take_action(self,act):
        step_size = 1
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

    # Add the passed in incremental reward to the agents total reward
    def add_total_reward(self,incremental_reward):
        self.total_reward = self.total_reward + incremental_reward
        
    # Select the next action to preform based on a softmax of each module
    def select_next_action(self):

        action_weights = np.zeros(len(Action))
        for i in range(0,len(self.modules)):
            mod_action_weights = self.modules[i].get_action_weights()

            if (Simulation.ControllerType == 0): # Steve+Bucci Approach
                action_weights = action_weights + Agent.module_weights[i]*mod_action_weights 

            elif (Simulation.ControllerType == 1): # Importance Function Approach
                
                if(len(self.modules) == 1):
                    # If only using one module, just use its action weights as is
                    action_weights = mod_action_weights
                else:
                    # Normalize the weights to put them all on the same order of magnitude
                    if(np.sum(mod_action_weights) != 0):
                        mod_action_weights = mod_action_weights / np.sum(mod_action_weights)
                    else:
                        mod_action_weights = np.ones(len(Action))/len(Action)
                    
                    action_weights = action_weights + self.modules[i].get_module_weight()*mod_action_weights
            else:
                print("Level Not Yet Unlocked")
                pass            

        sum_action_weights = 0        
        # Normalize the weights to create probabilities
        if(np.sum(action_weights) == float('inf')):
            sum_action_weights = 1.7976931348623157e+308
        else:
            sum_action_weights = np.sum(action_weights)

        #TODO remember that double norm might be needed for 'inf' case.... investigate!
        if sum_action_weights == 0:
            action_weights = np.ones(len(Action))/len(Action)
        elif sum_action_weights != 1:
            action_weights = action_weights/sum_action_weights
            action_weights = action_weights/np.sum(action_weights)
        else:
            action_weights = action_weights / sum_action_weights

        # Use a discrete random variable distribution to select the next action
        x=list(map(int,Action))
        px=action_weights
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        # Set state_prime to be the selected next action
        if(Simulation.take_best_action):
            # Take the action with the highest Q value
            indices = np.argwhere(action_weights == np.amax(action_weights))
            if(len(indices) == 1):
                action_prime = Action(np.argmax(action_weights))
            else:
                # If multiple entries in the Q table row are tied for highest, randomly select one of them
                index = random.randint(0,len(indices)-1)
                action_prime = Action(indices[index])
        else:
            # Use a discrete random variable distribution to select the next action
            x=list(map(int,Action))
            px=action_weights
            sample=rv_discrete(values=(x,px)).rvs(size=1)
            action_prime = Action(sample)

        for mod in self.modules:
            mod.action_prime = action_prime 

##############################################################################
#   Agent Class
##############################################################################
    