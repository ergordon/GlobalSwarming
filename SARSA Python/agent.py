import numpy as np
import module as module
from action import Action
from action import ActionHelper
from simulation import Simulation
from simulation import Controller
import sys
import random
from scipy.stats import rv_discrete

##############################################################################
#   Agent class
##############################################################################

# The objects that will be trained on and carry out the distributed policy 
class Agent:

    
    #class constructor
    def __init__(self,pos): 
        self.position = pos         # The positon of the agent
        self.total_reward = 0       # Running reward received by the agent
        self.modules = []           # A list of modules that the agent carries out

        # These are the weights for each module. they should sum to 1. 
        #  If they don't, they will be scaled accordingly during initialization
        #  Also, there should be a weight entry for each module
        self.module_weights = Simulation.module_weights
        
        #the combined action weights determined during select_next_action()
        # stored for use with continuous steering
        self.action_weights = np.zeros(len(Action))

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
        if(len(self.modules) != len(self.module_weights)):
            print(len(self.module_weights),len(self.modules) )
            sys.exit('number of module weights and number of modules must be the same. Fix these definitions in the Agent class')

        # Make sure the module weight list sums to 1
        if(sum(self.module_weights) != 1):
            weight_sum = sum(self.module_weights)
            for i in range(len(self.module_weights)):   
                self.module_weights[i] = self.module_weights[i]/weight_sum

    # Change the agent's position based on the action passed in
    def take_action(self,act):
        
        step_size = 1#Simulation.agent_step_size
        step_angle = 0
        step_scalar = 0

        if Simulation.continuous_steering: 
            #handle direciton first. (other code exists for finding highest action index)
                
            action_leading = act
            if act == Action.STAY:
                # print(self.action_weights)
                # print(self.action_weights[:-1])
                indices = np.argwhere(self.action_weights[:-1] == np.amax(self.action_weights[:-1]))
                    
                if(len(indices) == 1):
                    action_leading = Action(np.argmax(self.action_weights[:-1]))
                else:
                    # If multiple entries in the Q table row are tied for highest, randomly select one of them
                    index = random.randint(0,len(indices)-1)
                    action_leading = Action(indices[index])    
            
            action_CW = ActionHelper.nearest_actions[action_leading][0]
            action_CCW = ActionHelper.nearest_actions[action_leading][1]
            action_opposite = ActionHelper.opposite_actions[action_leading]

            angle_leading =  ActionHelper.action_headings[action_leading]
            angle_CW =   ActionHelper.action_headings[action_leading]
            if abs(angle_CW - angle_leading) > np.pi:
                angle_CW = 2*np.pi - angle_CW
            angle_CCW =  ActionHelper.action_headings[action_leading]
            if abs(angle_CCW - angle_leading) > np.pi:
                angle_CCW = 2*np.pi - angle_CCW
            angle_opposite = ActionHelper.action_headings[action_leading]

            weight_leading = self.action_weights[action_leading.value]
            weight_CW = self.action_weights[action_CW.value]
            weight_CCW = self.action_weights[action_CCW.value]
            weight_opposite = self.action_weights[action_opposite.value]
            weight_stay = self.action_weights[-1]

            #calculate direction
            w_CCW = 1.0/(2+(weight_leading - weight_CCW)**2)
            w_CW = 1.0/(2+(weight_leading - weight_CW)**2)

            step_angle = angle_leading + w_CCW*(angle_CCW - angle_leading) + w_CW*(angle_CW - angle_leading)

            # print('step_angle', step_angle)


            #handle step size scalar
            #NOTE: not 100% sure this will always work how we want it to
            if weight_opposite < weight_stay and act != Action.STAY:
                if weight_opposite + weight_leading + weight_stay != 0:
                    step_scalar = (weight_opposite*-1.0 + weight_leading*1.0)/(weight_opposite + weight_leading + weight_stay) 
                else:
                    step_scalar = 0
            else:
                if weight_leading + weight_stay != 0:
                    step_scalar = (weight_leading*1.0)/(weight_leading + weight_stay)
                else:
                    step_scalar = 0
            # print('step_scalar', step_scalar)

        else:
        
            if act == Action.MOVE_PLUS_X :
                step_scalar = 1
                step_angle = 0
            elif  act == Action.MOVE_PLUS_X_PLUS_Y :
                step_scalar = 1
                step_angle = 0.25*np.pi
            elif  act == Action.MOVE_PLUS_Y :
                step_scalar = 1
                step_angle = 0.5*np.pi
            elif  act == Action.MOVE_MINUS_X_PLUS_Y :
                step_scalar = 1
                step_angle = 0.75*np.pi
            elif  act == Action.MOVE_MINUS_X :
                step_scalar = 1
                step_angle = np.pi
            elif  act == Action.MOVE_MINUS_X_MINUS_Y :
                step_scalar = 1
                step_angle = 1.25*np.pi
            elif  act == Action.MOVE_MINUS_Y :
                step_scalar = 1
                step_angle = 1.5*np.pi
            elif  act == Action.MOVE_PLUS_X_MINUS_Y :
                step_scalar = 1
                step_angle = 1.75*np.pi
            else: #act == Action.STAY
                step_scalar = 0
                

        step_vector = step_size*step_scalar*np.array([np.cos(step_angle), np.sin(step_angle)])
        self.position = self.position + step_vector

    # Add the passed in incremental reward to the agents total reward
    def add_total_reward(self,incremental_reward):
        self.total_reward = self.total_reward + incremental_reward


    def normalize(vector, range): # range should be (lower_bound, upper_bound) 
        a = np.max(vector)
        c = np.min(vector)
        b = range[1]
        d = range[0]

        m = (b - d) / (a - c)
        pslope = (m * (vector - c)) + d
        return pslope


    def ranked_importance_select_next_action(self):
        ranked_if = True
        if ranked_if:
            #lists used for IF MMAS
            aw_set = []
            importance_set = []
            module_tier_set = []
            module_name_set = []

            #TODO handle one object case!
            for i in range(0,len(self.modules)):
                mod_action_weights = self.modules[i].get_action_weights()
                mod_weights = self.modules[i].get_module_weights()
                for w in range(0,len(mod_action_weights)):
                    aw_set.append(mod_action_weights[w])
                    importance_set.append(mod_weights[w])
                    module_tier_set.append(self.modules[i].mmas_tier)
                    module_name_set.append(self.modules[i].__class__.__name__)

            debug = False
            if debug:
                print('')
                print('')
                print('RESULTS FOR IF MMAS ')
                print('')
            
            restricted_action_list = []
            combined_aw_list = []

            # normalize the action weights in a way that mostly decouples the reward scheme from the MMAS
            diff_normed_weights = []
            
            
            for i in range(0,len(aw_set)):
                if not max(aw_set[i]) - min(aw_set[i]) == 0:
                    norm = Agent.normalize(aw_set[i], np.array([-1,1]))
                else:
                    norm = np.zeros(len(Action))

                diff_normed_weights.append(norm)
                # diff_normed_weights.append(Agent.diff_norm(aw_set[i]))

            if debug:
                print('importances')
                for x in range(0,len(module_name_set)):
                    print(module_name_set[x] + str(importance_set[x]))
                # print('importances')
                # for x in range(0,len(module_name_set)):
                #     print()

                print('diff_normed_weights')
                for i in range(0,len(diff_normed_weights)):
                    print(module_name_set[i])
                    print(diff_normed_weights[i])

            # next find out the number of tiers
            num_tiers = np.max(module_tier_set) + 1
            if debug:
                print('num tiers', num_tiers) 


            #start by combining the higest tier, then the highest tier with the next highest, and so on.
            #dont forget to store the results from each tier combination
            for t in range(0,num_tiers):
                restricted_action_list.append(np.array([]))
                combined_aw_list.append(np.array([]))

                if t in module_tier_set:
                    combined_aw = np.zeros(len(Action))
                    
                    for w in range(0,len(aw_set)):
                        if module_tier_set[w] <= t:
                            for a in range(0,len(aw_set[w])):
                                combined_aw[a] = combined_aw[a] + aw_set[w][a]*importance_set[w]

                    #create a restricted action list for each tier
                    restricted_actions = np.where(combined_aw == combined_aw.min())
                    if debug:
                        print('restricted_actions', restricted_actions[0])

                    #if all weights are the same, there is not preference so none should be restricted
                    if len(restricted_actions[0]) != len(Action):
                        restricted_action_list[t] = restricted_actions
                    
                    combined_aw_list[t] = combined_aw
                    if debug:
                        print('combined_aw', combined_aw)
                        print('')
                    # combined_diff_norm = diff_norm(combined_aw)

                    # print('combined_diff_norm', combined_diff_norm)

                    # probabilities = soft_max(combined_diff_norm)

                    # print('action probabilities', probabilities)

            if debug:
                print('restricted action list', restricted_action_list)

            # now select an action by starting with the lowest tier and 
            # iterating over its actions in descending order of preference
            # and seeing which one is not a least preferable action of any
            # of the higher tiers. if all actions are included in the least
            # preferable actions of the above tiers, neglect that tier and
            # repeat the process with each tier above it until an action can
            # be selected

            action_to_take = -1 #defaults to stay for the action selector
            action_taken = False

            for t in range(num_tiers-1,-1,-1):
                # arr = np.array([2,7,3,6,1])
                # sort = np.argsort(arr)
                # print(np.argsort(arr))
                # for i in range(len(arr)-1,-1,-1):
                #     print(arr[sort[i]])

                #have to check if top tier or if there are even any tiers above us...
                top_tier_reached = True
                # if t == 0:
                #     top_tier_reached == true
                # else:

                if t != 0:
                    for u in range(t-1,-1,-1):
                        if combined_aw_list[u].size == len(Action):
                            top_tier_reached = False

                #how to handle repeated entries???
                if top_tier_reached:
                    if debug:
                        print('TOP TIER REACHED')
                    #handle duplicate entries
                    indices = np.argwhere(combined_aw_list[t] == np.amax(combined_aw_list[t]))
                    if(len(indices) == 1):
                        # action_prime = Action(np.argmax(combined_aw_list[t]))
                        if debug:
                            print('action taken is', np.argmax(combined_aw_list[t]))
                        action_to_take = np.argmax(combined_aw_list[t])
                    else:
                        # If multiple entries in the Q table row are tied for highest, randomly select one of them
                        index = random.randint(0,len(indices)-1)
                        # action_prime = Action(indices[index])
                        if debug:
                            print('action taken is', indices[index])
                        action_to_take = indices[index]
                    

                    # print('action taken is', np.argmax(combined_aw_list[t]))
                    action_taken = True
                    
                else:    
                    aw_sort = np.argsort(combined_aw_list[t])
                    if debug:
                        print('aw_sort',aw_sort)

                    #take best action case
                    if aw_sort.size == len(Action):
                        for w in range(len(aw_sort)-1,-1,-1):
                            if debug:
                                print('combined_aw_list[t][aw_sort[w]]',combined_aw_list[t][aw_sort[w]])
                                print('aw_sort[w]',aw_sort[w])
                            #starting with the highest reward 

                            #get the set of indices that match this value.
                            duplicates = np.where(combined_aw_list[t] == combined_aw_list[t][aw_sort[w]])
                            if debug:
                                print('duplicates[0]', duplicates[0])
                                print('len(duplicates[0])', len(duplicates[0]))
                            
                            
                            if len(duplicates[0]) > 1:
                                choices = np.random.choice(duplicates[0], len(duplicates[0]), replace=False)
                            else:
                                choices = np.array([aw_sort[w]])
                            
                            if debug:
                                print('choices', choices)
                            

                            #check if the choices are restriced in any of the upper tiers
                            choices_restricted = np.zeros(choices.shape,dtype=bool)
                            if debug:
                                print('choices_restricted', choices_restricted)
                            # eligible_action = False
                            for c in range(0,len(choices)):
                                if debug:
                                    print('testing choice ', choices[c])
                                for u in range(t-1,-1,-1):
                                    if debug:
                                        print('restricted_action_list[u]',restricted_action_list[u])
                                        print('len(restricted_action_list[u])',len(restricted_action_list[u]))
                                    if len(restricted_action_list[u]) != 0:    
                                        if debug:
                                            print('restricted_action_list[u][0]',restricted_action_list[u][0])
                                        
                                        if choices[c] in restricted_action_list[u][0]:
                                            if debug:
                                                print("it's restricted")
                                            choices_restricted[c] = True
                                            

                            #now i know which choices are restriced, loop over them
                            for c in range(0,len(choices_restricted)):
                                if not choices_restricted[c]:
                                    action_to_take = choices[c]
                                    self.action_weights = combined_aw_list[t]
                                    action_taken = True
                                    break

                            if action_taken:
                                break

                if action_taken:
                    if debug:
                        print('action taken is', action_to_take)
                    if debug:
                        print('Action is:', Action(action_to_take))
                    # break
                    
                    for mod in self.modules:
                        mod.action_prime = Action(action_to_take)
                    
                    return       
                    #TODO if not take best aciton, sample and check. how many samples though?
                    
                    #TODO implement continuous steering logic for this MMAS!
                    # self.action_weights = action_weights

            


    def biased_importance_select_next_action(self):
        # print('selecting next action')
        T = 0
        epsilon = -10000000
        action_weights = np.zeros(len(Action))
        
        transition = False
        for i in range(0,len(self.modules)):
            #check if any of the modules had a state transition, if so, choose a new action
            #this kinda sucks for running other modules with the boundary module though because it will trigger every time(for the current implementation)
            #maybe i should just make the boundary module variably discrete as well... 
            if 1 in self.modules[i].state_transition:
                transition = True
                # print('transision!!!')
                break
                
        if not transition:
            for i in range(0,len(self.modules)):
                self.modules[i].action_prime = self.modules[i].action
            
            return

        for i in range(0,len(self.modules)):

            mod_action_weights = self.modules[i].get_action_weights()
            # print(str(self.modules[i].__class__.__name__) + 'module weights', mod_action_weights)
            # print('state_prime', self.modules[i].state_prime)
            T = T + self.modules[i].get_T()
            
            epsilon = max(epsilon,self.modules[i].get_epsilon())
            # print(epsilon)
            
            if(len(self.modules) == 1 and len(self.modules[0].Q) == 1):
                # If only using one module with one q table, just use its action weights as is
                action_weights = mod_action_weights[0]
            else:

                mod_weights = self.modules[i].get_module_weights()
    
                for j in range(0,len(mod_action_weights)):

                    if not max(mod_action_weights[j]) - min(mod_action_weights[j]) == 0:
                        mod_action_weights[j] = Agent.normalize(mod_action_weights[j], np.array([-1,1]))
                    else:
                        mod_action_weights[j] = np.zeros(len(Action))

                    # Normalize the weights to put them all on the same order of magnitude
                    # asum = np.sum(np.absolute(mod_action_weights[j])) #TODO fix infinity again
                    # #TODO divide by the sum of the absolute values? 
                    # if(asum != 0):
                    #     mod_action_weights[j] = mod_action_weights[j] / asum
                    # else: #NOTE may or may not want this else statement
                            #should a row of zeros have any contribution????
                    #     # print('... action weight sum is zero')
                    #     # print(mod_action_weights[j])
                    #     mod_action_weights[j] = np.ones(len(Action))/len(Action)
                
                    #NOTE *(1+module_priorities) is a quick hack
                    action_weights = action_weights + mod_weights[j]*mod_action_weights[j]*(1+Simulation.module_priorities[i])



        if not max(action_weights) - min(action_weights) == 0:
            action_weights = Agent.normalize(action_weights, np.array([-1,1]))
        else:
            action_weights = np.zeros(len(Action))
       
        T = 1
        for i in range(0,len(action_weights)):
            action_weights[i] = np.exp(action_weights[i]/T)
       
        sum_action_weights = np.sum(action_weights)        
        
        if sum_action_weights == 0:
            action_weights = np.ones(len(Action))/len(Action)
        elif sum_action_weights != 1:
            action_weights = action_weights/sum_action_weights
            
        self.action_weights = action_weights
        
        # epsilon = epsilon/len(self.modules)
        ksi = random.uniform(0,1)

        # Set state_prime to be the selected next action
        if ksi >= epsilon or Simulation.take_best_action:
            # print('taking best action')
            # Take the action with the highest Q value
            indices = np.argwhere(action_weights == np.amax(action_weights))

            if(len(indices) == 1):
                action_prime = Action(np.argmax(action_weights))
            else:
                # If multiple entries in the Q table row are tied for highest, randomly select one of them
                index = random.randint(0,len(indices)-1)
                action_prime = Action(indices[index])
        else:
            # print('softmax')
            # Use a discrete random variable distribution to select the next action
            x=list(map(int,Action))
            px=action_weights
            sample=rv_discrete(values=(x,px)).rvs(size=1)
            action_prime = Action(sample)


        # print('action weights: ' + str(action_weights))
        for mod in self.modules:
            mod.action_prime = action_prime 
   
    def greatest_mass_select_next_action(self):
        # print('selecting next action')
        T = 0
        epsilon = 0
        action_weights = np.zeros(len(Action))
        
        transition = False
        for i in range(0,len(self.modules)):
            #check if any of the modules had a state transition, if so, choose a new action
            #this kinda sucks for running other modules with the boundary module though because it will trigger every time(for the current implementation)
            #maybe i should just make the boundary module variably discrete as well... 
            if 1 in self.modules[i].state_transition:
                transition = True
                # print('transision!!!')
                break
                
        if not transition:
            for i in range(0,len(self.modules)):
                self.modules[i].action_prime = self.modules[i].action
            
            return

        for i in range(0,len(self.modules)):

            mod_action_weights = self.modules[i].get_action_weights()
            # print(str(self.modules[i].__class__.__name__) + 'module weights', mod_action_weights)
            # print('state_prime', self.modules[i].state_prime)
            T = T + self.modules[i].get_T()
            
            # epsilon = epsilon + self.modules[i].get_epsilon()
            epsilon = max(epsilon,self.modules[i].get_epsilon())
            # print('epsilon',epsilon)
                
            if(len(self.modules) == 1 and len(self.modules[0].Q) == 1):
                # If only using one module with one q table, just use its action weights as is
                action_weights = mod_action_weights[0]
            else:
                for j in range(0,len(mod_action_weights)):
                    action_weights = action_weights + self.module_weights[i]*mod_action_weights[j]
                    
        
        if Simulation.Tokic_VDBE:
            
            if not max(action_weights) - min(action_weights) == 0:
                action_weights = Agent.normalize(action_weights, np.array([-1,1]))
            else:
                action_weights = np.zeros(len(Action))

            T = 1
            for i in range(0,len(action_weights)):
                action_weights[i] = np.exp(action_weights[i]/T)

            sum_action_weights = np.sum(action_weights)        
            if sum_action_weights == 0:
                action_weights = np.ones(len(Action))/len(Action)
            elif sum_action_weights != 1:
                action_weights = action_weights/sum_action_weights

            self.action_weights = action_weights

            # epsilon = epsilon/len(self.modules)
            ksi = random.uniform(0,1)

            # Set state_prime to be the selected next action
            if ksi >= epsilon or Simulation.take_best_action:
                # print('taking best action')
                # Take the action with the highest Q value
                indices = np.argwhere(action_weights == np.amax(action_weights))
                # if self.modules[0].action == Action.STAY:
                #     print('random ints', indices)
                    
                if(len(indices) == 1):
                    # if self.modules[0].action == Action.STAY:
                    #     print('only one :(', Action(np.argmax(action_weights)) )
                    action_prime = Action(np.argmax(action_weights))
                else:
                    # If multiple entries in the Q table row are tied for highest, randomly select one of them
                    index = random.randint(0,len(indices)-1)
                    action_prime = Action(indices[index])
            else:
                # print('softmax')
                # Use a discrete random variable distribution to select the next action
                x=list(map(int,Action))
                px=action_weights
                sample=rv_discrete(values=(x,px)).rvs(size=1)
                action_prime = Action(sample)
        else:
            
            if not max(action_weights) - min(action_weights) == 0:
                action_weights = Agent.normalize(action_weights, np.array([0,1]))
            else:
                action_weights = np.zeros(len(Action))

            T_i = 1.5
            T_f = 0.285

            #i think this shouldnt happen
            if epsilon > 1:
                print('EPSILON > 1, INVESTIGATE')
                epsilon = 1
            if epsilon < 0:
                print('EPSILON < 0, INVESTIGATE')
                epsilon = 0

            T = T_i - (1-epsilon)*(T_i - T_f)
            for i in range(0,len(action_weights)):
                action_weights[i] = np.exp(action_weights[i]/T)

            sum_action_weights = np.sum(action_weights)        
            if sum_action_weights == 0:
                action_weights = np.ones(len(Action))/len(Action)
            elif sum_action_weights != 1:
                action_weights = action_weights/sum_action_weights

            self.action_weights = action_weights

            
            # Set state_prime to be the selected next action
            if Simulation.take_best_action:
                # print('taking best action')
                # Take the action with the highest Q value
                indices = np.argwhere(action_weights == np.amax(action_weights))
                # if self.modules[0].action == Action.STAY:
                #     print('random ints', indices)
                    
                if(len(indices) == 1):
                    # if self.modules[0].action == Action.STAY:
                    #     print('only one :(', Action(np.argmax(action_weights)) )
                    action_prime = Action(np.argmax(action_weights))
                else:
                    # If multiple entries in the Q table row are tied for highest, randomly select one of them
                    index = random.randint(0,len(indices)-1)
                    action_prime = Action(indices[index])
            else:
                # print('softmax')
                # Use a discrete random variable distribution to select the next action
                x=list(map(int,Action))
                px=action_weights
                sample=rv_discrete(values=(x,px)).rvs(size=1)
                action_prime = Action(sample)

        # print('action weights: ' + str(action_weights))
        for mod in self.modules:
            mod.action_prime = action_prime 


    # Select the next action to preform based on a softmax of each module
    def select_next_action(self):

        if (Simulation.ControllerType == Controller.GreatestMass or Simulation.ControllerType == Controller.GenAlg): # Steve+Bucci Approach
            self.greatest_mass_select_next_action()
        elif (Simulation.ControllerType == Controller.Importance): # Importance Function Approach
            # self.biased_importance_select_next_action()
            self.ranked_importance_select_next_action()


##############################################################################
#   Agent Class
##############################################################################
    
