import numpy as np
import module as module
import agent as agent
import action as action
import matplotlib.pyplot as plt
import pickle
from simulation import Simulation
from qlearning import Qlearning
import time
from action import Action
import random

def diff_norm(action_weights):
    aw_tmp = action_weights
    aw_max = max(action_weights)
    aw_min = min(action_weights)
    
    for j in range(0,len(action_weights)):
        if aw_max - aw_min != 0:
            action_weights[j] = (action_weights[j] - aw_min)/(aw_max - aw_min)        
        else:
            action_weights[j] = 0

    return action_weights

def soft_max(action_weights):
    T = 0.5
    for i in range(0,len(action_weights)):
        action_weights[i] = np.exp(action_weights[i]/T)

    aw_sum = np.sum(action_weights)
    for i in range(0,len(action_weights)):
        action_weights[i] = action_weights[i]/aw_sum
    
    return action_weights


def cohesion_if(state):
    print(state)
    return 1

def collision_if(state):
    print(state)
    return 1

def boundary_if(state):
    print(state)
    return 1

def target_if(state):
    print(state)
    return 1

def obstacle_if(state):
    print(state)
    return 1


aw_list = []
state_list = []
importance_list = []
module_tier_list = []

################################
# IF MMAS thought experiment 1 #
################################
cohesion_aw = np.array([-17.3450597,	-16.93497032,	-17.86593676,	-17.25448444,	-17.99413244])
collision_aw = np.array([-0.2171946,	-0.11552542,	-4.83118414,	0,	-0.07333829])
target_aw = np.array([9.53846154,	9.43589744,	9.48589824,	9.48589824,	9.48717949])

cohesion_state = np.array([-20,0])
collision_state = np.array([0,4])
target_state = np.array([20,0])

aw_set = []
aw_set.append(cohesion_aw)
aw_set.append(collision_aw)
aw_set.append(target_aw)

state_set = []
state_set.append(cohesion_state)
state_set.append(collision_state)
state_set.append(target_state)

importance_set = []
importance_set.append(cohesion_if(cohesion_state))
importance_set.append(collision_if(collision_state))
importance_set.append(target_if(target_state))

module_tier_set = []
module_tier_set.append(3)
module_tier_set.append(1)
module_tier_set.append(2)


aw_list.append(aw_set)
state_list.append(state_set)
importance_list.append(importance_set)
module_tier_list.append(module_tier_set)

################################
# IF MMAS thought experiment 2 #
################################
collision_aw_1 = np.array([0,	-1.8796722,	0,	-1.96014686,	-0.93337602])
collision_aw_2 = np.array([-0.2171946,	-0.11552542,	-4.83118414,	0,	-0.07333829])
collision_aw_3 = np.array([0,	-4.26216529,	-1.46489139,	-0.47950066,	-0.20694544])
collision_aw_4 = np.array([-4.21389644,	0,	-0.39730469,	-0.51252411,	-0.43856837])
collision_aw_5 = np.array([-0.51032582,	-0.2136608,	0,	-4.28516757,	-0.62101071])

collision_state_1 = np.array([-3,3])
collision_state_2 = np.array([0,4])
collision_state_3 = np.array([-4,0])
collision_state_4 = np.array([4,0])
collision_state_5 = np.array([0,-4])


aw_set = []
aw_set.append(collision_aw_1)
aw_set.append(collision_aw_2)
aw_set.append(collision_aw_3)
aw_set.append(collision_aw_4)
aw_set.append(collision_aw_5)

state_set = []
state_set.append(collision_state_1)
state_set.append(collision_state_2)
state_set.append(collision_state_3)
state_set.append(collision_state_4)
state_set.append(collision_state_5)

#TODO apply importance functions
importance_set = []
importance_set.append(collision_if(collision_state_1))
importance_set.append(collision_if(collision_state_2))
importance_set.append(collision_if(collision_state_3))
importance_set.append(collision_if(collision_state_4))
importance_set.append(collision_if(collision_state_5))

module_tier_set = []
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(1)


aw_list.append(aw_set)
state_list.append(state_set)
importance_list.append(importance_set)
module_tier_list.append(module_tier_set)

################################
# IF MMAS thought experiment 3 #
################################
cohesion_aw = np.array([-17.3450597,	-16.93497032,	-17.86593676,	-17.25448444,	-17.99413244])
collision_aw = np.array([-0.2171946,	-0.11552542,	-4.83118414,	0,	-0.07333829])
obstacle_aw = np.array([9.53846154,	9.43589744,	9.48589824,	9.48589824,	9.48717949])
boundary_aw_1 = np.array([-1.10864615,	-4.86E-04,	-1.00250899,	-1.0825339,	-1.07723135])
boundary_aw_2 = np.array([0.0,	0.0,	0.0,	0.0,	0.0])
boundary_aw_3 = np.array([0.0,	0.0,	0.0,	0.0,	0.0])
boundary_aw_4 = np.array([0.0,	0.0,	0.0,	0.0,	0.0])
target_aw = np.array([9.53846154,	9.43589744,	9.48589824,	9.48589824,	9.48717949])

cohesion_state = np.array([15,0])
collision_state = np.array([-3,0])
obstacle_state = np.array([3,1,2])
boundary_state_1 = np.array([3])
boundary_state_2 = np.array([-40])
boundary_state_3 = np.array([40])
boundary_state_4 = np.array([-40])
target_state = np.array([20,0])


aw_set = []
aw_set.append(cohesion_aw)
aw_set.append(collision_aw)
aw_set.append(obstacle_aw)
aw_set.append(boundary_aw_1)
aw_set.append(boundary_aw_2)
aw_set.append(boundary_aw_3)
aw_set.append(boundary_aw_4)
aw_set.append(target_aw)

state_set = []
state_set.append(cohesion_state)
state_set.append(collision_state)
state_set.append(obstacle_state)
state_set.append(boundary_state_1)
state_set.append(boundary_state_2)
state_set.append(boundary_state_3)
state_set.append(boundary_state_4)
state_set.append(target_state)

#TODO apply importance functions
importance_set = []
importance_set.append(cohesion_if(cohesion_state))
importance_set.append(collision_if(collision_state))
importance_set.append(obstacle_if(obstacle_state))
importance_set.append(boundary_if(boundary_state_1))
importance_set.append(boundary_if(boundary_state_2))
importance_set.append(boundary_if(boundary_state_3))
importance_set.append(boundary_if(boundary_state_4))
importance_set.append(target_if(target_state))

module_tier_set = []
module_tier_set.append(3)
module_tier_set.append(1)
module_tier_set.append(0)
module_tier_set.append(0)
module_tier_set.append(0)
module_tier_set.append(0)
module_tier_set.append(0)
module_tier_set.append(2)


aw_list.append(aw_set)
state_list.append(state_set)
importance_list.append(importance_set)
module_tier_list.append(module_tier_set)


################################
# IF MMAS thought experiment 4 #
################################

cohesion_aw = np.array([-14.33545907,	-13.56076877,	-12.83711015,	-13.88925426,	-13.62531067])
collision_aw_1 = np.array([-7.63567384,	-0.37932693,	-3.26806525,	3.64305981,	-4.99953036])
collision_aw_2 = np.array([0,	-3.26638796,	-1.39205812,	0,	-0.7816067])
collision_aw_3 = np.array([-0.05961544,	-7.60779431,	-3.62399129,	-4.37190685,	-4.58180773])
collision_aw_4 = np.array([0,	-1.8796722,	0,	-1.96014686,	-0.93337602])
collision_aw_5 = np.array([-4.81687254,	-3.57644489,	-0.1710811,	-7.34864816,	-4.92842504])
obstacle_aw_1 = np.array([-2.772,	-0.94281833,	-1.90473675,	-1.54213889,	-1.818425])
obstacle_aw_2 = np.array([-2.64408375,	-0.98918832,	-1.7814125,	-1.7034675,	-1.8572015])
target_aw = np.array([9.49103288,	9.39280361,	9.45561394,	9.42617333,	9.44201985])


cohesion_state = np.array([-16,0])
collision_state_1 = np.array([3,0])
collision_state_2 = np.array([-3,3])
collision_state_3 = np.array([-3,0])
collision_state_4 = np.array([-3,-3])
collision_state_5 = np.array([0,-3])
obstacle_state_1 = np.array([2,-2,3])
obstacle_state_2 = np.array([2,2,2])
target_state = np.array([20,6])


aw_set = []
aw_set.append(cohesion_aw)
aw_set.append(collision_aw_1)
aw_set.append(collision_aw_2)
aw_set.append(collision_aw_3)
aw_set.append(collision_aw_4)
aw_set.append(collision_aw_5)
aw_set.append(obstacle_aw_1)
aw_set.append(obstacle_aw_2)
aw_set.append(target_aw)

state_set = []
state_set.append(cohesion_state)
state_set.append(collision_state_1)
state_set.append(collision_state_2)
state_set.append(collision_state_3)
state_set.append(collision_state_4)
state_set.append(collision_state_5)
state_set.append(obstacle_state_1)
state_set.append(obstacle_state_2)
state_set.append(target_state)

importance_set = []
importance_set.append(cohesion_if(cohesion_state))
importance_set.append(collision_if(collision_state_1))
importance_set.append(collision_if(collision_state_2))
importance_set.append(collision_if(collision_state_3))
importance_set.append(collision_if(collision_state_4))
importance_set.append(collision_if(collision_state_5))
importance_set.append(obstacle_if(obstacle_state_1))
importance_set.append(obstacle_if(obstacle_state_2))
importance_set.append(target_if(target_state))

module_tier_set = []
module_tier_set.append(3)
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(1)
module_tier_set.append(0)
module_tier_set.append(0)
module_tier_set.append(2)


aw_list.append(aw_set)
state_list.append(state_set)
importance_list.append(importance_set)
module_tier_list.append(module_tier_set)


################################
# IF MMAS thought experiment 5 #
################################

test_aw_1 = np.array([0.0,	0.0,	0.0,	0.0,	0.0])
test_aw_2 = np.array([0.0,	0.0,	4.0,	4.0,	0.0])

test_state_1 = np.array([1,1])
test_state_2 = np.array([1,1])

aw_set = []
aw_set.append(test_aw_1)
aw_set.append(test_aw_2)

state_set = []
state_set.append(test_state_1)
state_set.append(test_state_2)

importance_set = []
importance_set.append(1)
importance_set.append(1)

module_tier_set = []
module_tier_set.append(1)
module_tier_set.append(2)


aw_list.append(aw_set)
state_list.append(state_set)
importance_list.append(importance_set)
module_tier_list.append(module_tier_set)






# #################################
# #       IF MMAS Algorithm       #
# #################################
# for s in range(0, len(aw_list)):
#     print('')
#     print('')
#     print('RESULTS FOR EXPERIMENT ', (s+1))
#     print('')
#     aw_set = aw_list[s]
#     state_set = state_list[s]
#     importance_set = importance_list[s]

#     diff_normed_weights = []
#     for i in range(0,len(aw_set)):
#         diff_normed_weights.append(diff_norm(aw_set[i]))

#     print('diff_normed_weights')
#     for i in range(0,len(diff_normed_weights)):
#         print(diff_normed_weights[i])

#     print('')
#     combined_aw = np.zeros(5)
#     for i in range(0,len(aw_set[0])):
#         for j in range(0,len(aw_set)):
#             combined_aw[i] = combined_aw[i] + aw_set[j][i]*importance_set[j]  

#     print('combined_aw', combined_aw)
#     print('')
#     combined_diff_norm = diff_norm(combined_aw)

#     print('combined_diff_norm', combined_diff_norm)

#     probabilities = soft_max(combined_diff_norm)

#     print('action probabilities', probabilities)


#################################
#       IF MMAS Algorithm       #
#################################

# in this test, we calculate potiential acitons based on tiers.
# we first calculate most/least desired actions based on categories and combinations of categories
# and finally make some sort of compromise to finally select an action 

for s in range(0, len(aw_list)):
    print('')
    print('')
    print('RESULTS FOR EXPERIMENT ', (s+1))
    print('')
    aw_set = aw_list[s]
    state_set = state_list[s]
    importance_set = importance_list[s]
    module_tier_set = module_tier_list[s]

    restricted_action_list = []
    combined_aw_list = []

    # normalize the action weights in a way that mostly decouples the reward scheme from the MMAS
    diff_normed_weights = []
    for i in range(0,len(aw_set)):
        diff_normed_weights.append(diff_norm(aw_set[i]))

    print('diff_normed_weights')
    for i in range(0,len(diff_normed_weights)):
        print(diff_normed_weights[i])

    # next find out the number of tiers
    num_tiers = np.max(module_tier_set) + 1
    print('num tiers', num_tiers) 


    #start by combining the higest tier, then the highest tier with the next highest, and so on.
    #dont forget to store the results from each tier combination
    for t in range(0,num_tiers):
        restricted_action_list.append(np.array([]))
        combined_aw_list.append(np.array([]))

        if t in module_tier_set:
            combined_aw = np.zeros(5)
            
            for w in range(0,len(aw_set)):
                if module_tier_set[w] <= t:
                    for a in range(0,len(aw_set[w])):
                        combined_aw[a] = combined_aw[a] + aw_set[w][a]*importance_set[w]            


            #create a restricted action list for each tier
            restricted_actions = np.where(combined_aw == combined_aw.min())
            print('restricted_actions', restricted_actions[0])
            #if all weights are the same, there is not preference so none should be restricted
            if len(restricted_actions[0]) != len(Action):
                restricted_action_list[t] = restricted_actions

            combined_aw_list[t] = combined_aw
            print('combined_aw', combined_aw)
            print('')
            # combined_diff_norm = diff_norm(combined_aw)

            # print('combined_diff_norm', combined_diff_norm)

            # probabilities = soft_max(combined_diff_norm)

            # print('action probabilities', probabilities)

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
            print('TOP TIER REACHED')
            #handle duplicate entries
            indices = np.argwhere(combined_aw_list[t] == np.amax(combined_aw_list[t]))
            if(len(indices) == 1):
                # action_prime = Action(np.argmax(combined_aw_list[t]))
                print('action taken is', np.argmax(combined_aw_list[t]))
                action_to_take = np.argmax(combined_aw_list[t])
            else:
                # If multiple entries in the Q table row are tied for highest, randomly select one of them
                index = random.randint(0,len(indices)-1)
                # action_prime = Action(indices[index])
                print('action taken is', indices[index])
                action_to_take = indices[index]
            

            # print('action taken is', np.argmax(combined_aw_list[t]))
            action_taken = True
        else:    
            aw_sort = np.argsort(combined_aw_list[t])
            print('aw_sort',aw_sort)

            #take best action case
            if aw_sort.size == len(Action):
                for w in range(len(aw_sort)-1,-1,-1):
                    print('combined_aw_list[t][aw_sort[w]]',combined_aw_list[t][aw_sort[w]])
                    print('aw_sort[w]',aw_sort[w])
                    #starting with the highest reward 

                    #get the set of indices that match this value.
                    duplicates = np.where(combined_aw_list[t] == combined_aw_list[t][aw_sort[w]])
                    print('duplicates[0]', duplicates[0])
                    print('len(duplicates[0])', len(duplicates[0]))
                    if len(duplicates[0]) > 1:
                        choices = np.random.choice(duplicates[0], len(duplicates[0]), replace=False)
                    else:
                        choices = np.array([aw_sort[w]])
                    print('choices', choices)
                    

                    #check if the choices are restriced in any of the upper tiers
                    choices_restricted = np.zeros(choices.shape,dtype=bool)
                    print('choices_restricted', choices_restricted)
                    # eligible_action = False
                    for c in range(0,len(choices)):
                        print('testing choice ', choices[c])
                        for u in range(t-1,-1,-1):
                            # if combined_aw_list[u].size == len(Action):
                            print('restricted_action_list[u]',restricted_action_list[u])
                            print('len(restricted_action_list[u])',len(restricted_action_list[u]))
                            if len(restricted_action_list[u]) != 0:    
                                print('restricted_action_list[u][0]',restricted_action_list[u][0])
                                
                                # print('restricted_action_list[u]',restricted_action_list[u][0])
                                
                                if choices[c] in restricted_action_list[u][0]:
                                    print("it's restricted")
                                    choices_restricted[c] = True
                                    

                    #now i know which choices are restriced, loop over them
                    for c in range(0,len(choices_restricted)):
                        if not choices_restricted[c]:
                            action_to_take = choices[c]
                            action_taken = True
                            break

                    if action_taken:
                        break

        if action_taken:
            print('action taken is', action_to_take)
            print('Action is:', Action(action_to_take))
            break

            #TODO if not take best aciton, sample and check. how many samples though?