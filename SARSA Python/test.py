import numpy as np
import module as module
import agent as agent
# import action as action
from action import Action
import matplotlib.pyplot as plt
import pickle
from simulation import Simulation
from qlearning import Qlearning
import time

# vec1 = np.array([[1,3,1],
#                  [1,1,1]])
# excerpt = vec1[0]
# print(excerpt)
# vec2 = np.array([1,3,1])
# if np.array_equal(excerpt, vec2):
#     print('match!')
# else:
#     print('...so sad')


# #testMod = module.Module()

# agt1 = agent.Agent([1,1])
# agt2 = agent.Agent([3,3])

# #testMod.startTracking(agt1)
# #print(testMod.tracked_agents[0].position)

# agt1.modules[0].start_tracking(agt2)

# # testMod.startTracking(agt2)
# # print(testMod.tracked_agents[0].position)
# # print(testMod.tracked_agents[1].position)

# print(agt1.modules[0].tracked_agents[0].position)

# agt1.modules[0].update_state()
# print(agt1.modules[0].state)



# test_matrix = np.array([[1,2],
#                 [3,4],
#                [5,6]])

# print(len(test_matrix))

# print(test_matrix.size)
# print(test_matrix.shape[1])

# print(test_matrix[2])

# matrix = np.array([])
# empty_row = np.zeros(len(action.Action))
# matrix = empty_row
# print(empty_row)
# matrix = np.vstack([matrix,empty_row])
# print(matrix)
# matrix = np.vstack([matrix,empty_row])
# print(matrix)
# matrix = np.vstack([matrix,empty_row])
# print(matrix)
# matrix = np.vstack([matrix,empty_row])
# print(matrix)

# act = action.Action.STAY
# print(act.value)

# test_matrix = np.zeros(len(action.Action))

# print(test_matrix)

# print('value is')
# print(76.21392)
# print('exp is')
# print(np.exp(7600.21392))

# q_states = np.array([[1,1],
#                     [1,2],
#                     [1,3],
#                     [1,4]])

# q_table = np.array([[1,1,1,1,1],
#                     [2,2,2,2,2],
#                     [3,3,3,3,3],
#                     [4,4,4,4,4]])

# state = np.array([1,2])

# #TODO research a better (more efficeint) way of doing this
# index = -1

# if q_states.shape[0] != 0: #check for empty matrix

#     matches = np.equal(q_states,[state]).all(1).nonzero()

#     if matches[0].size == 0:
#         #state not in q states add it along with the row
#         empty_row = np.zeros(len(action.Action))
#         q_states = np.vstack([q_states, np.copy(state)])
#         q_table = np.vstack([q_table, empty_row])
        
#         index = q_states.shape[0]-1
#     else:
#         #working state already in q states for this module, 
#         #sum the working q row with the corresponding entry in the q table for this module
#         #incerement the number of times this row has been updated
#         index = matches[0][0] 
        
# else: 
#     #tables are empty, put ours in
#     q_table = np.zeros((1,len(action.Action)))
#     q_states = np.copy(state.reshape((1,state.shape[0])))
#     index = 0

# print('index is:')
# print(index)
# print('table is:')
# print(q_table)
# print('states is:')
# print(q_states)

# print('row is')
# print(q_table[index])

# ss = np.array([])

#put the search space into a single array to make for easier comparison with the agent's state
#resulting format is [x1, x2, y1, y2, ...] where 1 is the lower bound and 2 is the upper bound
# ss = np.array([])
# for i in range(0,len(sim.Simulation.search_space)):
#     for j in range(0,len(sim.Simulation.search_space[i])):
#         ss = np.hstack((ss, sim.Simulation.search_space[i][j]))

# print('ss')
# print(ss)


# agt1 = agent.Agent([1,1])
# agt2 = agent.Agent([2,2])
# agt1.modules[0].start_tracking(agt2)
# agt1.modules[0].update_state()

# rewards = [10, 5, -1]
# #the discrete ranges at which the agent can collect rewards
# ranges_squared = [25,225,625]

# dist_squared = 24

# instant_reward = 0
# rewarded = False
# for i in range(0,len(ranges_squared)):
#     if dist_squared <= ranges_squared[i]:
#         instant_reward = rewards[i]
#         rewarded = True    
#         break

# print('the reward is')
# print(instant_reward)

# test = np.empty((2,), dtype=object)
# test = np.array([])


# test[0] = Qlearning()
# test[1] = Qlearning()

# test[0].fetch_row_by_state(np.array([2,2], dtype='f'))


# vector1 = np.array([1,1]) #D1
# vector2 = np.array([2,2]) #D1
# vector3 = np.array([3,3]) #D2
# vector4 = np.array([4,4]) #D2

# table = np.empty((2,),dtype=object)

# temp_table_1 = np.array([vector1, vector2])
# temp_table_2 = np.array([vector3, vector4])

# print(temp_table_1)
# print(temp_table_2)
# table[0] = temp_table_1
# table[1] = temp_table_2

# print(table)

# vector5 = np.array([5,5])
# vector6 = np.array([6,6])

# table[0] = np.vstack([table[0],vector5])
# table[0] = np.vstack([table[0],vector6])

# print(table)

# keys = np.array([])
# values = np.array([])

# BigDict = {}
# for i in range(0, 100000):
#     key = np.array([i,i])
#     value = np.array([i,i])
#     BigDict.update({hash(tuple(key)): value })

#     if i == 0:
#         keys = np.array([i,i])
#         values = np.array([i,i])
#     else:
#         keys = np.vstack([keys, np.array([i,i])])
#         values = np.vstack([values, np.array([i,i])])


# # print(keys)

# start = time.time()


# for i in range(0, 100000):
#     key = np.array([i,i])
#     tmp = BigDict[hash(tuple(key))]


# end = time.time()

# duration = end - start

# print('operation(s) took')
# print(duration)


# #########################################################

# start = time.time()


# for i in range(0, 100000):
#     key = np.array([i,i])
    
#     #check if the working state already exists in the q table for this module 
#     matches = np.equal(keys,[key]).all(1).nonzero()

#     matching_index = matches[0][0] 
#     tmp = values[matching_index]


# end = time.time()

# duration = end - start

# print('operation(s) took')
# print(duration)


# keys = np.array([])
# values = np.array([])

# BigDict = {}
# # BigDict.update({1:2})
# for i in range(0, 10):
#     key = np.array([i,i])
#     value = np.array([i,i])
#     BigDict.update({tuple(key): value })


# for key in BigDict:
#     if keys.shape[0] == 0:
#         keys = np.asarray(key)
#         values = BigDict[key]
#     else:
#         keys = np.vstack([keys, np.asarray(key)])
#         values = np.vstack([values, BigDict[key]])


# print(keys)
# print(values)

# boundary module example???
# agents = list()
# position = np.array([0,0])

# for a in range(0,2):
#     agents.append(agent.Agent(position))
#     for q in range(0,len(agents[a].modules[0].Q)):

#         for j in range(1,10):
#             state = np.array([j/10, -(j/10)])
#             index = agents[a].modules[0].Q[q].fetch_row_index_by_state(state)
#             agents[a].modules[0].Q[q].q_table[index] = np.array([a+q,a+q,a+q,a+q,a+q])
            

# print(agents[0].modules[0].Q[0].q_table)
# print(agents[0].modules[0].Q[0].q_states)
# print(agents[0].modules[0].Q[1].q_table)
# print(agents[0].modules[0].Q[1].q_states)
# print(agents[1].modules[0].Q[0].q_table)
# print(agents[1].modules[0].Q[0].q_states)
# print(agents[1].modules[0].Q[1].q_table)
# print(agents[1].modules[0].Q[1].q_states)

# agent_filename = 'test/agents.pkl'
# with open(agent_filename,'wb') as f:
#     pickle.dump(agents,f)



#cohesion module example
# agents = list()
# position = np.array([0,0])

# for a in range(0,4):
#     agents.append(agent.Agent(position))
    
#     for j in range(1,10):
#         state = np.array([j/10, -(j/10)])
#         index = agents[a].modules[0].Q[0].fetch_row_index_by_state(state)
#         agents[a].modules[0].Q[0].q_table[index] = np.array([a,a,a,a,a])
#             #Q[0]?

#     print(agents[a].modules[0].Q[0].q_table)
#     print(agents[a].modules[0].Q[0].q_states)
    
# agent_filename = 'test/agents.pkl'
# with open(agent_filename,'wb') as f:
#     pickle.dump(agents,f)





# load the saved data and see whats inside...
# training_filename = 'obstacle_tiered_retrain/ObstacleAvoidanceModule_training_data.pkl'
# with open(training_filename, 'rb') as f:
#     [module_name, q_data, q_updates] = pickle.load(f)

# test_state = np.array([4,4,3])
# print(q_data[0][tuple(test_state)])


# print(module_name)
# print(q_data)
# print(q_updates)

# myDict = {}
# myDict.update({'A':1})
# print(myDict['A'])
# myDict['A'] = myDict['A'] + 1
# print(myDict['A'])



# x=[1,2,3]
# y=[9,8,7]

# plt.plot(x,y)
# for a,b in zip(x, y): 
#     plt.text(a, b, str(b))
# plt.show()

# T = 0.1

# aw_min = np.array([1,0,0,0,0])
# aw_max = np.array([1,1,1,1,0])

# min_sum = 0
# max_sum = 0

# for i in range(0,len(aw_min)):
#     min_sum = min_sum + np.exp(aw_min[i]/T)
#     max_sum = max_sum + np.exp(aw_max[i]/T)

# result_min = np.zeros((5,))
# result_max = np.zeros((5,))

# for i in range(0,len(aw_min)):
#     result_min[i] = np.exp(aw_min[i]/T)/min_sum
#     result_max[i] = np.exp(aw_max[i]/T)/max_sum

# print('result_min', result_min)
# print('result_max', result_max)





# alpha = 0.7
# gamma = 0.99
# T = 1
# r = 7
# delta = 1/5
# sigma = 1


# Q_old = [1.22, 14.22, 3.11, 0, 3]



# def normalize(vector, range): # range should be (lower_bound, upper_bound) 
#         a = np.max(vector)
#         c = np.min(vector)
#         b = range[1]
#         d = range[0]

#         m = (b - d) / (a - c)
#         pslope = (m * (vector - c)) + d
#         return pslope

# action_weights = np.array([ -900.36435528,  828.87761814,  926.55647586,  811.38502291, 1000])

# if not min(action_weights) ==  max(action_weights):
#     action_weights = normalize(action_weights, np.array([-1,1]))

# action_weights = np.array([1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
# action_weights = np.array([1,1,1,1,1,1,1,1,0])



#############################
# throttling T via epsilon. the logarithmic way.

# num_div = 100
# max_probs = np.zeros(num_div)
# Tees = np.zeros(num_div)

# T_i = 1.445
# T_f = 0.285

# for n in range(0,num_div):
#     action_weights = np.array([1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    
#     epsilon = float(n)/float(num_div) 
#     # print(epsilon)
#     # T = T_f - (1.0-epsilon)*(T_f - T_i)
#     # T = T_f - (1.0-epsilon)*(T_f - T_i)
#     T = T_f**epsilon * T_i**(1-epsilon)
#     # max_probs[i] = 1
#     Tees[n] = T

#     for i in range(0,len(action_weights)):
#         action_weights[i] = np.exp(action_weights[i]/T)

#     sum_action_weights = np.sum(action_weights)        
#     if sum_action_weights == 0:
#         action_weights = np.ones(len(Action))/len(Action)
#     elif sum_action_weights != 1:
#         # print('not sum 1')
#         action_weights = action_weights/sum_action_weights
#     print(action_weights)
#     max_probs[n] = action_weights[0]

# plt.plot(Tees)
# plt.plot(max_probs)
# plt.show()
# print('action_weights', action_weights)

########################

# T = 1.445 #.35
# T = .58
# T = 100000000000000000
# T = 18000000000000000
# aw_sum = 0
# for i in range(0,len(action_weights)):
#     action_weights[i] = np.exp(action_weights[i]/T)
#     aw_sum = aw_sum + action_weights[i] 

# T = 1
# for i in range(0,len(action_weights)):
#     action_weights[i] = np.exp(action_weights[i]/T)

# sum_action_weights = np.sum(action_weights)        
# if sum_action_weights == 0:
#     action_weights = np.ones(len(Action))/len(Action)
# elif sum_action_weights != 1:
#     action_weights = action_weights/sum_action_weights



# print('action_weights/aw_sum', action_weights)

# action_weights = np.array([1,1,1,1,0])#,-1,-1,-1,1])

# print('action_weights', action_weights)

# # T = 1
# # T = 100000000000000000
# # T = 18000000000000000
# aw_sum = 0
# for i in range(0,len(action_weights)):
#     action_weights[i] = np.exp(action_weights[i]/T)
#     aw_sum = aw_sum + action_weights[i] 

# print('action_weights/aw_sum',action_weights/aw_sum)




# action_weights = np.array([ -900.36435528,  828.87761814,  926.55647586,  811.38502291, 1000])
# asum = np.sum(np.absolute(action_weights)) #TODO fix infinity again
# #TODO divide by the sum of the absolute values? 
# if(asum != 0):
#     action_weights = action_weights / asum

# print(action_weights)

# aw1 = np.array([-3.20645073, -0.32784625, -0.0580238,  -6.91932535, -0.91353664])
# aw2 = np.array([585.87790536, 375.68208995, 291.0006224,  373.60879064, 577.79851454])
# w1 = 20
# w2 = 1

# awsum = aw1+w1 + aw2+w2
# print(awsum)

# aw1 = np.array([-0.18400376,  -2.15344679, -40.3941463,   -0.15047648,  -0.84795454])
# aw1 = np.array([529.58503617, 338.26526735, 326.68028439, 175.29753524, 560.492753])

# awsum = aw1+w1 + aw2+w2
# print(awsum)


# a1 = np.array([1,1])
# a2 = np.array([1,2])
# print(np.array_equal(a1, a2))

# def myround(x, base=5):
#     return int(base * round(float(x)/base))

# def my_round(x, prec=2, base=0.05): 
#     return (base * (np.array(x) / base).round()).round(prec)

# number = 143.356

# print(my_round(number, 2, 1))


# a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])    
# ind = np.argpartition(a, -4)[-4:]
# print(a[ind])

# # for i in np.arange(0,len(x),2):
# #     connectpoints(x,y,i,i+1)
# x=[-1 ,0.5 ,1,-0.5]
# y=[ 0.5,  1, -0.5, -1]

# for i in np.arange(0,len(x),2):
#     plt.plot(x[i:i+2],y[i:i+2],'k-')

# plt.axis('equal')
# plt.show()


# MOVE_PLUS_X = 0
# MOVE_PLUS_X_PLUS_Y = 1
# MOVE_PLUS_Y = 2
# MOVE_MINUS_X_PLUS_Y = 3
# MOVE_MINUS_X = 4
# MOVE_MINUS_X_MINUS_Y = 5
# MOVE_MINUS_Y = 6
# MOVE_PLUS_X_MINUS_Y = 7
# STAY = 8

action_headings = {}
nearest_actions = {}
opposite_action = {}


# for i in range(0,len(Action)):
for action_data in Action:
    if not action_data == Action.STAY:
        heading = float(action_data.value)/float(len(Action)-1)*2*np.pi
        print(heading)
        action_headings.update({action_data : heading})


#TODO put this into a for loop
    
# #+X
# nearest = np.array([action_headings[Action.MOVE_PLUS_X_PLUS_Y], action_headings[Action.MOVE_PLUS_X_MINUS_Y]])
# opposite = action_headings[Action.MOVE_MINUS_X]
# nearest_action_headings.update({Action.MOVE_PLUS_X:nearest})
# opposite_action_headings.update({Action.MOVE_PLUS_X:opposite})
# #+X+Y
# nearest = np.array([action_headings[Action.MOVE_PLUS_Y], action_headings[Action.MOVE_PLUS_X]])
# opposite = action_headings[Action.MOVE_MINUS_X_MINUS_Y]
# nearest_action_headings.update({Action.MOVE_PLUS_X_PLUS_Y:nearest})
# opposite_action_headings.update({Action.MOVE_PLUS_X_PLUS_Y:opposite})
# #+Y
# nearest = np.array([action_headings[Action.MOVE_MINUS_X_PLUS_Y], action_headings[Action.MOVE_PLUS_X_PLUS_Y]])
# opposite = action_headings[Action.MOVE_MINUS_Y]
# nearest_action_headings.update({Action.MOVE_PLUS_Y:nearest})
# opposite_action_headings.update({Action.MOVE_PLUS_Y:opposite})
# #-X+Y
# nearest = np.array([action_headings[Action.MOVE_MINUS_X], action_headings[Action.MOVE_PLUS_Y]])
# opposite = action_headings[Action.MOVE_PLUS_X_MINUS_Y]
# nearest_action_headings.update({Action.MOVE_MINUS_X_PLUS_Y:nearest})
# opposite_action_headings.update({Action.MOVE_MINUS_X_PLUS_Y:opposite})
# #-X
# nearest = np.array([action_headings[Action.MOVE_MINUS_X_MINUS_Y], action_headings[Action.MOVE_MINUS_X_PLUS_Y]])
# opposite = action_headings[Action.MOVE_PLUS_X]
# nearest_action_headings.update({Action.MOVE_MINUS_X:nearest})
# opposite_action_headings.update({Action.MOVE_MINUS_X:opposite})
# #-X-Y
# nearest = np.array([action_headings[Action.MOVE_MINUS_Y], action_headings[Action.MOVE_MINUS_X]])
# opposite = action_headings[Action.MOVE_PLUS_X_PLUS_Y]
# nearest_action_headings.update({Action.MOVE_MINUS_X_MINUS_Y:nearest})
# opposite_action_headings.update({Action.MOVE_MINUS_X_MINUS_Y:opposite})
# #-Y
# nearest = np.array([action_headings[Action.MOVE_PLUS_X_MINUS_Y], action_headings[Action.MOVE_MINUS_X_MINUS_Y]])
# opposite = action_headings[Action.MOVE_PLUS_Y]
# nearest_action_headings.update({Action.MOVE_MINUS_Y:nearest})
# opposite_action_headings.update({Action.MOVE_MINUS_Y:opposite})
# #+X-Y
# nearest = np.array([action_headings[Action.MOVE_PLUS_X], action_headings[Action.MOVE_MINUS_Y]])
# opposite = action_headings[Action.MOVE_MINUS_X_PLUS_Y]
# nearest_action_headings.update({Action.MOVE_PLUS_X_MINUS_Y:nearest})
# opposite_action_headings.update({Action.MOVE_PLUS_X_MINUS_Y:opposite})


#+X
nearest = np.array([Action.MOVE_PLUS_X_PLUS_Y, Action.MOVE_PLUS_X_MINUS_Y])
opposite = Action.MOVE_MINUS_X
nearest_actions.update({Action.MOVE_PLUS_X:nearest})
opposite_action.update({Action.MOVE_PLUS_X:opposite})
#+X+Y
nearest = np.array([Action.MOVE_PLUS_Y, Action.MOVE_PLUS_X])
opposite = Action.MOVE_MINUS_X_MINUS_Y
nearest_actions.update({Action.MOVE_PLUS_X_PLUS_Y:nearest})
opposite_action.update({Action.MOVE_PLUS_X_PLUS_Y:opposite})
#+Y
nearest = np.array([Action.MOVE_MINUS_X_PLUS_Y, Action.MOVE_PLUS_X_PLUS_Y])
opposite = Action.MOVE_MINUS_Y
nearest_actions.update({Action.MOVE_PLUS_Y:nearest})
opposite_action.update({Action.MOVE_PLUS_Y:opposite})
#-X+Y
nearest = np.array([Action.MOVE_MINUS_X, Action.MOVE_PLUS_Y])
opposite = Action.MOVE_PLUS_X_MINUS_Y
nearest_actions.update({Action.MOVE_MINUS_X_PLUS_Y:nearest})
opposite_action.update({Action.MOVE_MINUS_X_PLUS_Y:opposite})
#-X
nearest = np.array([Action.MOVE_MINUS_X_MINUS_Y, Action.MOVE_MINUS_X_PLUS_Y])
opposite = Action.MOVE_PLUS_X
nearest_actions.update({Action.MOVE_MINUS_X:nearest})
opposite_action.update({Action.MOVE_MINUS_X:opposite})
#-X-Y
nearest = np.array([Action.MOVE_MINUS_Y, Action.MOVE_MINUS_X])
opposite = Action.MOVE_PLUS_X_PLUS_Y
nearest_actions.update({Action.MOVE_MINUS_X_MINUS_Y:nearest})
opposite_action.update({Action.MOVE_MINUS_X_MINUS_Y:opposite})
#-Y
nearest = np.array([Action.MOVE_PLUS_X_MINUS_Y, Action.MOVE_MINUS_X_MINUS_Y])
opposite = Action.MOVE_PLUS_Y
nearest_actions.update({Action.MOVE_MINUS_Y:nearest})
opposite_action.update({Action.MOVE_MINUS_Y:opposite})
#+X-Y
nearest = np.array([Action.MOVE_PLUS_X, Action.MOVE_MINUS_Y])
opposite = Action.MOVE_MINUS_X_PLUS_Y
nearest_actions.update({Action.MOVE_PLUS_X_MINUS_Y:nearest})
opposite_action.update({Action.MOVE_PLUS_X_MINUS_Y:opposite})


print(nearest_actions)
print(opposite_action)

action_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.4, 0.6, 0.8, 0.7])


#handle direciton first. (other code exists for finding highest action index)
action_best = Action.MOVE_PLUS_X
action_CW = nearest_actions[action_best][0]
action_CCW = nearest_actions[action_best][1]
action_opposite = opposite_action[action_best]

angle_best =  action_headings[action_best]
angle_CW =   action_headings[action_best]
if abs(angle_CW - angle_best) > np.pi:
    angle_CW = 2*np.pi - angle_CW
angle_CCW =  action_headings[action_best]
if abs(angle_CCW - angle_best) > np.pi:
    angle_CCW = 2*np.pi - angle_CCW
angle_opposite = action_headings[action_best]

weight_best = action_weights[action_best.value]
weight_CW = action_weights[action_CW.value]
weight_CCW = action_weights[action_CCW.value]
weight_opposite = action_weights[action_opposite.value]
weight_stay = action_weights[-1]

#now calculate direction
w_CCW = 1.0/(2+(weight_best - weight_CCW)**2)
w_CW = 1.0/(2+(weight_best - weight_CW)**2)

angle_actual = angle_best + w_CCW*(angle_CCW - angle_best) + w_CW*(angle_CW - angle_best)

print('angle_actual',angle_actual)

#now handle step size
step_scalar =(weight_opposite*-1.0 + weight_best*1.0)/(weight_opposite + weight_best + weight_stay) 
print('step_scalar',step_scalar)

