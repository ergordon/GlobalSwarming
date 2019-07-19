import numpy as np
import module as module
import agent as agent
import action as action
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
# training_filename = 'test/CohesionModule_training_data.pkl'
# with open(training_filename, 'rb') as f:
#     [module_name, table, states] = pickle.load(f)

# print(module_name)
# print(table)
# print(states)

# myDict = {}
# myDict.update({'A':1})
# print(myDict['A'])
# myDict['A'] = myDict['A'] + 1
# print(myDict['A'])


# T = 0.05
# table = np.array([0.4, 0.1, 0.1, 0.2, 0])

# for i in range(0, len(table)):
#     table[i] = np.exp(table[i]/T)

# sum = np.sum(table)

# for i in range(0, len(table)):
#     table[i] = table[i]/sum

# print(table)


# state = np.array([1, 1])
# state_prime = np.array([1,2])

# print(np.array_equal(state,state_prime))
T = 0.05

aw1 = np.array([-5.14983891, -5.15513717, -5.12367102, -5.18197367, -5.81814747])
aw2 = np.array([-3.50455409e-30, 0, -6.78003540e-36, 0, 0, -1.95114352e-36])


awc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
exp1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
exp2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
exp_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
expc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
for i in range(0,5):
    exp1[i] = np.exp(aw1[i]/T)
    exp2[i] = np.exp(aw2[i]/T)
    exp_sum[i] = exp1[i] + exp2[i]

    awc[i] = aw1[i] + aw2[i]
    expc[i] = np.exp(awc[i]/T)

print('without weights')
print('exp1: ')
print(exp1)
print('exp2: ')
print(exp2)

print('exp_sum: ')
print(exp_sum)

print('expc: ')
print(expc)


weight1 = 0.2
weight2 = 0.8
awc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
exp1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
exp2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
exp_sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
expc = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
for i in range(0,5):
    exp1[i] = np.exp(aw1[i]/T)
    exp2[i] = np.exp(aw2[i]/T)
    exp_sum[i] = exp1[i]*weight1 + exp2[i]*weight2

    awc[i] = aw1[i]*weight1 + aw2[i]*weight2
    expc[i] = np.exp(awc[i]/T)

print('with weights')
print('exp1: ')
print(exp1)
print('exp2: ')
print(exp2)

print('exp_sum: ')
print(exp_sum)

print('expc: ')
print(expc)