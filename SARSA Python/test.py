import numpy as np
import module as module
import agent as agent
import action as action
import matplotlib.pyplot as plt
import pickle
from simulation import Simulation
from qlearning import Qlearning

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


vector1 = np.array([1,1]) #D1
vector2 = np.array([2,2]) #D1
vector3 = np.array([3,3]) #D2
vector4 = np.array([4,4]) #D2

table = np.empty((2,),dtype=object)

temp_table_1 = np.array([vector1, vector2])
temp_table_2 = np.array([vector3, vector4])

print(temp_table_1)
print(temp_table_2)
table[0] = temp_table_1
table[1] = temp_table_2

print(table)

vector5 = np.array([5,5])
vector6 = np.array([6,6])

table[0] = np.vstack([table[0],vector5])
table[0] = np.vstack([table[0],vector6])

print(table)

print(table[0][1])