import numpy as np
import module as module
import agent as agent
import action as action
import matplotlib.pyplot as plt
import pickle

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

print("Q learining data found, loading it now")
#TODO handle if the desired number of agents is different from the number of agents saved to disk
with open('agents.pkl', 'rb') as f:
    agents = pickle.load(f)

num_agents = len(agents)

#average and save the Q tables for each agent
for i in range(0,len(agents[0].modules)):
    q_table = np.array([])
    q_states = np.array([])
    number_experienced = np.array([])

    for j in range(0,num_agents):
        for k in range(0, agents[j].modules[i].Q.q_states.shape[0]):
            working_state = agents[j].modules[i].Q.q_states[k]
            if q_states.shape[0] != 0:
                print('checking')
                print(np.equal(q_states,[working_state]).all(1))
                


            if not working_state.tolist() in q_states.tolist():
                #state not yet added to our local list, add it now
                if q_states.shape[0] != 0:
                    q_states = np.vstack([q_states, working_state])
                else:
                    q_states = working_state
            else:
                #state already exists, add to the current q_table value
                pass
            


            # print(any(np.equal(q_states,working_state).all(1)))
            # if not any(np.equal(q_states,working_state).all(1)):
            #     print('not there!')
