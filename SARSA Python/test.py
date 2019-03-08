import numpy as np
import module as module
import agent as agent
import action as action

vec1 = np.array([[1,3,1],
                 [1,1,1]])
excerpt = vec1[0]
print(excerpt)
vec2 = np.array([1,3,1])
if np.array_equal(excerpt, vec2):
    print('match!')
else:
    print('...so sad')


#testMod = module.Module()

agt1 = agent.Agent([1,1])
agt2 = agent.Agent([2,2])

#testMod.startTracking(agt1)
#print(testMod.tracked_agents[0].position)

agt1.modules[0].start_tracking(agt2)

# testMod.startTracking(agt2)
# print(testMod.tracked_agents[0].position)
# print(testMod.tracked_agents[1].position)

print(agt1.modules[0].tracked_agents[0].position)

agt1.modules[0].update_state()
print(agt1.modules[0].state)



test_matrix = np.array([[1,2],
                [3,4],
               [5,6]])

print(len(test_matrix))

print(test_matrix.size)
print(test_matrix.shape[1])

print(test_matrix[2])

matrix = np.array([])
empty_row = np.zeros(len(action.Action))
matrix = empty_row
print(empty_row)
matrix = np.vstack([matrix,empty_row])
print(matrix)
matrix = np.vstack([matrix,empty_row])
print(matrix)
matrix = np.vstack([matrix,empty_row])
print(matrix)
matrix = np.vstack([matrix,empty_row])
print(matrix)