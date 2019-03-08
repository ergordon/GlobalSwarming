import numpy as np
import module as module
import agent as agent

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

agt1.modules[0].startTracking(agt2)

# testMod.startTracking(agt2)
# print(testMod.tracked_agents[0].position)
# print(testMod.tracked_agents[1].position)

print(agt1.modules[0].tracked_agents[0].position)

agt1.modules[0].updateState(agt1)
print(agt1.modules[0].state)