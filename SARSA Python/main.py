from agent import Agent
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

#simulation variables
numAgents = 10

numEpisodes = 1 #number of times to run the training scenario
episodeLength = 10 #number of timesteps in each traning scenario

#bounds to initialize the agents inside of
initSpace = [[0,10],
             [0,10]]

agents = list() #list of agents


#initialize agent positions
for i in range(0,numAgents):
    position = np.array([i,i], dtype='f')
    agents.append(Agent(position))
    print(agents[i].position)

#initialize module paramters such as who each agent is tracking
#TODO make it so the tracked agents are based on range and upated every iteration
for i in range(0,numAgents):
    for j in range(0,numAgents):
        if(i != j):
           #loop through each module
           for m in range(0,len(agents[i].modules)):
                agents[i].module[m].startTracking(agents[j])



#main algorithm

frameRate = 1
plt.plot([1,2,3,4])
plt.draw()
plt.pause(1/frameRate)
plt.clf()
plt.cla()

for e in range(0,numEpisodes):
    for t in range(0,episodeLength):
        for agnt in agents:
            #take the action determined in the last step
            #update agent positions on plots
            agnt.changePos(agnt.position + np.array([1,.5]))
            plt.plot(agnt.position[0],agnt.position[1],'ro')
            
        for agnt in agents:
            #update state_prime


            #collect total reward

            #softmax to find action_prime

            #update each module(need a for loop)
            print('do something here!')

            
        
        plt.draw()
        plt.pause(1/frameRate)
        plt.clf()
        plt.cla()
    



