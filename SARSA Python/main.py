from agent import Agent
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


##############################################################################
#   Simulation Variables
##############################################################################


num_agents = 10 #number of agents to simulate

num_episodes = 1 #number of times to run the training scenario
episode_length = 10 #number of timesteps in each traning scenario

#bounds to initialize the agents inside of
init_space = [[0,10],
             [0,10]]


##############################################################################
#   Initialization
##############################################################################


agents = list() #list of agents

#initialize agent positions
for i in range(0,num_agents):
    position = np.array([i,i], dtype='f')
    agents.append(Agent(position))
    print(agents[i].position)

#initialize module paramters such as who each agent is tracking
#TODO make it so the tracked agents are based on range and upated every iteration
for i in range(0,num_agents):
    for j in range(0,num_agents):
        if(i != j):
           #loop through each module
           for m in range(0,len(agents[i].modules)):
                agents[i].modules[m].start_tracking(agents[j])
            #TODO initialize the module state as well???

#initialize module state parameters
for i in range(0,num_agents):
    #loop through each module
    for m in range(0,len(agents[i].modules)):
        agents[i].modules[m].update_state()
        agents[i].modules[m].state_prime = agents[i].modules[m].state

##############################################################################
#   main algorithm
##############################################################################
frame_rate = 1
plt.plot([1,2,3,4])
plt.draw()
plt.pause(1/frame_rate)
plt.clf()
plt.cla()

for e in range(0,num_episodes):
    for t in range(0,episode_length):
        for agnt in agents:
            #take the action determined in the last step
            #update agent positions on plots
            agnt.change_position(agnt.position + np.array([1,.5]))
            plt.plot(agnt.position[0],agnt.position[1],'ro')
            
        #softmax to find action_prime
        #TODO implement
        print('Softmax!')
         
        for agnt in agents:
            for mod in agnt.modules:
                mod.select_next_action()

                print('Updating state prime!')
                #update state_prime
                mod.update_state_prime()
                
                print('Updating instant reward!')                   
                #update instantaneous rewards
                mod.update_instant_reward()
                
                print('Updating total reward!')
                #update the total reward 
                mod.update_total_reward()
                # print(mod.parent_agent.total_reward)

                
                print('Updating Q!')
                print(mod.action)
                print(mod.action_prime)
                #update Q
                mod.update_q()

                
                mod.action = mod.action_prime
                mod.state  = mod.state_prime

        #dont forget to set action = action_prime
        #dont forget to set state = state_prime
        
        plt.draw()
        plt.pause(1/frame_rate)
        plt.clf()
        plt.cla()
    



