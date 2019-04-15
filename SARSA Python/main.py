from agent import Agent
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import pickle
import os.path


##############################################################################
#   Helper Functions
##############################################################################

#check if a given position is within given bounds
def checkInBounds(position,bounds):
    
    #TODO make sure position and bound have same number of 
    for i in range(0,len(position)):
        if not ( bounds[i][0] <= position[i] <= bounds[i][1]):
            return False
    return True

#TODO actually use bounds
#reset the agents to initial conditions (except for the Q states and tables)
def ReinitializeAgents(agents,bounds):
    #initialize agent parameters
    for i in range(0,len(agents)):
        #TODO make this initial position randomized
        agents[i].position = np.array([2*i,2*i], dtype='f')
        agents[i].total_reward = 0
        
    #initialize module parameters
    for i in range(0,num_agents):
        #loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].action = Action.STAY
            agents[i].modules[m].action_prime = Action.STAY
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = agents[i].modules[m].state

##############################################################################
#   Helper Functions
##############################################################################

##############################################################################
#   Simulation Variables
##############################################################################


num_agents = 5 #number of agents to simulate
num_episodes = 1 #number of times to run the training scenario
episode_length = 200 #number of time steps in each training scenario

#bounds to initialize the agents inside of
init_space = [[0,10],
             [0,10]]

#bounds to simulate the agents within
#exiting these bounds will end the episode immediately
search_space = [[-50,50],
                [-50,50]]

visualize = True   #whether to show a plot animation of the agent positions

agent_rewards = np.array ([])   # matrix containing total reward values for each agent for each episode

##############################################################################
#   Simulation Variables
##############################################################################

##############################################################################
#   Initialization
##############################################################################
print('initializing agents')

agents = list() #list of agents

#check if a file containing a list of agents already exits
if(os.path.isfile('agents.pkl')):
    #if so, load it
    print("Q learning data found, loading it now")
    #TODO handle if the desired number of agents is different from the number of agents saved to disk
    with open('agents.pkl', 'rb') as f:
        agents = pickle.load(f)
else:
    #if not, initialize a set of agents from scratch
    #initialize agent positions
    for i in range(0,num_agents):
        position = np.array([i,i], dtype='f')
        agents.append(Agent(position))
        #print(agents[i].position)

    #initialize module parameters such as who each agent is tracking
    #TODO make it so the tracked agents are based on range and updated every iteration
    for i in range(0,num_agents):
        for j in range(0,num_agents):
            if(i != j):
                #loop through each module
                for m in range(0,len(agents[i].modules)):
                    agents[i].modules[m].start_tracking(agents[j])

    #initialize module state parameters
    for i in range(0,num_agents):
        #loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = agents[i].modules[m].state
##############################################################################
#   Initialization
##############################################################################

##############################################################################
#   main algorithm
##############################################################################

#plotting for visualization
if(visualize):
    frame_rate = 60
    axis_bounds = [search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]]
    plt.axis(axis_bounds)
    plt.draw()
    plt.pause(1/frame_rate)
    plt.clf()
    plt.cla()

print('beginning training')
for e in range(0,num_episodes):
    print("beginning episode #" + str(e+1))

    for t in range(0,episode_length):
        agent_out_of_bounds = False

        for agnt in agents:
            #take the action determined in the last step
            #update agent positions on plots
            #TODO use action across multiple modules
            agnt.take_action(agnt.modules[0].action)

            #check if any agent went out of search space.
            #terminate episode if so
            if not (checkInBounds(agnt.position,search_space)):
                print("agent left search space, ending episode")
                agent_out_of_bounds = True

            if(visualize):
                for agnt in agents:
                    plt.plot(agnt.position[0],agnt.position[1],'ro')
                    plt.axis(axis_bounds)
                    for mod in agnt.modules:
                        mod.visualize()

        #criteria for ending the episode early.
        if(agent_out_of_bounds):
            break


        for agnt in agents:
            for mod in agnt.modules:
                
                #select the next action (action_prime) for the agent to take 
                mod.select_next_action()
                
                #find what the state (state_prime) would be if that action were taken
                mod.update_state_prime()

                #determine the reward for executing the action (not prime) in the state (not prime)
                #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
                mod.update_instant_reward()
                
                #Add the reward for this action to the total reward earned by the agent 
                mod.update_total_reward()
                
                #update the Q table
                mod.update_q()

                #prepare for next time step
                mod.action = mod.action_prime
                mod.state  = mod.state_prime

 
        #plotting for visualization
        if(visualize):
            plt.draw()
            plt.pause(1/frame_rate)
            plt.clf()
            plt.cla()


    
    #store the total reward for each agent at the end of each episode for algorithm performance analysis
    episode_rewards = np.zeros(num_agents) 
    for a in range(0,num_agents):
        episode_rewards[a] = agents[a].total_reward

    if agent_rewards.size == 0:
        agent_rewards = episode_rewards
    else:
        agent_rewards = np.vstack([agent_rewards,episode_rewards])

    #reset the agents (except for the Q tables and Q states) to start fresh for the next episode         
    ReinitializeAgents(agents,init_space)

    #save the trained agents to a file
    agent_filename = 'agents.pkl'
    with open(agent_filename,'wb') as f:
        pickle.dump(agents,f)

##############################################################################
#   main algorithm
##############################################################################

##############################################################################
#   data 
##############################################################################

#store the iterations and total rewards for each agent for each episode
iterations = np.arange(num_episodes)
agent_reward_filename = 'agent_rewards.pkl'
with open(agent_reward_filename,'wb') as f:
    pickle.dump([iterations, agent_rewards],f)  

#close the visualization plot and create a new plot of each agents total reward over time
plt.close()
for i in range(0,num_agents):
    plt.plot(iterations,agent_rewards[:,i])
plt.show()

##############################################################################
#   data
##############################################################################
  
    

