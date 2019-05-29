from agent import Agent
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import pickle
import os.path
import copy as cp


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
            agents[i].modules[m].state_prime = np.copy(agents[i].modules[m].state)

##############################################################################
#   Helper Functions
##############################################################################

##############################################################################
#   Simulation Variables
##############################################################################

num_agents = 5 #number of agents to simulate

num_episodes = 1000 #number of times to run the training scenario
episode_length = 100 #number of timesteps in each traning scenario

#bounds to initialize the agents inside of
init_space = [[0,10],
             [0,10]]

#bounds to simulate the agents within
#exiting these bounds will end the episode immediately
search_space = Simulation.search_space

# search_space = [[-50,50],
#                 [-50,50]]

visualize = True    #whether to show a plot animation of the agent positions

agent_rewards = np.array ([])   # matrix containing total reward values for each agent for each episode

#TODO how to handle if both are set to true??? Right now, the training data will overwrite the agent qlearning data
#should i just exit with an error?
load_agents = False #whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
load_training_data = False #whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)

##############################################################################
#   Simulation Variables
##############################################################################

##############################################################################
#   Initialization
##############################################################################
print('initializing agents')

#store the program start time so we can calculate how long it took for the code to execute
start_time = time.time() 

agents = list() #list of agents
initialized = False
#check if a file containing a list of agents already exits
if load_agents:
    if os.path.isfile('agents.pkl'):
        #if so, load it
        print("Agent data found, loading it now")
        #TODO handle if the desired number of agents is different from the number of agents saved to disk
        with open('agents.pkl', 'rb') as f:
            agents = pickle.load(f)
        initialized = True

if not initialized:
    #if not, initialize a set of agents from scratch
    #initialize agent positions
    for i in range(0,num_agents):
        position = np.array([2*i,2*i], dtype='f')
        agents.append(Agent(position))
        print(agents[i].position)

    #initialize module parameters such as who each agent is tracking
    #TODO make it so the tracked agents are based on range and updated every iteration
    for i in range(0,num_agents):
        for j in range(0,num_agents):
            if(i != j):
                #TODO chagne this, not every module will care about tracking other agents
                #loop through each module
                for m in range(0,len(agents[i].modules)):
                    agents[i].modules[m].start_tracking(agents[j])

    #initialize module state parameters
    for i in range(0,num_agents):
        #loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = np.copy(agents[i].modules[m].state)

if load_training_data:
    if os.path.isfile('training_data.pkl'):
        #if so, load it
        print("Q learning data found, loading it now")
        with open('training_data.pkl', 'rb') as f:
            [module_names, tables, states] = pickle.load(f)
        
        # for agnt in agents:
        #     for mod in agnt.modules
        #         agents[0].modules[i].__class__.__name__
        
        for h in range(0,len(module_names)):
            for i in range(0,num_agents):
                for j in range(0,len(agents[0].modules)):
                    print('loading training data!!!')
                    if agents[i].modules[j].__class__.__name__ == module_names[h]:
                        agents[i].modules[j].Q.q_table = cp.copy(tables[h])
                        agents[i].modules[j].Q.q_states = cp.copy(states[h])

##############################################################################
#   Initialization
##############################################################################

##############################################################################
#   main algorithm
##############################################################################

#plotting for visualization
if(visualize):
    frame_rate = 10
    axis_bounds = [search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]]
    plt.axis(axis_bounds)
    plt.draw()
    plt.pause(1/frame_rate)
    plt.clf()
    plt.cla()

print('begnning training')
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
                # print("agent left search space, ending episode")
                agent_out_of_bounds = True
                # instead, move agent back in bounds.
                # agnt.position = np.array([0,0], dtype='f')

        if(visualize):
            for agnt in agents:
                plt.plot(agnt.position[0],agnt.position[1],'ro')
                plt.axis(axis_bounds)
                for mod in agnt.modules:
                    mod.visualize()

        # criteria for ending the episode early.
        if(agent_out_of_bounds):
            break

        for agnt in agents:

            #select the next action (action_prime) for the agent to take 
            agnt.select_next_action()

            for mod in agnt.modules:

                #TODO move this up a level. Will only select one action based on all modules
                #select the next action (action_prime) for the agent to take 
                #mod.select_next_action()
                
                #find what the state (state_prime) would be if that action were taken
                mod.update_state_prime()

                #determine the reward for executing the action (not prime) in the state (not prime)
                #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
                mod.update_instant_reward()
                
                #Add the reward for this action to the total reward earned by the agent 
                mod.update_total_reward()
                
                #update the Q table
                mod.update_q()

                #run additional functions specific to each module
                #for example, the collision module uses this to track collisions with other agents 
                mod.auxilariy_functions()

                #prepare for next time step
                mod.action = cp.copy(mod.action_prime)
                mod.state  = np.copy(mod.state_prime)
 
        #plotting for visualization
        if(visualize):
            # plt.draw()
            plt.pause(1/frame_rate)
            plt.clf()
            plt.cla()
    
    #store the total reward for each agent at the end of each episode for algorithm performance analysis
    episode_rewards = np.zeros(num_agents) 
    for a in range(0,num_agents):
        episode_rewards[a] = cp.copy(agents[a].total_reward)

    if agent_rewards.size == 0:
        agent_rewards = cp.copy(episode_rewards)
    else:
        agent_rewards = np.vstack([agent_rewards,episode_rewards])

    #reset the agents (except for the Q tables and Q states) to start fesh for the next episode         
    ReinitializeAgents(agents,init_space)

    #save the trained agents to a file
    agent_filename = 'agents.pkl'
    with open(agent_filename,'wb') as f:
        pickle.dump(agents,f)

print('training complete')


#store the program end time so we can calculate how long it took for the code to execute
end_time = time.time() 
print('Program execution time:')
print(end_time-start_time)

##############################################################################
#   main algorithm
##############################################################################

##############################################################################
#   data 
##############################################################################

# mat = agents[0].modules[0].Q.q_table# np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# with open('outfile.txt') as f:
#     for line in mat:
#         np.savetxt(f, line, fmt='%.2f')



# #average and save the Q tables for each agent
# for i in range(0,len(agents[0].modules)):
#     q_table = np.array([])
#     q_states = np.array([])
#     number_experienced = np.array([])

#     for j in range(0,num_agents):
#         for k in range(0, agents[j].modules[i].Q.q_states):
#             working_state = agents[j].modules[i].Q.q_states[k]
#             if(any(np.equal(q_states,working_state).all(1))):




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
  
    

