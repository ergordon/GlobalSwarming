from agent import Agent
from simulation import Simulation
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import pickle
import os.path
import argparse
import random
import copy as cp
import imageio

##############################################################################
#   Argument Parser
##############################################################################
# EXAMPLE: python plot_data.py --file agent_rewards_DistanceOnly.pkl
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--simName", type=str, default="SimulationResults", required=False,
	help="simName == Name of Simulation or Test")
args = vars(ap.parse_args())


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
    #reintizilize target
    search_space = Simulation.search_space
    Simulation.targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                         random.randint(search_space[1][0], search_space[1][1])])
    Simulation.obstacles = np.array([[random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]),random.randint(1,10), random.randint(1,10)], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)]])
    #initialize agent parameters
    for i in range(0,len(agents)):
        #TODO make this initial position randomized
        agents[i].position = np.array([2*i,2*i], dtype='f')
        agents[i].total_reward = 0
        
    #initialize module parameters
    for i in range(0,Simulation.num_agents):
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

agent_rewards = np.array([])   # matrix containing total reward values for each agent for each episode

## Make new Directories
raw_path = os.getcwd()
filename = str(args["simName"])
path = filename
try:  
   os.mkdir(path)
except OSError:
    print("")

##############################################################################
#   Simulation Variables
##############################################################################

##############################################################################
#   Initialization
##############################################################################
print('initializing agents')

#store the program start time so we can calculate how long it took for the code to execute
start_time = time.time() 

# agents = list() #list of agents

initialized = False
#check if a file containing a list of agents already exits
if Simulation.load_agents:
    if os.path.isfile(filename + '/agents.pkl'):
        #if so, load it
        print("Agent data found, loading it now")
        #TODO handle if the desired number of agents is different from the number of agents saved to disk
        with open(filename + '/agents.pkl', 'rb') as f:
            Simulation.agents = pickle.load(f)
        initialized = True

if not initialized:
    #if not, initialize a set of agents from scratch
    #initialize agent positions
    for i in range(0,Simulation.num_agents):
        position = np.array([2*i,2*i], dtype='f')
        Simulation.agents.append(Agent(position))

    #initialize module parameters such as who each agent is tracking
    #TODO make it so the tracked agents are based on range and updated every iteration
    for i in range(0,Simulation.num_agents):
        for j in range(0,Simulation.num_agents):
            if(i != j):
                #TODO chagne this, not every module will care about tracking other agents
                #loop through each module
                for m in range(0,len(Simulation.agents[i].modules)):
                    Simulation.agents[i].modules[m].start_tracking(Simulation.agents[j])

    #initialize module state parameters
    for i in range(0,Simulation.num_agents):
        #loop through each module
        for m in range(0,len(Simulation.agents[i].modules)):
            Simulation.agents[i].modules[m].update_state()
            Simulation.agents[i].modules[m].state_prime = np.copy(Simulation.agents[i].modules[m].state)

# if Simulation.load_training_data:
#     if os.path.isfile('training_data.pkl'):
#         #if so, load it
#         print("Q learning data found, loading it now")
#         with open('training_data.pkl', 'rb') as f:
#             [module_names, tables, states] = pickle.load(f)
        
#         # for agnt in Simulation.agents:
#         #     for mod in agnt.modules
#         #         Simulation.agents[0].modules[i].__class__.__name__
        
#         for h in range(0,len(module_names)):
#             for i in range(0,Simulation.num_agents):
#                 for j in range(0,len(Simulation.agents[0].modules)):
#                     print('loading training data!!!')
#                     if Simulation.agents[i].modules[j].__class__.__name__ == module_names[h]:
#                         Simulation.agents[i].modules[j].Q.q_table = cp.copy(tables[h])
#                         Simulation.agents[i].modules[j].Q.q_states = cp.copy(states[h])

    if Simulation.load_training_data:
        for i in range(0,len(Simulation.agents[0].modules)):
            filename = Simulation.agents[0].modules[i].__class__.__name__ + '_training_data.pkl'
            if os.path.isfile(filename):
                print("Q learning data found, loading it now")        
                with open(filename, 'rb') as f:
                    [module_name, table, states] = pickle.load(f)

                for j in range(0,Simulation.num_agents):
                    for k in range(0, Simulation.agents[j].modules[i].Q):
                        Simulation.agents[j].modules[i].Q[k].q_table = cp.copy(table)
                        Simulation.agents[j].modules[i].Q[k].q_states = cp.copy(states)


##############################################################################
#   Initialization
##############################################################################

##############################################################################
#   main algorithm
##############################################################################

#plotting for visualization
if(Simulation.visualize):
    fig, ax = plt.subplots()
    images = []
    frame_rate = 10
    axis_bounds = [Simulation.search_space[0][0], Simulation.search_space[0][1], Simulation.search_space[1][0], Simulation.search_space[1][1]]
    plt.axis(axis_bounds)
    plt.draw()
    plt.pause(1/frame_rate)
    plt.clf()
    plt.cla()
    plt.axis('equal')

print('beginning training')
for e in range(0,Simulation.num_episodes):
    Simulation.episode_iter_num = 0

    print("beginning episode #" + str(e+1))

    for t in range(0,Simulation.episode_length):
        
        agent_out_of_bounds = False
        Simulation.episode_iter_num = t

        for agnt in Simulation.agents:

            #take the action determined in the last step
            #update agent positions on plots
            agnt.take_action(agnt.modules[0].action)

            #check if any agent went out of search space.
            #terminate episode if so
            if not (checkInBounds(agnt.position,Simulation.search_space)):
                print("agent left search space, ending episode")
                agent_out_of_bounds = True

        if(Simulation.visualize):
            plt.grid(linestyle='--', linewidth='0.5', color='grey')
            for agnt in Simulation.agents:
                plt.plot(agnt.position[0],agnt.position[1],'ro')
                plt.axis(axis_bounds)
                
                for mod in agnt.modules:
                    mod.visualize()
            #convert the figure into an array and append it to images array        
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)

        for agnt in Simulation.agents:
            for mod in agnt.modules:
                #find what the state (state_prime) would be if that action were taken
                mod.update_state_prime()

            #select the next action (action_prime) for the agent to take 
            agnt.select_next_action()



        for agnt in Simulation.agents:
            for mod in agnt.modules:
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
        if(Simulation.visualize):
            plt.pause(1/frame_rate)
            plt.clf()
            plt.cla()

        # criteria for ending the episode early.
        if(agent_out_of_bounds):
            break    
    
    #store the total reward for each agent at the end of each episode for algorithm performance analysis
    episode_rewards = np.zeros(Simulation.num_agents) 
    for a in range(0,Simulation.num_agents):
        episode_rewards[a] = cp.copy(Simulation.agents[a].total_reward)

    if agent_rewards.size == 0:
        agent_rewards = np.array([cp.copy(episode_rewards)])
    else:
        agent_rewards = np.vstack([agent_rewards,episode_rewards])


    #reset the agents (except for the Q tables and Q states) to start fresh for the next episode         
    ReinitializeAgents(Simulation.agents,Simulation.init_space)



    #there are occasional permission errors, this block will keep retrying until the dump succeeds
    agent_filename = filename+'/agents.pkl'

    max_dump_attempts = 5
    dump_attempts = 0
    pe = True
    while pe:
        pe = False
        try:
            with open(agent_filename,'wb') as f:
                pickle.dump(Simulation.agents,f)  
        except:
            pe = True
            dump_attempts = dump_attempts + 1
            
            print('permission error while saving to disk, retrying...')
            time.sleep(0.5)

            if dump_attempts == max_dump_attempts:
                print('******PERMISSION ERROR, COULD NOT DUMP AGENTS TO DISK********')
        
        
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
timestr = time.strftime("%m%d-%H%M")

#export the visualizer as a *.gif
if(Simulation.visualize):
    kwargs_write = {'fps':10, 'quantizer':'nq'}
    imageio.mimsave(os.path.join(filename, "Animation.gif"), images, fps=10)   

#store the iterations and total rewards for each agent for each episode
iterations = np.arange(Simulation.num_episodes)
if(os.path.isfile(filename+'/agent_rewards.pkl')):
    agent_reward_filename = filename+'/agent_rewards'+timestr+'.pkl'
else:
    agent_reward_filename = filename+'/agent_rewards.pkl'



#there are occasional permission errors, this block will keep retrying until the dump succeeds
#TODO make this save every so often in case of errors so the history isn't lost
max_dump_attempts = 5
dump_attempts = 0
pe = True
while pe:
    pe = False
    try:
        with open(agent_reward_filename,'wb') as f:
            pickle.dump([iterations, agent_rewards],f)  
    except:
        pe = True
        dump_attempts = dump_attempts + 1
    
        print('permission error while saving to disk, retrying...')
        time.sleep(0.5)

        if dump_attempts == max_dump_attempts:
            print('******PERMISSION ERROR, COULD NOT DUMP AGENT REWARDS TO DISK********')

#close the visualization plot and create a new plot of each agents total reward over time
plt.close()
for i in range(0,Simulation.num_agents):
    plt.plot(iterations,agent_rewards[:,i])
plt.xlabel("Iterations")
plt.ylabel("Reward Value")
plt.title('Iterations V. Reward')

if(os.path.isfile(filename+'/IterationsVReward.jpeg')):
    plt.savefig(os.path.join(filename, "IterationsVReward"+timestr+".jpeg") , orientation='landscape', quality=95)
else:
    plt.savefig(os.path.join(filename, "IterationsVReward.jpeg") , orientation='landscape', quality=95)

plt.show()
##############################################################################
#   data
##############################################################################
  
    

