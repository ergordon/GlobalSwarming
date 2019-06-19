from agent import Agent
from simulation import Simulation
import numpy as np
from action import Action
from module import Module
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
    # Save Last Episodes Collisions, Reset Collision
    Simulation.obstacle_episode_collision_count.append(Simulation.obstacle_collision_count)
    Simulation.obstacle_collision_count = 0

    Simulation.agent_episode_collision_count.append(Simulation.agent_collision_count)
    Simulation.agent_collision_count = 0

    Simulation.boundary_episode_collision_count.append(Simulation.boundary_collision_count)
    Simulation.boundary_collision_count = 0

    Simulation.target_episode_entries_count.append(Simulation.target_entries_count)
    Simulation.target_entries_count = 0
    
    #Reinitialize Setting Parameters
    if (Simulation.Arena == 0):
        search_space = Simulation.search_space
        #Simulation.targets = np.array([-40,40])
        Simulation.targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                                       random.randint(search_space[1][0], search_space[1][1])])

        Simulation.obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10])
        for i in range(1,Simulation.num_obstacles):
            temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10])
            Simulation.obstacles = np.vstack((Simulation.obstacles, temp_obstacles))

    #initialize agent parameters
    for i in range(0,len(agents)):
        init_space = Simulation.init_space
        agents[i].position = np.array([random.randint(init_space[0][0], init_space[0][1]),random.randint(init_space[1][0], init_space[1][1])], dtype='f')
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
#   Save Simulation Configuration Settings
##############################################################################

## Save Configuration to a test file
    if(not Simulation.visualize):
        file = open(os.path.join(path ,'Simulation_Configuration.txt'),'w') 

        file.write("num_agents:  " + str(Simulation.num_agents)+"\n")
        file.write("num_episodes:  " + str(Simulation.num_episodes)+"\n")
        file.write("episode_length :  " + str(Simulation.episode_length)+"\n")
        file.write("exploitation_rise_time :  " + str(Simulation.exploitation_rise_time )+"\n")
        file.write("init_space:  " + str(Simulation.init_space)+"\n")
        file.write("search_space :  " + str(Simulation.search_space)+"\n")

        #file.write("Mean Luminance:  " + str(statistics.mean(LuminanceExact))+"\n")
        #file.write("STDe Luminance:  " + str(statistics.stdev(LuminanceExact))+"\n")
        #file.write("Mean Area :  " + str(statistics.mean(Area))+"\n")
        file.close() 
##############################################################################
#   Save Simulation Configuration Settings
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
if Simulation.load_agents:
    if os.path.isfile(filename + '/agents.pkl'):
        #if so, load it
        print("Agent data found, loading it now")
        #TODO handle if the desired number of agents is different from the number of agents saved to disk
        with open(filename + '/agents.pkl', 'rb') as f:
            agents = pickle.load(f)
        initialized = True

if not initialized:
    #if not, initialize a set of agents from scratch
    #initialize agent positions
    for i in range(0,Simulation.num_agents):
        init_space = Simulation.init_space
        position = np.array([random.randint(init_space[0][0], init_space[0][1]),random.randint(init_space[1][0], init_space[1][1])], dtype='f')
        agents.append(Agent(position))

    #initialize module parameters such as who each agent is tracking
    #TODO make it so the tracked agents are based on range and updated every iteration
    for i in range(0,Simulation.num_agents):
        for j in range(0,Simulation.num_agents):
            if(i != j):
                #TODO chagne this, not every module will care about tracking other agents
                #loop through each module
                for m in range(0,len(agents[i].modules)):
                    agents[i].modules[m].start_tracking(agents[j])

    #initialize module state parameters
    for i in range(0,Simulation.num_agents):
        #loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = np.copy(agents[i].modules[m].state)

if Simulation.load_training_data:
    if os.path.isfile('training_data.pkl'):
        #if so, load it
        print("Q learning data found, loading it now")
        with open('training_data.pkl', 'rb') as f:
            [module_names, tables, states] = pickle.load(f)
        
        # for agnt in agents:
        #     for mod in agnt.modules
        #         agents[0].modules[i].__class__.__name__
        
        for h in range(0,len(module_names)):
            for i in range(0,Simulation.num_agents):
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
    print("beginning episode #" + str(e+1))

    for t in range(0,Simulation.episode_length):
        agent_out_of_bounds = False

        for agnt in agents:

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
            for agnt in agents:
                plt.plot(agnt.position[0],agnt.position[1],'ro')
                plt.axis(axis_bounds)
                
                for mod in agnt.modules:
                    mod.visualize()
            #convert the figure into an array and append it to images array        
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)
        
        for agnt in agents:
            for mod in agnt.modules:
                #find what the state (state_prime) would be if that action were taken
                mod.update_state_prime()

            #select the next action (action_prime) for the agent to take 
            agnt.select_next_action()

        for agnt in agents:
            #select the next action (action_prime) for the agent to take 
            # agnt.select_next_action()

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
        episode_rewards[a] = cp.copy(agents[a].total_reward)

    if agent_rewards.size == 0:
        agent_rewards = cp.copy(episode_rewards)
    else:
        agent_rewards = np.vstack([agent_rewards,episode_rewards])

    #reset the agents (except for the Q tables and Q states) to start fresh for the next episode         
    ReinitializeAgents(agents,Simulation.init_space)
    
    for agnt in agents:
        for mod in agnt.modules:
            mod.reset_init(e)


    #there are occasional permission errors, this block will keep retrying until the dump succeeds
    #TODO make this save every so often in case of errors so the history isn't lost
    agent_filename = filename+'/agents.pkl'

    max_dump_attempts = 5
    dump_attempts = 0
    pe = True
    while pe:
        pe = False
        try:
            with open(agent_filename,'wb') as f:
                pickle.dump(agents,f)  
        except:
            pe = True
            dump_attempts = dump_attempts + 1
            
            print('permission error while saving to disk, retrying...')

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

#export the visualizer as a *.gif
if(Simulation.visualize):
    kwargs_write = {'fps':10, 'quantizer':'nq'}
    imageio.mimsave(os.path.join(filename, "Animation"+timestr+".gif"), images, fps=10)

#store the iterations and total rewards for each agent for each episode
iterations = np.arange(Simulation.num_episodes)
if(os.path.isfile(filename+'/agent_rewards.pkl')):
    agent_reward_filename = filename+'/agent_rewards'+timestr+'.pkl'
else:
    agent_reward_filename = filename+'/agent_rewards.pkl'

# pe = False

# try:
#     with open(agent_reward_filename,'wb') as f:
#         pickle.dump([iterations, agent_rewards],f)  
# except permissionError:
#     pe = True

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

        if dump_attempts == max_dump_attempts:
            print('******PERMISSION ERROR, COULD NOT DUMP AGENT REWARDS TO DISK********')

## Iterations-Reward Plot
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


## Collision Box and Whisker Plot
fig1, ax1 = plt.subplots()
ax1.set_title('Collision Tracker')
ax1.boxplot([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count])
plt.xlabel("Collision Type")
plt.ylabel("Collisions")
ax1.set_xticklabels(['Agent Collisions', 'Obstacle Collisions', 'Boundary Collisions'])

fig2, ax2 = plt.subplots()
plt.plot(iterations,Simulation.target_reached_episode_end)
plt.xlabel("Iterations")
plt.ylabel("Targets Reached")
plt.title('Iterations V. Targets Reached')

if(os.path.isfile(filename+'/IterationsVReward.jpeg')):
    plt.savefig(os.path.join(filename, "IterationsVReward"+timestr+".jpeg") , orientation='landscape', quality=95)
else:
    plt.savefig(os.path.join(filename, "IterationsVReward.jpeg") , orientation='landscape', quality=95)


total_collisions = np.sum([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count], axis=0)

if(os.path.isfile(filename+'/total_collisions.pkl')):
    total_collisions_filename = filename+'/total_collisions'+timestr+'.pkl'
else:
    total_collisions_filename = filename+'/total_collisions.pkl'

with open(total_collisions_filename,'wb') as f:
    pickle.dump([iterations, total_collisions_filename],f)  


if(os.path.isfile(filename+'/Collisions.jpeg')):
    fig1.savefig(os.path.join(filename, "Collisions"+timestr+".jpeg") , orientation='landscape', quality=95)
else:
    fig1.savefig(os.path.join(filename, "Collisions.jpeg") , orientation='landscape', quality=95)
    
plt.show()
##############################################################################
#   data
##############################################################################
  
    

