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
# EXAMPLE: python main.py --simName test --description "Hello World!"

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--simName", type=str, default="SimulationResults", required=False,
	help="simName == Name of Simulation or Test")
ap.add_argument("--description", type=str, default=" ", required=False,
	help="description == Description of the test being run") 
args = vars(ap.parse_args())


##############################################################################
#   Helper Functions
##############################################################################

# Check if a given position is within given bounds
def checkInBounds(position,bounds):
    
    # TODO: Make sure position and bound have same number of 
    for i in range(0,len(position)):
        if not ( bounds[i][0] <= position[i] <= bounds[i][1]):
            return False
    return True

# Reset the agents to initial conditions (except for the Q states and tables)
def ReinitializeAgents(agents,bounds):

    arena_space = Simulation.arena_space 
    
    # Save Last Episodes Collisions, Reset Collision
    Simulation.obstacle_episode_collision_count.append(Simulation.obstacle_collision_count)
    Simulation.obstacle_collision_count = 0

    Simulation.agent_episode_collision_count.append(Simulation.agent_collision_count)
    Simulation.agent_collision_count = 0

    Simulation.boundary_episode_collision_count.append(Simulation.boundary_collision_count)
    Simulation.boundary_collision_count = 0

    Simulation.target_episode_entries_count.append(Simulation.target_entries_count)
    Simulation.target_entries_count = 0
    
    # if(Simulation.target_random):
    #     # Simulation.targets = np.array([random.randint(arena_space[0][0], arena_space[0][1]),
    #     #                                random.randint(arena_space[1][0], arena_space[1][1])])
    #     Simulation.targets = np.array([Simulation.r*np.cos(Simulation.n*2*np.pi*(Simulation.num_episodes)), Simulation.r*np.sin(0)])
    # else:
    #     Simulation.targets = Simulation.target_array[0]

    # Reinitialize Setting Parameters
    if (Simulation.Arena == 0):
        Simulation.obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,10), random.randint(1,10)])
        for i in range(1,Simulation.num_obstacles):
            temp_obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,10), random.randint(1,10)])
            Simulation.obstacles = np.vstack((Simulation.obstacles, temp_obstacles))

    # Initialize agent parameters
    for i in range(0,len(agents)):
        init_space = Simulation.init_space
        agents[i].position = np.array([random.randint(init_space[0][0], init_space[0][1]),random.randint(init_space[1][0], init_space[1][1])], dtype='f')
        agents[i].total_reward = 0
        
    # Initialize module parameters
    for i in range(0,len(Simulation.agents)):
        # Loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].action = Action.STAY
            agents[i].modules[m].action_prime = Action.STAY
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = np.copy(agents[i].modules[m].state)

##############################################################################
#   Simulation Variables
##############################################################################

agent_rewards = np.array([])   # matrix containing total reward values for each agent for each episode

# Make new Directories
raw_path = os.getcwd()
filename = str(args["simName"])
path = filename
try:  
   os.mkdir(path)
except OSError:
    print("")

##############################################################################
#   Save Simulation Configuration Settings
##############################################################################
# Store the program start time so we can calculate how long it took for the code to execute
start_time = time.time() 

# Save Configuration to a test file
if(not Simulation.visualize):
    if os.path.exists(os.path.join(path ,'Simulation_Configuration.txt')):
        file = open(os.path.join(path ,'Simulation_Configuration.txt'),'a') 
    else:    
        file = open(os.path.join(path ,'Simulation_Configuration.txt'),'w') 
    file.write(" \n \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
    file.write(str(args["simName"])+" -- "+ str(start_time) +" \n")
    file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n \n")
    file.write(str(args["description"]) + "\n \n")
    
    if Simulation.ControllerType == 0:
        file.write("Active Controller: Steve and Bucci Controller \n \n")
    elif Simulation.ControllerType == 1:
        file.write("Active Controller: Importance Function \n \n")
    else:
        file.write("Active Controller: Neural Network \n \n")

    file.write("~~~~ ARENA PARAMETERS ~~~~ \n")
    file.write("num_agents:    " + str(Simulation.num_agents)+"\n")
    file.write("num_episodes:  " + str(Simulation.num_episodes)+"\n")
    file.write("Number of Obstacles: "+str(Simulation.num_obstacles)+"\n")
    file.write("episode_length:  " + str(Simulation.episode_length)+"\n")
    file.write("exploitation_rise_time:  " + str(Simulation.exploitation_rise_time )+"\n")
    file.write("init_space:  " + str(Simulation.init_space)+"\n")
    file.write("search_space:  " + str(Simulation.search_space)+"\n \n")

    file.write("~~~~ ACTIVE MODULES ~~~~ \n")
    file.write("Cohesion Module ------- "+str(Simulation.CohesionModule) + "\n")
    file.write("Collision Module ------ "+str(Simulation.CollisionAvoidanceModule) + "\n")
    file.write("Out of Bounds Module -- "+str(Simulation.OutOfBoundsModule) + "\n")
    file.write("Target Seek Module ---- "+str(Simulation.TargetSeekingModule) + "\n")
    file.write("Obstacle Module ------- "+str(Simulation.ObstacleAvoidanceModule) + "\n \n")
    file.write("Module Weights: " + str(Simulation.module_weights) + "\n \n")
    file.close() 

##############################################################################
#   Initialization
##############################################################################
print('initializing agents')
initialized = False
# Check if a file containing a list of agents already exits
if Simulation.load_agents:
    if os.path.isfile(filename + '/agents.pkl'):
        # If so, load it
        print("Agent data found, loading it now")
        # TODO: Handle if the desired number of agents is different from the number of agents saved to disk
        with open(filename + '/agents.pkl', 'rb') as f:
            Simulation.agents = pickle.load(f)
        initialized = True

if not initialized:
    # If not, initialize a set of agents from scratch

    # Initialize agent positions
    for i in range(0,Simulation.num_agents):
        init_space = Simulation.init_space
        position = np.array([random.randint(init_space[0][0], init_space[0][1]),random.randint(init_space[1][0], init_space[1][1])], dtype='f')
        # position = np.array([2*i,2*i], dtype='f')
        Simulation.agents.append(Agent(position))

    # Initialize module parameters such as who each agent is tracking
    # TODO: Make it so the tracked agents are based on range and updated every iteration
    # NOTE: It is important to start tracking agents before loading training data
    for i in range(0,Simulation.num_agents):
        for j in range(0,Simulation.num_agents):
            if(i != j):
                # TODO: Change this? not every module will care about tracking other agents
                # Loop through each module
                for m in range(0,len(Simulation.agents[i].modules)):
                    Simulation.agents[i].modules[m].start_tracking(Simulation.agents[j])

    # Initialize module state parameters
    for i in range(0,Simulation.num_agents):
        #loop through each module
        for m in range(0,len(Simulation.agents[i].modules)):
            Simulation.agents[i].modules[m].update_state()
            Simulation.agents[i].modules[m].state_prime = np.copy(Simulation.agents[i].modules[m].state)

    # NOTE: It is important to start tracking agents before loading training data
    if Simulation.load_training_data:
        for i in range(0,len(Simulation.agents[0].modules)):
            training_filename = path +'/'+ Simulation.agents[0].modules[i].__class__.__name__ + '_training_data.pkl'
            
            if os.path.isfile(training_filename):
                print("Q learning data found, loading it now")        
                with open(training_filename, 'rb') as f:
                    [module_name, table, states] = pickle.load(f)

                if Simulation.agents[0].modules[i].collapsable_Q:
                    for agnt in Simulation.agents:
                        for Q in agnt.modules[i].Q:
                            Q.q_table = cp.copy(table)
                            Q.q_states = cp.copy(states)
                else:
                    for agnt in Simulation.agents:
                        for q in range(0,len(agnt.modules[i].Q)):
                            agnt.modules[i].Q[q].q_table = cp.copy(table[q])
                            agnt.modules[i].Q[q].q_states = cp.copy(states[q])
                    # for q in range(0,len(Simulation.agents[0].module[i].Q)):
                    #     for agt in Simulation.agents:



##############################################################################
#   main algorithm
##############################################################################

# Plotting for visualization
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

    # if(Simulation.target_random):
    #     # For Training Only: Increase Search Space as episodes increase.
    #     if (e != 0):
    #         if (e%2000 == 0):
    #             Simulation.arena_space = [[Simulation.arena_space[0][0] -10, Simulation.arena_space[0][1] + 10],[Simulation.arena_space[1][0] -10, Simulation.arena_space[1][1] +10]]
    #             print(Simulation.arena_space)

    if(Simulation.target_random):
        Simulation.targets = np.array([Simulation.r*np.cos(Simulation.n*2*np.pi*(e/Simulation.num_episodes)), Simulation.r*np.sin(Simulation.n*2*np.pi*(e/Simulation.num_episodes))])
    else:
        Simulation.targets = Simulation.target_array[0]

    for t in range(0,Simulation.episode_length):
        
        agent_out_of_bounds = False
        Simulation.episode_iter_num = t


        # print('agents take actions')
        for agnt in Simulation.agents:

            # Take the action determined in the last step
            #  Update agent positions on plots
            # print('state is', agnt.modules[0].state)
            agnt.take_action(agnt.modules[0].action)
            # print('taking action ', agnt.modules[0].action)

            # Check if any agent went out of search space.
            #  Terminate episode if so
            if not (checkInBounds(agnt.position,Simulation.search_space)):
                print("agent left search space, ending episode")
                Simulation.boundary_collision_count = Simulation.boundary_collision_count + 1
                agent_out_of_bounds = True

        if(Simulation.visualize):
            plt.grid(linestyle='--', linewidth='0.5', color='grey')
            for agnt in Simulation.agents:
                plt.plot(agnt.position[0],agnt.position[1],'ro')
                plt.axis(axis_bounds)
                
                for mod in agnt.modules:
                    mod.visualize()
            # Convert the figure into an array and append it to images array        
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)

        # print('update state prime and select next action')
        for agnt in Simulation.agents:
            for mod in agnt.modules:
                # Find what the state (state_prime) would be if that action were taken
                mod.update_state_prime()
                # print('state prime is ', mod.state_prime)

            # Select the next action (action_prime) for the agent to take 
            agnt.select_next_action()
            # print('next action is ', agnt.modules[0].action_prime)

        # print('instant and total reward, update q, action == action prime, state == state prime')
        for agnt in Simulation.agents:
            for mod in agnt.modules:
                # Determine the reward for executing the action (not prime) in the state (not prime)
                #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
                mod.update_instant_reward()
                # print('instant reward is ', mod.instant_reward[0])

                # Add the reward for this action to the total reward earned by the agent 
                mod.update_total_reward()
                
                # Update the Q table
                mod.update_q()

                # Run additional functions specific to each module
                #  For example, the collision module uses this to track collisions with other agents 
                mod.auxilariy_functions()

                # Prepare for next time step
                mod.action = cp.copy(mod.action_prime)
                mod.state  = np.copy(mod.state_prime)
 
        # Plotting for visualization
        if(Simulation.visualize):
            plt.pause(1/frame_rate)
            plt.clf()
            plt.cla()

        # Criteria for ending the episode early.
        if(agent_out_of_bounds):
            break    
    
    # Store the total reward for each agent at the end of each episode for algorithm performance analysis
    episode_rewards = np.zeros(len(Simulation.agents)) 
    for a in range(0,len(Simulation.agents)):
        episode_rewards[a] = cp.copy(Simulation.agents[a].total_reward)

    if agent_rewards.size == 0:
        agent_rewards = np.array([cp.copy(episode_rewards)])
    else:
        agent_rewards = np.vstack([agent_rewards,episode_rewards])

    # Reset the agents (except for the Q tables and Q states) to start fresh for the next episode         
    ReinitializeAgents(Simulation.agents,Simulation.init_space)
    
    for agnt in Simulation.agents:
        for mod in agnt.modules:
            mod.reset_init(e)


    # There are occasional permission errors, this block will keep retrying until the dump succeeds
    # TODO: Make this save every so often in case of errors so the history isn't lost
    agent_filename = filename+'/agents.pkl'

    max_dump_attempts = 5
    dump_attempts = 0
    pe = True
    while pe:
        pe = False
        try:
            with open(agent_filename,'wb') as f:
                pickle.dump(Simulation.agents,f)  
        except Exception as e:
            pe = True
            dump_attempts = dump_attempts + 1
            
            print(e)
            print('permission error while saving to disk, retrying...')
            time.sleep(0.5)

            if dump_attempts == max_dump_attempts:
                print('******PERMISSION ERROR, COULD NOT DUMP AGENTS TO DISK********')
        
print('Training complete')

# Store the program end time so we can calculate how long it took for the code to execute
end_time = time.time() 
print('Program execution time:')
print(end_time-start_time)

##############################################################################
#   Data Storage
##############################################################################
timestr = time.strftime("%m%d-%H%M")

# Export the visualizer as a *.gif
if(Simulation.visualize):
    kwargs_write = {'fps':10, 'quantizer':'nq'}
    imageio.mimsave(os.path.join(filename, "Animation"+timestr+".gif"), images, fps=10)

# Store the iterations and total rewards for each agent for each episode
iterations = np.arange(Simulation.num_episodes)
if(os.path.isfile(filename+'/agent_rewards.pkl')):
    agent_reward_filename = filename+'/agent_rewards'+timestr+'.pkl'
else:
    agent_reward_filename = filename+'/agent_rewards.pkl'

# Store the iterations and total collisions for each episode 
total_collisions = np.sum([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count], axis=0)
if(os.path.isfile(filename+'/total_collisions.pkl')):
    total_collisions_filename = filename+'/total_collisions'+timestr+'.pkl'
else:
    total_collisions_filename = filename+'/total_collisions.pkl'

#NOTE: There are occasional permission errors, this block will keep retrying until the dump succeeds
#TODO: Make this save every so often in case of errors so the history isn't lost
max_dump_attempts = 5
dump_attempts = 0
pe = True
while pe:
    pe = False
    try:
        with open(agent_reward_filename,'wb') as f:
            pickle.dump([iterations, agent_rewards],f)  
        with open(total_collisions_filename,'wb') as g:
            pickle.dump([iterations, total_collisions_filename],g) 
    except Exception as e:
        pe = True
        dump_attempts = dump_attempts + 1
    
        print(e)
        print('permission error while saving to disk, retrying...')
        time.sleep(0.5)

        if dump_attempts == max_dump_attempts:
            print('******PERMISSION ERROR, COULD NOT DUMP AGENT REWARDS TO DISK********')

# Iterations-Reward Plot
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


# Collision Box and Whisker Plot
fig1, ax1 = plt.subplots()
ax1.set_title('Collision Tracker')
ax1.boxplot([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count])
plt.xlabel("Collision Type")
plt.ylabel("Collisions")
ax1.set_xticklabels(['Agent Collisions', 'Obstacle Collisions', 'Boundary Collisions'])

if(os.path.isfile(filename+'/Collisions.jpeg')):
    fig1.savefig(os.path.join(filename, "Collisions"+timestr+".jpeg") , orientation='landscape', quality=95)
else:
    fig1.savefig(os.path.join(filename, "Collisions.jpeg") , orientation='landscape', quality=95)

# Iterations-Targets Entered Plot
fig2, ax2 = plt.subplots()
plt.plot(iterations,Simulation.target_reached_episode_end)
plt.xlabel("Iterations")
plt.ylabel("Targets Reached")
plt.title('Iterations V. Targets Reached')

if(os.path.isfile(filename+'/TargetsReached.jpeg')):
    plt.savefig(os.path.join(filename, "TargetsReached"+timestr+".jpeg") , orientation='landscape', quality=95)
else:
    plt.savefig(os.path.join(filename, "TargetsReached.jpeg") , orientation='landscape', quality=95)


# Box Histograms
f, axarr = plt.subplots(2, 2)
axarr[0, 0].set_title('Target 1')
axarr[0, 1].set_title('Target 2')
axarr[1, 0].set_title('Target 3')
axarr[1, 1].set_title('Target 4')
# # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
# plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
# plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
temp1 = []
temp2 = []
temp3 = []
temp4 = []
for i in range(0,len(Simulation.target_histogram_data)):
    if Simulation.target_histogram_data[i][0] == 1:
        temp1.append(Simulation.target_histogram_data[i][1])
    if Simulation.target_histogram_data[i][0] == 2:
        temp2.append(Simulation.target_histogram_data[i][1])
    if Simulation.target_histogram_data[i][0] == 3:
        temp3.append(Simulation.target_histogram_data[i][1])
    if Simulation.target_histogram_data[i][0] == 4:
        temp4.append(Simulation.target_histogram_data[i][1])

num_bins = 100
bin = Simulation.episode_length/num_bins
bins = []
for i in range(0,num_bins):
    bins.append(i*bin)
axarr[0,0].hist(temp1,bins)
axarr[0,1].hist(temp2,bins)
axarr[1,0].hist(temp3,bins)
axarr[1,1].hist(temp4,bins)
#hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)[source]

plt.show()

# Append Results Data to Simulation.txt
if(not Simulation.visualize):
    file = open(os.path.join(path ,'Simulation_Configuration.txt'),'a')
    file.write("~~~~ RESULTS ~~~~ \n")
    file.write("Program Execution Time: "+str((end_time-start_time)/60)+" minutes \n")
    file.write("Mean Episode Agent-Agent Collisions: "+str(np.mean(Simulation.agent_episode_collision_count))+"\n")
    file.write("Mean Episode Agent-Obstacle Collisions: "+str(np.mean(Simulation.obstacle_episode_collision_count))+"\n")
    file.write("Mean Episode Agent-Boundary Collisions: "+str(np.mean(Simulation.boundary_episode_collision_count))+"\n")
    file.write("Mean Numver of Targets Reached: "+str(np.mean(Simulation.target_reached_episode_end))+"\n")
    