from agent import Agent
from simulation import Simulation
from simulation import TargetPath
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
import math
import ga

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
    
    # Reinitialize Setting Parameters
    if (Simulation.Arena == 0):
        Simulation.obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,Simulation.max_obstacle_size), random.randint(1,Simulation.max_obstacle_size)])
        for i in range(1,Simulation.num_obstacles):
            temp_obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,Simulation.max_obstacle_size), random.randint(1,Simulation.max_obstacle_size)])
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

#for z in range(0,10):
def mainSARSA():
    ##############################################################################
    #   Simulation Variables
    ##############################################################################

    agent_rewards = np.array([])   # matrix containing total reward values for each agent for each episode

    # Make new Directories
    raw_path = os.getcwd()
    filename = "TrainedData"
    path = filename
    try:  
        os.mkdir(path)
    except OSError:
        pass
        #print("")

    ##############################################################################
    #   Save Simulation Configuration Settings
    ##############################################################################
    # Store the program start time so we can calculate how long it took for the code to execute
    start_time = time.time()
    timestr = time.strftime("%m%d-%H%M")


    ##############################################################################
    #   Initialization
    ##############################################################################
    #print('initializing agents')
    initialized = False
    # # Check if a file containing a list of agents already exits
    # if Simulation.load_agents:
    #     if os.path.isfile(filename + '/agents.pkl'):
    #         # If so, load it
    #         print("Agent data found, loading it now")
    #         # TODO: Handle if the desired number of agents is different from the number of agents saved to disk
    #         with open(filename + '/agents.pkl', 'rb') as f:
    #             Simulation.agents = pickle.load(f)
    #         initialized = True

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
                    #print("Q learning data found, loading it now")        
                    with open(training_filename, 'rb') as f:
                        [module_name, data, updates] = pickle.load(f)

                    if Simulation.agents[0].modules[i].collapsable_Q:
                        for agnt in Simulation.agents:
                            for Q in agnt.modules[i].Q:
                                Q.q_data = cp.copy(data[0])
                                Q.q_updates = cp.copy(updates[0])
                    else:
                        for agnt in Simulation.agents:
                            for q in range(0,len(agnt.modules[i].Q)):
                                agnt.modules[i].Q[q].q_data = cp.copy(data[q])
                                agnt.modules[i].Q[q].q_updates = cp.copy(updates[q])
                        



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

    #print('beginning training')
    for e in range(0,Simulation.num_episodes):
        Simulation.episode_iter_num = 0
        
        # At the start of a new episode, initilize the target to appropriate location.
        TargetType = Simulation.TargetType

        # Planned Target Trajectory
        if (TargetType == TargetPath.Planned):
            Simulation.targets = Simulation.target_array[0]

        # Circular Target Trajectory
        elif (TargetType == TargetPath.Circle):
            Simulation.targets = np.array([Simulation.r*np.cos(Simulation.n*2*np.pi*(e/Simulation.num_episodes)), Simulation.r*np.sin(Simulation.n*2*np.pi*(e/Simulation.num_episodes))])
            Simulation.targets = np.round(Simulation.targets)

        # Random Target Trajectory
        elif (TargetType == TargetPath.Random):
            Simulation.targets = np.array([random.randint(Simulation.arena_space[0][0], Simulation.arena_space[0][1]),
                                random.randint(Simulation.arena_space[1][0], Simulation.arena_space[1][1])])


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
                    Simulation.boundary_collision_count = Simulation.boundary_collision_count + 1
                    agent_out_of_bounds = True

            if(Simulation.visualize):
                plt.grid(linestyle='--', linewidth='0.5', color='grey')
                for agnt in Simulation.agents:
                    plt.plot(agnt.position[0],agnt.position[1],'ro')
                    plt.axis(axis_bounds)
                    
                    for mod in agnt.modules:
                        mod.visualize()
                
                if (t%5 == 0):
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
                    # For example, the collision module uses this to track collisions with other agents 
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
            
    #print('Training complete')

    ##############################################################################
    #   Data Storage
    ##############################################################################
    '''
    # Export the visualizer as a *.gif
    if(Simulation.visualize):
        fps = 60
        kwargs_write = {'fps':fps, 'quantizer':'nq'}
        imageio.mimsave(os.path.join(filename, "Animation"+timestr+".gif"), images, fps=fps)

    '''

    iterations = np.arange(Simulation.num_episodes)
    total_collisions = np.sum([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count], axis=0)

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

    temp1 = np.mean(temp1)
    temp2 = np.mean(temp2)
    temp3 = np.mean(temp3)
    temp4 = np.mean(temp4)

    # print("Iter to Target 1 : "+str(temp1))
    # print("Iter to Target 2 : "+str(temp2))
    # print("Iter to Target 3 : "+str(temp3))
    # print("Iter to Target 4 : "+str(temp4))
    # print("Total Collisions "+str(np.sum(total_collisions)))

    if math.isnan(temp1):
        temp1 = 5000
    if math.isnan(np.mean(temp2)):
        temp2 = 5000
    if math.isnan(np.mean(temp4)):
        temp3 = 5000
    if math.isnan(np.mean(temp4)):
        temp4 = 5000 # Arbitraty penalty number for not reaching target
    
    return temp1 + temp2 + temp3 + temp4 + np.sum(total_collisions)

# #############################################################################
#   Reset Simulation Parameters for Continuose Testing
# #############################################################################
def resetInits():
    Simulation.agents = list()        # List of agents
    Simulation.obstacle_collision_count = 0          # Number of collisions (Agent-Agent)
    Simulation.obstacle_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)

    Simulation.agent_collision_count = 0             # Number of collisions (Agent-Agent)
    Simulation.agent_episode_collision_count = []    # Number of collisions during a single episode (Agent-Agent)

    Simulation.boundary_collision_count = 0          # Number of collisions (Agent-Agent)
    Simulation.boundary_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)

    Simulation.target_entries_count = 0              # Number of agents in the target region
    Simulation.target_episode_entries_count = []     # Number agents in a target at the end of an episode
    Simulation.target_agents_remaining = 0
    Simulation.target_histogram_data = [] # Data about what iteration a target was reached.

    Simulation.episode_iter_num = 0   # Track the current interation of the episode. Used with exploration/exploitation


##############################################################################
#   Genetic Algorithm Module-Weight Selection
##############################################################################
np.set_printoptions(precision=2)

## GA Parameters
num_weights = 5         # Number of the weights we are looking to optimize.
sol_per_pop = 4         # Population Size
num_parents_mating = 2  # Mating Pool Size
num_generations = 2     # Number of Generations
best_outputs = []

# Defining the population size.
pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

#Creating the initial population.
new_population = np.random.uniform(low=0, high=1e10, size=pop_size)

for generation in range(num_generations):

    fitness = []

    print("\nGeneration : " + str(generation))

    # Normalize every population
    for i in range(0,len(new_population)):
        new_population[i] /= new_population[i].sum()

    print("Population : \n", np.round(new_population,3))
    
    # Measuring the fitness of each chromosome in the population.
    fitness = np.empty((sol_per_pop,), dtype=object)
    for pop in range(sol_per_pop):
        Simulation.module_weights = new_population[pop]
        fitness[pop] = mainSARSA()
        resetInits()

    print("Fitness     : ", fitness)

    # The best result in the current generation
    best_outputs.append(np.min(fitness))
    
    # Select the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)
    # print("Parents")
    # print(np.round(parents,3))

    # Generate the next generation using crossover.
    offspring_crossover = ga.crossover(parents, offspring_size=(pop_size[0]-parents.shape[0], num_weights))
    # print("Crossover")
    # print(np.round(offspring_crossover,3))

    # Add some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
    # print("Mutation")
    # print(np.round(offspring_mutation,3))

    # Create the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
    # Get the best solution after a complete generation. Finish for all generations.

print("\nFinal Generation")

# Normalize every population
for i in range(0,len(new_population)):
    new_population[i] /= new_population[i].sum()

print("Population : \n", np.round(new_population,3))

fitness = np.empty((sol_per_pop,), dtype=object)
for pop in range(sol_per_pop):
    Simulation.module_weights = new_population[pop]
    fitness[pop] = mainSARSA()
    resetInits()

print("Fitness     : ", fitness)

# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.min(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()