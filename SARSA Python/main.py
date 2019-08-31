from agent import Agent
from simulation import Simulation
from simulation import TargetPath
from simulation import Arena
from simulation import Controller
import ga
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
ap.add_argument("--trainingPath", type=str, default="TrainedData", required=False,
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
    
    # max_obstacle_size = 50
    # obs_width = max_obstacle_size
    # obs_height = max_obstacle_size
    # # obs_width = random.randint(1,max_obstacle_size)
    # # obs_height = random.randint(1,max_obstacle_size)
    # #bounds to initialize the agents inside of
    # # Simulation.init_space = [[-np.round(obs_width*0.5)-4,np.round(obs_width*0.5)+4],
    # #             [-np.round(obs_height*0.5)-4,np.round(obs_height*0.5)+4]]
    # Simulation.obstacles = np.array([[-np.round(obs_width*0.5),-np.round(obs_height*0.5),obs_width,obs_height]])



    # Reinitialize Setting Parameters
    if (Simulation.Arena == Arena.Playground):
        Simulation.obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,Simulation.max_obstacle_size), random.randint(1,Simulation.max_obstacle_size)])
        for i in range(1,Simulation.num_obstacles):
            temp_obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,Simulation.max_obstacle_size), random.randint(1,Simulation.max_obstacle_size)])
            Simulation.obstacles = np.vstack((Simulation.obstacles, temp_obstacles))
    
    


    # Initialize agent parameters
    for i in range(0,len(agents)):
        init_space = Simulation.init_space

        if (Simulation.ControllerType != Controller.GenAlg):
            agents[i].position = np.array([random.randint(init_space[0][0], init_space[0][1]),random.randint(init_space[1][0], init_space[1][1])], dtype='f')
        else:
            inits = ([init_space[0][0],init_space[1][0]],[init_space[0][0],init_space[1][1]], 
                     [init_space[0][1],init_space[1][0]],[init_space[0][1],init_space[1][1]])
            agents[i].position = np.array(inits[i], dtype='f')

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
#   Main Controller Operations
##############################################################################
def mainSARSA(simName,desc,trainingPath):

    ##############################################################################
    #   Simulation Variables
    ##############################################################################

    agent_rewards = np.array([])   # matrix containing total reward values for each agent for each episode

    
    # Make new Directories
    filename = simName
    path = filename
    path_agentReward = path+"/AgentRewards"
    path_animation = path+"/Animations"
    path_rewardPlot = path+"/RewardPlots"
    path_collision = path+"/Collisions"
    path_targetPlot = path+"/TargetPlots"
    try:  
        os.mkdir(path)
    except OSError:
        pass

    if (Simulation.ControllerType != Controller.GenAlg):
        try:  
            os.mkdir(path_agentReward)
            os.mkdir(path_animation)
            os.mkdir(path_rewardPlot)
            os.mkdir(path_collision)
            os.mkdir(path_targetPlot)
        except OSError:
            pass


    ##############################################################################
    #   Save Simulation Configuration Settings
    ##############################################################################
    # Store the program start time so we can calculate how long it took for the code to execute
    start_time = time.time()
    timestr = time.strftime("%m%d-%H%M")
    if (Simulation.ControllerType != Controller.GenAlg):
        # Save Configuration to a test file
        if(not Simulation.visualize):
            if os.path.exists(os.path.join(path ,'Simulation_Configuration.txt')):
                file = open(os.path.join(path ,'Simulation_Configuration.txt'),'a') 
            else:    
                file = open(os.path.join(path ,'Simulation_Configuration.txt'),'w') 
            file.write(" \n \n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
            file.write(simName+" -- "+ timestr +" \n")
            file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n \n")
            file.write(str(args["description"]) + "\n \n")
            
            if Simulation.ControllerType == Controller.GreatestMass:
                file.write("Active Controller: Steve and Bucci Controller \n \n")
            elif Simulation.ControllerType == Controller.Importance:
                file.write("Active Controller: Importance Function \n \n")
            else:
                file.write("Active Controller: Neural Network \n \n")

            file.write("~~~~ ARENA PARAMETERS ~~~~ \n")
            file.write("num_agents:    " + str(Simulation.num_agents)+"\n")
            file.write("num_episodes:  " + str(Simulation.num_episodes)+"\n")
            file.write("episode_length:  " + str(Simulation.episode_length)+"\n")
            file.write("exploitation_rise_time:  " + str(Simulation.exploitation_rise_time )+"\n")
            file.write("exploitation_rise_percent:  " + str(Simulation.exploitation_rise_percent )+"\n")
            file.write("init_space:  " + str(Simulation.init_space)+"\n")
            file.write("search_space:  " + str(Simulation.search_space)+"\n \n")

            file.write("~~~~ ACTIVE MODULES ~~~~ \n")
            file.write("Cohesion Module ------- "+str(Simulation.CohesionModule) + "\n")
            file.write("Collision Module ------ "+str(Simulation.CollisionAvoidanceModule) + "\n")
            file.write("Out of Bounds Module -- "+str(Simulation.OutOfBoundsModule) + "\n")
            file.write("Target Seek Module ---- "+str(Simulation.TargetSeekingModule) + "\n")
            file.write("Obstacle Module ------- "+str(Simulation.ObstacleAvoidanceModule) + "\n \n")
            file.write("Module Weights: " + str(Simulation.module_weights) + "\n \n")

            if (Simulation.TargetSeekingModule == True):
                file.write("~~~~ TARGET PARAMETERS ~~~~ \n")
                if (Simulation.TargetType == TargetPath.Planned):
                    file.write("Planned Target Location \n")
                    file.write("Change Target on Arrival = "+ str(Simulation.changeTargetOnArrival)+" \n")
                    file.write("Target Array:  \n")
                    file.write(str(Simulation.target_array)+"\n \n")

                # Circular Target Trajectory
                elif (Simulation.TargetType == TargetPath.Circle):
                    file.write("Circular Target Path \n")
                    file.write("Change Target on Arrival = "+ str(Simulation.changeTargetOnArrival)+" \n")
                    file.write("Target Path Radius:  " + str(Simulation.r)+"\n")
                    file.write("Number of Loops:  " + str(Simulation.n)+"\n \n")

                # Random Target Trajectory
                elif (Simulation.TargetType == TargetPath.Random):
                    file.write("Random Target Location \n")
                    file.write("Change Target on Arrival = "+ str(Simulation.changeTargetOnArrival)+" \n \n")

            if (Simulation.ObstacleAvoidanceModule == True):
                file.write("~~~~ OBSTACLE PARAMETERS ~~~~ \n")
                file.write("Number of Obstacles: "+str(Simulation.num_obstacles)+"\n")
                file.write("Max Obstacle Size: "+str(Simulation.max_obstacle_size)+"\n \n")

            file.close() 

    ##############################################################################
    #   Initialization
    ##############################################################################
    if (Simulation.ControllerType != Controller.GenAlg):
        print('initializing agents')

    initialized = False
    # Check if a file containing a list of agents already exits
    if (Simulation.ControllerType != Controller.GenAlg):
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
            if (Simulation.ControllerType != Controller.GenAlg):
                init_space = Simulation.init_space
                position = np.array([random.randint(init_space[0][0], init_space[0][1]),random.randint(init_space[1][0], init_space[1][1])], dtype='f')
                # position = np.array([2*i,2*i], dtype='f')
                Simulation.agents.append(Agent(position))
            else:
                init_space = Simulation.init_space

                inits = ([init_space[0][0],init_space[1][0]],[init_space[0][0],init_space[1][1]], 
                         [init_space[0][1],init_space[1][0]],[init_space[0][1],init_space[1][1]])

                position = np.array(inits[i], dtype='f')
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
                training_filename = trainingPath +'/'+ Simulation.agents[0].modules[i].__class__.__name__ + '_training_data.pkl'

                if os.path.isfile(training_filename):
                    if (Simulation.ControllerType != Controller.GenAlg):
                        print("Q learning data found, loading it now")  

                    with open(training_filename, 'rb') as f:
                        [module_name, data, updates, epsilon] = pickle.load(f)

                    if Simulation.agents[0].modules[i].collapsable_Q:
                        for agnt in Simulation.agents:
                            for Q in agnt.modules[i].Q:
                                Q.q_data = cp.copy(data[0])
                                Q.q_updates = cp.copy(updates[0])
                                Q.q_epsilon = cp.copy(epsilon[0])
                    else:
                        for agnt in Simulation.agents:
                            for q in range(0,len(agnt.modules[i].Q)):
                                agnt.modules[i].Q[q].q_data = cp.copy(data[q])
                                agnt.modules[i].Q[q].q_updates = cp.copy(updates[q])
                                agnt.modules[i].Q[q].q_epsilon = cp.copy(epsilon[q])
                        
                        



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

    if (Simulation.ControllerType != Controller.GenAlg):
        print('beginning training')
    for e in range(0,Simulation.num_episodes):
        Simulation.episode_iter_num = 0
        if (Simulation.ControllerType != Controller.GenAlg):
            print("beginning episode #" + str(e+1))

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
            for a in range(0, len(Simulation.agents)):
                agnt = Simulation.agents[a]

                # Take the action determined in the last step
                #  Update agent positions on plots
                # print('state is', agnt.modules[0].state)
                agnt.take_action(agnt.modules[0].action)
                # print('taking action ', agnt.modules[0].action)

                # Check if any agent went out of search space.
                #  Terminate episode if so
                if not (checkInBounds(agnt.position,Simulation.search_space)):
                    if (Simulation.ControllerType != Controller.GenAlg):
                        print("agent left search space, ending episode")
                    Simulation.boundary_collision_count = Simulation.boundary_collision_count + 1
                    agent_out_of_bounds = True

            # print('update state prime and select next action')
            # for agnt in Simulation.agents:
            # for a in range(0, len(Simulation.agents)):
            #     agnt = Simulation.agents[a]

                for mod in agnt.modules:
                    # Find what the state (state_prime) would be if that action were taken
                    mod.update_state_prime()
                    mod.check_state_transition()
                    # print('state prime is ', mod.state_prime)

                # Select the next action (action_prime) for the agent to take 
                # print('~~~~~~~~~~~~agent ' + str(a) + ', select next action~~~~~~~~~~~~~~~~~~')
                # if e == 175:
                #     print('agent action: ', agnt.modules[0].action)
                agnt.select_next_action()
                # print('next action is ', agnt.modules[0].action_prime)

            # print('instant and total reward, update q, action == action prime, state == state prime')
            # for agnt in Simulation.agents:
                for mod in agnt.modules:
                    if (Simulation.ControllerType != Controller.GenAlg):
                        
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
                    mod.auxiliary_functions()

                    # Prepare for next iteration
                    mod.action = cp.copy(mod.action_prime)
                    mod.state  = np.copy(mod.state_prime)

            # Plotting for visualization
            if(Simulation.visualize):
                plt.grid(linestyle='--', linewidth='0.5', color='grey')
                plt.text(Simulation.search_space[0][0], Simulation.search_space[1][1]+1, ('Episode '+str(e)+' Iteration '+str(t)), dict(size=8))
                # for agnt in Simulation.agents:
                for a in range(0, len(Simulation.agents)):
                    agnt = Simulation.agents[a]
                    plt.plot(agnt.position[0],agnt.position[1],'ro')
                    plt.text(agnt.position[0],agnt.position[1],str(a))

                    plt.axis(axis_bounds)
                    
                    for mod in agnt.modules:
                        mod.visualize()
                
                if (t%5 == 0):
                    # Convert the figure into an array and append it to images array        
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    images.append(image)
                
                plt.pause(1/frame_rate)
                plt.clf()
                plt.cla()

            # Criteria for ending the episode early.
            if(agent_out_of_bounds):
                break  

            # # print('agents take actions')
            # for agnt in Simulation.agents:

            #     # Take the action determined in the last step
            #     #  Update agent positions on plots
            #     # print('state is', agnt.modules[0].state)
            #     agnt.take_action(agnt.modules[0].action)
            #     # print('taking action ', agnt.modules[0].action)

            #     # Check if any agent went out of search space.
            #     #  Terminate episode if so
            #     if not (checkInBounds(agnt.position,Simulation.search_space)):
            #         if (Simulation.ControllerType != Controller.GenAlg):
            #             print("agent left search space, ending episode")
            #         Simulation.boundary_collision_count = Simulation.boundary_collision_count + 1
            #         agent_out_of_bounds = True

            # if(Simulation.visualize):
            #     plt.grid(linestyle='--', linewidth='0.5', color='grey')
            #     plt.text(Simulation.search_space[0][0], Simulation.search_space[1][1]+1, ('Episode '+str(e)+' Iteration '+str(t)), dict(size=8))
            #     # for agnt in Simulation.agents:
            #     for a in range(0, len(Simulation.agents)):
            #         agnt = Simulation.agents[a]
            #         plt.plot(agnt.position[0],agnt.position[1],'ro')
            #         plt.text(agnt.position[0],agnt.position[1],str(a))

            #         plt.axis(axis_bounds)
                    
            #         for mod in agnt.modules:
            #             mod.visualize()
                
            #     if (t%5 == 0):
            #         # Convert the figure into an array and append it to images array        
            #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            #         image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #         images.append(image)

            # # print('update state prime and select next action')
            # # for agnt in Simulation.agents:
            # for a in range(0, len(Simulation.agents)):
            #     agnt = Simulation.agents[a]

            #     for mod in agnt.modules:
            #         # Find what the state (state_prime) would be if that action were taken
            #         mod.update_state_prime()
            #         mod.check_state_transition()
            #         # print('state prime is ', mod.state_prime)

            #     # Select the next action (action_prime) for the agent to take 
            #     # print('~~~~~~~~~~~~agent ' + str(a) + ', select next action~~~~~~~~~~~~~~~~~~')
            #     agnt.select_next_action()
            #     # print('next action is ', agnt.modules[0].action_prime)

            # # print('instant and total reward, update q, action == action prime, state == state prime')
            # for agnt in Simulation.agents:
            #     for mod in agnt.modules:
            #         if (Simulation.ControllerType != Controller.GenAlg):
                        
            #             # Determine the reward for executing the action (not prime) in the state (not prime)
            #             #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
            #             mod.update_instant_reward()
            #             # print('instant reward is ', mod.instant_reward[0])

            #             # Add the reward for this action to the total reward earned by the agent 
            #             mod.update_total_reward()
                        
            #             # Update the Q table
            #             mod.update_q()

            #         # Run additional functions specific to each module
            #         # For example, the collision module uses this to track collisions with other agents 
            #         mod.auxiliary_functions()

            #         # Prepare for next iteration
            #         mod.action = cp.copy(mod.action_prime)
            #         mod.state  = np.copy(mod.state_prime)

            # # Plotting for visualization
            # if(Simulation.visualize):
            #     plt.pause(1/frame_rate)
            #     plt.clf()
            #     plt.cla()

            # # Criteria for ending the episode early.
            # if(agent_out_of_bounds):
            #     break    
        
        if (Simulation.ControllerType != Controller.GenAlg):
            # Store the total reward for each agent at the end of each episode for algorithm performance analysis
            episode_rewards = np.zeros(len(Simulation.agents)) 
            for a in range(0,len(Simulation.agents)):
                episode_rewards[a] = cp.copy(Simulation.agents[a].total_reward)

            if agent_rewards.size == 0:
                agent_rewards = np.array([cp.copy(episode_rewards)])
            else:
                agent_rewards = np.vstack([agent_rewards,episode_rewards])

        Simulation.cohesionFactor.append(np.mean(Simulation.cohesionDist))

        # Reset the agents (except for the Q tables and Q states) to start fresh for the next episode         
        ReinitializeAgents(Simulation.agents,Simulation.init_space)
        
        for agnt in Simulation.agents:
            for mod in agnt.modules:
                mod.reset_init(e)

        if (e%100 == 0 and Simulation.ControllerType != Controller.GenAlg):
            # There are occasional permission errors, this block will keep retrying until the dump succeeds
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

    if (Simulation.ControllerType != Controller.GenAlg):      
        print('Training complete')

        # Store the program end time so we can calculate how long it took for the code to execute
        end_time = time.time() 
        print('Program execution time:')
        print(str(end_time-start_time)+" seconds")
        print(str((end_time-start_time)/60)+" minutes")
        print(str((end_time-start_time)/3600)+" hours")

    ##############################################################################
    #   Data Storage
    ##############################################################################
    if (Simulation.ControllerType != Controller.GenAlg):    
        # There are occasional permission errors, this block will keep retrying until the dump succeeds
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

                if dump_attempts >= max_dump_attempts:
                    print('******PERMISSION ERROR, COULD NOT DUMP AGENTS TO DISK********')
        
        # Export the visualizer as a *.gif
        if(Simulation.visualize):
            fps = 10
            kwargs_write = {'fps':fps, 'quantizer':'nq'}
            imageio.mimsave(os.path.join(path_animation, timestr+"_Animation.gif"), images, fps=fps)

        iterations = np.arange(Simulation.num_episodes)
        total_collisions = np.sum([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count], axis=0)

        if (Simulation.getMetricPlots):
            # Store the iterations and total rewards for each agent for each episode
            agent_reward_filename = path_agentReward+'/'+timestr+'_agent_rewards.pkl'

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
                except Exception as e:
                    pe = True
                    dump_attempts = dump_attempts + 1
                
                    print(e)
                    print('permission error while saving to disk, retrying...')
                    time.sleep(0.5)

                    if dump_attempts >= max_dump_attempts:
                        print('******PERMISSION ERROR, COULD NOT DUMP AGENT REWARDS TO DISK********')

        # Iterations-Reward Plot
        plt.close()
        for i in range(0,Simulation.num_agents):
            plt.plot(iterations,agent_rewards[:,i])
        plt.xlabel("Iterations")
        plt.ylabel("Reward Value")
        plt.title('Iterations V. Reward')
        plt.savefig(os.path.join(path_rewardPlot, timestr+"_IterationsVReward.jpeg") , orientation='landscape', quality=95)

        if (Simulation.getMetricPlots):
            #Collision Box and Whisker Plot
            fig1, ax1 = plt.subplots()
            ax1.set_title('Collision Tracker')
            ax1.boxplot([Simulation.agent_episode_collision_count, Simulation.obstacle_episode_collision_count, Simulation.boundary_episode_collision_count])
            plt.xlabel("Collision Type")
            plt.ylabel("Collisions")
            ax1.set_xticklabels(['Agent Collisions', 'Obstacle Collisions', 'Boundary Collisions'])

            fig1.savefig(os.path.join(path_collision, timestr+"_Collisions.jpeg") , orientation='landscape', quality=95)

            # Iterations-Targets Entered Plot
            fig2, ax2 = plt.subplots()
            plt.plot(iterations,Simulation.target_reached_episode_end)
            plt.xlabel("Iterations")
            plt.ylabel("Targets Reached")
            plt.title('Iterations V. Targets Reached')

            plt.savefig(os.path.join(path_targetPlot, timestr+"_TargetsReached.jpeg") , orientation='landscape', quality=95)


            # Box Histograms
            f, axarr = plt.subplots(2, 2)
            axarr[0, 0].set_title('Target 1')
            axarr[0, 1].set_title('Target 2')
            axarr[1, 0].set_title('Target 3')
            axarr[1, 1].set_title('Target 4')
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

            num_bins = 40
            bin = Simulation.episode_length/num_bins
            bins = []
            for i in range(0,num_bins):
                bins.append(i*bin)
            axarr[0,0].hist(temp1,bins)
            axarr[0,1].hist(temp2,bins)
            axarr[1,0].hist(temp3,bins)
            axarr[1,1].hist(temp4,bins)
            #hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)[source]
            plt.savefig(os.path.join(path_targetPlot, timestr+"_TargetsIteration.jpeg") , orientation='landscape', quality=95)



        plt.show()

        # Append Results Data to Simulation.txt
        if(not Simulation.visualize):
            file = open(os.path.join(path ,'Simulation_Configuration.txt'),'a')
            file.write("~~~~ RESULTS ~~~~ \n")
            file.write("Program Execution Time: "+str((end_time-start_time)/60)+" minutes \n")
            file.write("Mean Episode Agent-Agent Collisions: "+str(np.mean(Simulation.agent_episode_collision_count))+"\n")
            file.write("Mean Episode Agent-Obstacle Collisions: "+str(np.mean(Simulation.obstacle_episode_collision_count))+"\n")
            file.write("Mean Episode Agent-Boundary Collisions: "+str(np.mean(Simulation.boundary_episode_collision_count))+"\n")
            file.write("Mean Target 1 Iter: "+str(np.mean(temp1))+"\n")
            file.write("Mean Target 2 Iter: "+str(np.mean(temp2))+"\n")
            file.write("Mean Target 3 Iter: "+str(np.mean(temp3))+"\n")
            file.write("Mean Target 4 Iter: "+str(np.mean(temp4))+"\n")

    elif (Simulation.ControllerType == Controller.GenAlg):
        iterations = np.arange(Simulation.num_episodes)
        col1 = np.mean(Simulation.agent_episode_collision_count)
        col2 = np.mean(Simulation.obstacle_episode_collision_count)
        col3 = np.mean(Simulation.boundary_episode_collision_count)*1000
        total_collisions = np.sum([col1,col2,col3])

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
        
        CF = np.mean(Simulation.cohesionFactor)

        return temp1 + temp2 + temp3 + temp4 + total_collisions + CF
    
##############################################################################
#   Reset Simulation Parameters for Continuose Testing
##############################################################################
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


simName = str(args["simName"])
description = str(args["description"])
trainingPath = str(args["trainingPath"])

if (Simulation.ControllerType == Controller.GreatestMass or Simulation.ControllerType == Controller.Importance):
    mainSARSA(simName, description, trainingPath)

elif (Simulation.ControllerType == Controller.GenAlg):
    ##############################################################################
    #   Genetic Algorithm Module-Weight Selection
    ##############################################################################
    np.set_printoptions(precision=2) # Set the precision of floats on print screen


    pop_file = simName+'/population.pkl' # Save Latest Population

    ## GA Parameters
    num_weights = 5                                     # Number of the weights we are looking to optimize.
    sol_per_pop = Simulation.sol_per_pop                # Population Size
    num_parents_mating = Simulation.num_parents_mating  # Mating Pool Size
    num_generations = Simulation.num_generations        # Number of Generations

    best_outputs = []

    # Defining the population size.
    pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

    #Creating the initial population.
    if os.path.exists(pop_file):
        with open(pop_file, 'rb') as f:
            new_population = pickle.load(f)   
    else:
        new_population = np.random.uniform(low=0, high=1e10, size=pop_size)
        new_population[0] = [0.11, 0.18, 0.03, 0.53, 0.14]
        new_population[1] = [0.25, 0.1,  0.07, 0.27, 0.31]
        new_population[2] =  [4.95231076e-06, 5.98819376e-01, 2.59955191e-09, 4.01175657e-01, 1.21312422e-08]
        # new_population[3] = [0.21, 0.57, 0.03, 0.08, 0.12]

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
            fitness[pop] = mainSARSA(simName, description, trainingPath)
            print("Population ",pop," Tested : Fitness ", fitness[pop])
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

        # Save new population
        with open(pop_file,'wb') as g:
            pickle.dump(new_population,g) 

    print("\nFinal Generation")

    # Normalize every population
    for i in range(0,len(new_population)):
        new_population[i] /= new_population[i].sum()

    print("Population : \n", np.round(new_population,3))

    fitness = np.empty((sol_per_pop,), dtype=object)
    for pop in range(sol_per_pop):
        Simulation.module_weights = new_population[pop]
        fitness[pop] = mainSARSA(simName, description, trainingPath)
        resetInits()

    print("Fitness     : ", fitness)

    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(fitness == np.min(fitness))

    print("Best solution : ", new_population[best_match_idx, :])
    print("Best solution fitness : ", fitness[best_match_idx])

    f1 = plt.figure()
    plt.plot(best_outputs)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")

    f1.savefig(os.path.join(simName,"GARewards.jpeg"), orientation='landscape', quality=95)
    plt.show()