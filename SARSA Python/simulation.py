import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    num_agents = 6 #number of agents to simulate
    num_episodes = 200 #number of times to run the training scenario
    episode_length = 50 #number of time steps in each training scenario [iterations]
    exploitation_rise_time = 50 #the amount of time over which we transition from exploration to exploitation [seconds]

    #bounds to initialize the agents inside of
    init_space = [[0,10],
                [0,10]]

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-20,20],
                    [-20,20]]

    visualize = True  #whether to show a plot animation of the agent positions

    load_agents = True  #whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
    load_training_data = False #whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
    
    #TODO think of better name for this
    take_best_action = True #whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 

    targets = np.array([-40,40])
    # targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
    #                     random.randint(search_space[1][0], search_space[1][1])])


##############################################################################
#   Simulation Class
##############################################################################
    
