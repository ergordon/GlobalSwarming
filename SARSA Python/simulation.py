import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    num_agents = 5 #number of agents to simulate
    num_episodes = 1000 #number of times to run the training scenario
    episode_length = 200 #number of time steps in each training scenario [iterations]
    exploitation_rise_time = 300 #the amount of time over which we transition from exploration to exploitation [seconds]

    #bounds to initialize the agents inside of
    init_space = [[0,10],
                [0,10]]

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-50,50],
                    [-50,50]]

    ThreeTwoOne = False
    visualize = ThreeTwoOne  #whether to show a plot animation of the agent positions

    load_agents = ThreeTwoOne #whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
    load_training_data = False #whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
    
    #TODO think of better name for this
    take_best_action = False #whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 

    #targets = np.array([-40,40])
    targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                        random.randint(search_space[1][0], search_space[1][1])])

    # Obstacles to Avoid
    ## [x, y, width, height]
    #obstacles = np.array([[-40,-40,30,50], [10, -40, 20, 50], [-40, 10, 60, 10]])
    #obstacles = np.array([[-40,-40,30,50]])
    obstacles = np.array([[-40,-40,10,10], 
                          [-15, -20, 10, 10], 
                          [15, 10, 10, 10],
                          [-5, 30, 10, 10],
                          [15, -20, 10, 10], 
                          [10, -40, 10, 10],
                          [-40, 10, 30, 10]])
##############################################################################
#   Simulation Class
##############################################################################
    
