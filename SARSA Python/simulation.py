import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    num_agents = 1 #number of agents to simulate
    num_episodes = 10000 #number of times to run the training scenario
    episode_length = 100 #number of time steps in each training scenario [iterations]
    exploitation_rise_time = 0 #the amount of time over which we transition from exploration to exploitation [seconds]

    #bounds to initialize the agents inside of
    init_space = [[0,10],
                [0,10]]

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-50,50],
                    [-50,50]]

    visualize = False  #whether to show a plot animation of the agent positions

    #targets = np.array(np.random.random_integers(50, size=(1,2)))
    targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                        random.randint(search_space[1][0], search_space[1][1])])

    print(targets)
##############################################################################
#   Simulation Class
##############################################################################
    