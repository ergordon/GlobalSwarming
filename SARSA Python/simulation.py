import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-50,50],
                    [-50,50]]

    num_agents = 5 #number of agents to simulate
    num_episodes = 1000 #number of times to run the training scenario
    episode_length = 200 #number of time steps in each training scenario
    exploitation_rise_time = 50 #the amount of time over which we transition from exploration to exploitation

    #bounds to initialize the agents inside of
    init_space = [[0,10],
                [0,10]]

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-50,50],
                    [-50,50]]

    visualize = False   #whether to show a plot animation of the agent positions

    #targets = np.array(np.random.random_integers(50, size=(1,2)))
    targets = np.array([random.randint(-50, 50),random.randint(-50, 50)])

##############################################################################
#   Simulation Class
##############################################################################
    