import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    num_agents = 3 #number of agents to simulate
    
    num_episodes = 2 #number of times to run the training scenario
    episode_length = 100 #number of time steps in each training scenario [iterations]
    exploitation_rise_time = 400 #the amount of time over which we transition from exploration to exploitation [seconds]
            #TODO change to exploitation rise percentage and make it explore for the 1st percent of every episode
    exploitation_rise_percent = 20 #the percentage of each episode over which we transition from exploration to exploitation

    #bounds to initialize the agents inside of
    init_space = [[-10,10],
                [-10,10]]

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-25,25],
                    [-25,25]]
    # search_space = [[-10,10],
    #                 [-10,10]]


    ThreeTwoOne = True
    visualize = ThreeTwoOne  #whether to show a plot animation of the agent positions

    load_agents = ThreeTwoOne #whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
    load_training_data = False #whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
    #TODO fix load_training_data, gave permission error when saving to disk at the end


    #TODO think of better name for this
    take_best_action = ThreeTwoOne #whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 

    #targets = np.array([-40,40])
    targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                        random.randint(search_space[1][0], search_space[1][1])])

    # Obstacles to Avoid
    ## [x, y, width, height]
    #obstacles = np.array([[-40,-40,30,50], [10, -40, 20, 50], [-40, 10, 60, 10]])
    #obstacles = np.array([[-40,-40,30,50]])
    #obstacles = np.array([[-40,-40,10,10], 
    #                      [-15, -20, 10, 10], 
    #                      [15, 10, 10, 10],
    #                      [-5, 30, 10, 10],
    #                      [15, -20, 10, 10], 
    #                      [10, -40, 10, 10],
    #                      [-40, 10, 30, 10]])

    #TODO the obstacles are created here, but reinitialized in the main loop so if you want to change it, you have to change it both places.
    #so think of a better implementation
    #TODO create obstacle objects instead of using an array.
    obstacles = np.array([[random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)]])


    #DO NOT EDIT VARIABLES BELOW THIS COMMENT

    episode_iter_num = 0 #DO NOT EDIT. variable used only to track the current interation of the episode. used with exploration/exploitation



##############################################################################
#   Simulation Class
##############################################################################
    
