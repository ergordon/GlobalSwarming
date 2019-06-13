import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    num_agents = 4 #number of agents to simulate
    num_episodes = 10 #number of times to run the training scenario
    episode_length = 100 #number of time steps in each training scenario [iterations]
    exploitation_rise_time = 0 #the amount of time over which we transition from exploration to exploitation [seconds]
            #TODO change to exploitation rise percentage and make it explore for the 1st percent of every episode
    exploitation_rise_percent = 0 #the percentage of each episode over which we transition from exploration to exploitation

    #bounds to initialize the agents inside of
    init_space = [[0,10],
                [0,10]]

    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    search_space = [[-30,30],
                    [-30,30]]

    ThreeTwoOne = True
    visualize = ThreeTwoOne  #whether to show a plot animation of the agent positions

    load_agents = ThreeTwoOne #whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
    load_training_data = False #whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
    
    #TODO think of better name for this
    take_best_action = False #whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 


    # Activate Modules
    CohesionModule = True           # Cohesion module makes the agents stay together as a swarm
    CollisionAvoidanceModule = True # Collision module prevents the agents from hitting each other
    OutOfBoundsModule = True        # Boundary module prevents the agents from leaving the search space
    TargetSeekingModule = True      # Target module encourages agents to travel to waypoint
    ObstacleAvoidanceModule = True  # Obstacle module prevents the agents from hitting obstacles

    #these are the weights for each module. they should sum to 1. 
    #If they don't, they will be scaled accordingly during initialization
    #also, there should be a weight entry for each module
    module_weights = [0.1, 0.2, 0.1, 0.8, 0.4]#[0.0001,0.99] 
    #module_weights = [0.01, 0.02, 0.01, 0.8, 0.04]#[0.0001,0.99] 
    #module_weights = [1]

    # Controller to be activated.
    # 0 = Steve and Bucci Controller
    # 1 = Importance Function
    # 2 = NeuralNetwork
    ControllerType = 0

    #targets = np.array([-40,40])
    targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                        random.randint(search_space[1][0], search_space[1][1])])

    # Obstacles to Avoid
    ## [x, y, width, height]
    #obstacles = np.array([[-30,-40,30,50], [10, -40, 20, 50], [-40, 10, 60, 10]])
    #obstacles = np.array([[-40,-40,30,50]])
    #obstacles = np.array([[-40,-40,10,10], 
    #                      [-15, -20, 10, 10], 
    #                      [15, 10, 10, 10],
    #                      [-5, 30, 10, 10],
    #                      [15, -20, 10, 10], 
    #                      [10, -40, 10, 10],
    #                      [-40, 10, 30, 10]])

    obstacles = np.array([[random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10], 
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10],
                          [random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 10, 10]])


    obstacle_collision_count = 0        # Number of collisions (Agent-Agent)
    obstacle_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)

    agent_collision_count = 0        # Number of collisions (Agent-Agent)
    agent_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)

    boundary_collision_count = 0        # Number of collisions (Agent-Agent)
    boundary_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)
##############################################################################
#   Simulation Class
##############################################################################
    
