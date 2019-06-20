import numpy as np
import random

##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:
    # Define Which Test Arena Being Used
    # 0 = Custom
    # 1 = Urban Terrain
    # 2 = Open Terrain
    Arena = 0

    if (Arena == 0): # Custom Terrain. Edit These Ones 
        
        num_agents = 3                 # Number of agents to simulate
        num_episodes = 3            # Number of times to run the training scenario
        episode_length = 50            # Number of time steps in each training scenario [iterations]
        exploitation_rise_time = 0   # The amount of time over which we transition from exploration to exploitation [seconds]
        exploitation_rise_percent = 0  # The percentage of each episode over which we transition from exploration to exploitation

        # Bounds to initialize the agents inside of
        init_space = [[0,10],
                    [0,10]]

        # Bounds to simulate the agents within
        # Exiting these bounds will end the episode immediately
        search_space = [[-20,20],
                        [-20,20]]

        visualize = True            # Whether to show a plot animation of the agent positions
        load_agents = True          # Whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
        load_training_data = False  # Whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
        take_best_action = True     # Whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 


        # Activate Modules
        CohesionModule = False            # Cohesion module makes the agents stay together as a swarm
        CollisionAvoidanceModule = False  # Collision module prevents the agents from hitting each other
        OutOfBoundsModule = False         # Boundary module prevents the agents from leaving the search space
        TargetSeekingModule = False       # Target module encourages agents to travel to waypoint
        ObstacleAvoidanceModule = True  # Obstacle module prevents the agents from hitting obstacles

        # These are the weights for each module. they should sum to 1. 
        # If they don't, they will be scaled accordingly during initialization
        # Also, there should be a weight entry for each module
        module_weights = [1]

        ## Controller to be activated.
        # 0 = Steve and Bucci Controller
        # 1 = Importance Function
        # 2 = NeuralNetwork
        ControllerType = 0

        target_random = False
        #target_array = np.array([[-40,40],[20,-10],[50,50],[40,-50]]) # If target_random is False
        target_array = np.array([[10,10],[-10,-10],[-10,10],[10,-10]]) # If target_random is False
        if(target_random):
            targets = np.array([random.randint(search_space[0][0]+5, search_space[0][1]-5),
                                random.randint(search_space[1][0]+5, search_space[1][1]-5)])
        else:
            targets = target_array[0]
        
        

        # Obstacles to Avoid
        ## [x, y, width, height]
        #obstacles = np.array([[-30,-40,30,50], [10, -40, 20, 50], [-40, 10, 60, 10]])
        num_obstacles = 3
        obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)])
        for i in range(1,num_obstacles):
            temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), random.randint(1,10), random.randint(1,10)])
            obstacles = np.vstack((obstacles, temp_obstacles))

    elif (Arena == 1): # Urban Terrain

        num_agents = 9                 # number of agents to simulate
        num_episodes = 5               # number of times to run the training scenario
        episode_length = 1000          # number of time steps in each training scenario [iterations]
        exploitation_rise_time = 0     # the amount of time over which we transition from exploration to exploitation [seconds]
        exploitation_rise_percent = 0  # the percentage of each episode over which we transition from exploration to exploitation

        #bounds to initialize the agents inside of
        init_space = [[-60,-40],
                    [-60,-40]]

        #bounds to simulate the agents within
        #exiting these bounds will end the episode immediately
        search_space = [[-80,80],
                        [-80,80]]

        visualize = True            # whether to show a plot animation of the agent positions
        load_agents = True          # whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
        load_training_data = False  # whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
        take_best_action = True     # whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 


        # Activate Modules
        CohesionModule = True           # Cohesion module makes the agents stay together as a swarm
        CollisionAvoidanceModule = True # Collision module prevents the agents from hitting each other
        OutOfBoundsModule = True        # Boundary module prevents the agents from leaving the search space
        TargetSeekingModule = True      # Target module encourages agents to travel to waypoint
        ObstacleAvoidanceModule = True  # Obstacle module prevents the agents from hitting obstacles

        # These are the weights for each module. they should sum to 1. 
        # If they don't, they will be scaled accordingly during initialization
        # Also, there should be a weight entry for each module
        module_weights = [0.1, 0.2, 0.1, 0.8, 0.4]

        ## Controller to be activated.
        # 0 = Steve and Bucci Controller
        # 1 = Importance Function
        # 2 = NeuralNetwork
        ControllerType = 0

        #targets = np.array([-40,40],[20,-10],[50,50],[40,-50])
        targets = np.array([40,-50])

        # Obstacles to Avoid
        ## [x, y, width, height]
        obstacles = np.array([[-10,-60,30,15], 
                              [-45, -15, 30, 20],
                              [45, -10, 10, 30],
                              [-5, 25, 20, 30]])
        for i in range(0,10):
            temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
            obstacles = np.vstack((obstacles, temp_obstacles))

    elif (Arena == 2): # Open Terrain

        num_agents = 25                # number of agents to simulate
        num_episodes = 5               # number of times to run the training scenario
        episode_length = 1000          # number of time steps in each training scenario [iterations]
        exploitation_rise_time = 0     # the amount of time over which we transition from exploration to exploitation [seconds]
        exploitation_rise_percent = 0  # the percentage of each episode over which we transition from exploration to exploitation

        #bounds to initialize the agents inside of
        init_space = [[-40,-30],
                    [-40,-30]]

        #bounds to simulate the agents within
        #exiting these bounds will end the episode immediately
        search_space = [[-60,60],
                        [-60,60]]

        visualize = True            # whether to show a plot animation of the agent positions
        load_agents = True          # whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
        load_training_data = False  # whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
        take_best_action = True     # whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 


        # Activate Modules
        CohesionModule = True           # Cohesion module makes the agents stay together as a swarm
        CollisionAvoidanceModule = True # Collision module prevents the agents from hitting each other
        OutOfBoundsModule = True        # Boundary module prevents the agents from leaving the search space
        TargetSeekingModule = True      # Target module encourages agents to travel to waypoint
        ObstacleAvoidanceModule = True  # Obstacle module prevents the agents from hitting obstacles

        # These are the weights for each module. they should sum to 1. 
        # If they don't, they will be scaled accordingly during initialization
        # Also, there should be a weight entry for each module
        module_weights = [0.1, 0.2, 0.1, 0.8, 0.4]

        ## Controller to be activated.
        # 0 = Steve and Bucci Controller
        # 1 = Importance Function
        # 2 = NeuralNetwork
        ControllerType = 0

        #targets = np.array([-40,40])
        targets = np.array([random.randint(search_space[0][0], search_space[0][1]),
                            random.randint(search_space[1][0], search_space[1][1])])

        # Obstacles to Avoid
        ## [x, y, width, height]
        obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
        for i in range(0,25):
            temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
            obstacles = np.vstack((obstacles, temp_obstacles))


    ##########################################
    ## STORED VARIABLES - (DO NOT EDIT)
    ##########################################

    obstacle_collision_count = 0          # Number of collisions (Agent-Agent)
    obstacle_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)

    agent_collision_count = 0             # Number of collisions (Agent-Agent)
    agent_episode_collision_count = []    # Number of collisions during a single episode (Agent-Agent)

    boundary_collision_count = 0          # Number of collisions (Agent-Agent)
    boundary_episode_collision_count = [] # Number of collisions during a single episode (Agent-Agent)

    target_entries_count = 0              # Number of agents in the target region
    target_episode_entries_count = []     # Number agents in a target at the end of an episode
    target_reached_episode_end = np.zeros(num_episodes)       # Number of targets the agents arrived at in one episode
    target_agents_remaining = 0

    episode_iter_num = 0   # Track the current interation of the episode. Used with exploration/exploitation
    agents = list()        # List of agents
##############################################################################
#   Simulation Class
##############################################################################
    
