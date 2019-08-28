import numpy as np
import random
import enum

class TargetPath(enum.Enum): 
    Circle = 1  # Plot Targets in Circular Path
    Random = 2  # Plot Targets Randomly
    Planned = 3 # Plot Targets in accordance to the defined targets array

class Arena(enum.Enum): 
    Playground = 1  # Arena that can be whatever
    BigUrban = 2    # Larger Urban Arena
    SmallUrban = 3  # Smaller Urban Arena
    Open = 4        # Open Arena

class Controller(enum.Enum): 
    GreatestMass = 1  # Use Greatest Mass Controller
    Importance = 2    # Use Importance Function
    GenAlg = 3        # Use Greatest Mass but with GA to find weights

class Reward(enum.Enum): 
    Continuous = 1  # Continous reward Scheme
    Tiered = 2      # Use Importance Function
    Hybrid = 3      # Hybrid Reward Scheme

##############################################################################
#   Simulation class
##############################################################################

# A class containing high level simulation variables
class Simulation:

    ## Define Which Test Arena Being Used
    Arena = Arena.Playground

    ## Multi-Module Action Selector (MMAS) to be activated.
    ControllerType = Controller.GreatestMass

    ## Target Parameters
    TargetType = TargetPath.Random
    # TargetType = TargetPath.Circle

    ## Reward Scheme
    RewardType = Reward.Tiered

    testing = True
    continuous_steering = False

    getMetricPlots = False

    visualize = True             # Whether to show a plot animation of the agent positions
    load_agents = False          # Whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
    load_training_data = True   # Whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)
    take_best_action = True    # Whether to select next actions based on highest Q table entry or use Q table values as probabilities for each action 

    # Genetic Algoritm Parameters
    sol_per_pop = 8         # Population Size
    num_parents_mating = 4  # Mating Pool Size
    num_generations = 100   # Number of Generations

    if (Arena == Arena.Playground): # Custom Terrain. Edit These Ones 
        num_agents = 20               # Number of agents to simulate
        num_episodes = 2           # Number of times to run the training scenario
        episode_length = 50           # Number of time steps in each training scenario [iterations]
        exploitation_rise_time = 0      # The amount of time over which we transition from exploration to exploitation [seconds]

        exploitation_rise_percent = 0  # The percentage of each episode over which we transition from exploration to exploitation
        
        max_obstacle_size = 50
        obs_width = max_obstacle_size
        obs_height = max_obstacle_size
        
        # obs_width = random.randint(1,max_obstacle_size)
        # obs_height = random.randint(1,max_obstacle_size)
        
        #bounds to initialize the agents inside of
        # init_space = [[-np.round(obs_width*0.5)-4,np.round(obs_width*0.5)+4],
        #             [-np.round(obs_height*0.5)-4,np.round(obs_height*0.5)+4]]
        init_space = [[-3,3],
                    [-3,3]]

        #bounds to simulate the agents within
        #exiting these bounds will end the episode immediately
        search_space = [[-30,30],
                        [-30,30]]

        # Bounds to intilize the targets and obstacles within
        arena_space = [[-20,20],
                       [-20,20]]

        # Activate Modules
        CohesionModule = False            # Cohesion module makes the agents stay together as a swarm
        CollisionAvoidanceModule = True  # Collision module prevents the agents from hitting each other
        OutOfBoundsModule = False        # Boundary module prevents the agents from leaving the search space
        TargetSeekingModule = False      # Target module encourages agents to travel to waypoint
        ObstacleAvoidanceModule = False  # Obstacle module prevents the agents from hitting obstacles

        # These are the weights for each module. they should sum to 1. 
        # If they don't, they will be scaled accordingly during initialization
        # Also, there should be a weight entry for each module
        # module_weights = [1,0.5,10,4,10]  # TODO: only do sanity checks against this if using Steve and Bucci controller
        module_weights = [1]  # TODO: only do sanity checks against this if using Steve and Bucci controller
        module_priorities = [1, 1, 0, 1, 1]

        # Planned Target Trajectory
        if (TargetType == TargetPath.Planned):
            #target_array = np.array([[-40,40],[20,-10],[50,50],[40,-50]])
            # target_array = np.array([[-20,20],[0,-10],[20,30],[-5,-35]])
            target_array = np.array([[20,20]])
            targets = target_array[0]
            changeTargetOnArrival = False
            #target_array = np.array([[-20,20]])
            #target_array = np.array([[10,10],[-10,-10],[-10,10],[10,-10]])

        # Circular Target Trajectory
        elif (TargetType == TargetPath.Circle):
            r = 0 # Target Location Circle Radius
            n = num_episodes/360   # Number of loops to complete
            targets = np.array([r*np.cos(0), r*np.sin(0)])
            changeTargetOnArrival = False

        # Random Target Trajectory
        elif (TargetType == TargetPath.Random):
            targets = np.array([random.randint(arena_space[0][0], arena_space[0][1]),
                                random.randint(arena_space[1][0], arena_space[1][1])])
            changeTargetOnArrival = False
        
        num_obstacles = 5
        max_obstacle_size = 2
        # obstacles = np.array([[0,10,2,2], 
        #                       [10, 0, 2, 2]])

        # obstacles = np.array([[-10,-10,20,20]])
        # obstacles = np.array([[-np.round(obs_width*0.5),-np.round(obs_height*0.5),obs_width,obs_height]])

        obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,max_obstacle_size), random.randint(1,max_obstacle_size)])
        for i in range(1,num_obstacles):
            temp_obstacles = np.array([random.randint(arena_space[0][0], arena_space[0][1]),random.randint(arena_space[0][0], arena_space[0][1]), random.randint(1,max_obstacle_size), random.randint(1,max_obstacle_size)])
            obstacles = np.vstack((obstacles, temp_obstacles))
        
    if (Arena == Arena.SmallUrban): # Custom Terrain. Edit These Ones 
        num_agents = 4              # number of agents to simulate
        num_episodes = 2000               # number of times to run the training scenario
        episode_length = 400          # number of time steps in each training scenario [iterations]
        exploitation_rise_time = 0     # the amount of time over which we transition from exploration to exploitation [seconds]
        exploitation_rise_percent = 0  # the percentage of each episode over which we transition from exploration to exploitation

        #bounds to initialize the agents inside of
        # init_space = [[-35,-25],
        #             [-35,-25]]

        init_space = [[-40,-35],
                    [-10,-5]]

        #bounds to simulate the agents within
        #exiting these bounds will end tP he episode immediately
        search_space = [[-50,50],
                        [-50,50]]

        # Bounds to intilize the targets and obstacles within
        arena_space = [[-40,40],
                       [-40,40]]

        # Activate Modules
        CohesionModule = True          # Cohesion module makes the agents stay together as a swarm
        CollisionAvoidanceModule = True # Collision module prevents the agents from hitting each other
        OutOfBoundsModule = True        # Boundary module prevents the agents from leaving the search space
        TargetSeekingModule = True      # Target module encourages agents to travel to waypoint
        ObstacleAvoidanceModule = True  # Obstacle module prevents the agents from hitting obstacles

        # These are the weights for each module. they should sum to 1. 
        # If they don't, they will be scaled accordingly during initialization
        # Also, there should be a weight entry for each module
        module_weights = [1, 1, 4, 1, 10]
        #module_weights = [cohesion, collision, bounds, target, obstacle]
        # module_weights = [5.60187082e-04, 6.66628594e-01, 4.61904762e-02, 2.60290217e-01, 2.63305262e-02]
        # module_weights = [5.60187082e-04, 6.66628594e-0, 4.61904762e-02, 2.60290217e-01, 2.63305262e-0]
        module_priorities = [0, 1, 0, 0, 1]

        TargetType = TargetPath.Planned
        target_array = np.array([[-20,20],[0,-10],[20,30],[-5,-35]])
        targets = target_array[0]
        changeTargetOnArrival = True

        # Obstacles to Avoid
        ## [x, y, width, height]
        num_obstacles = 5
        max_obstacle_size = 1
        obstacles = np.array([[-30,0,20,5], 
                              [-30, -10, 10, 10],
                              [10, 0, 10, 20],
                              [-5, -25, 35, 5]])

        if (ControllerType != Controller.GenAlg):
            for i in range(0,num_obstacles):
                temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
                obstacles = np.vstack((obstacles, temp_obstacles))

    if (Arena == Arena.BigUrban): # Custom Terrain. Edit These Ones 
        num_agents = 10                # number of agents to simulate
        num_episodes = 10               # number of times to run the training scenario
        episode_length = 1000          # number of time steps in each training scenario [iterations]
        exploitation_rise_time = 0     # the amount of time over which we transition from exploration to exploitation [seconds]
        exploitation_rise_percent = 0  # the percentage of each episode over which we transition from exploration to exploitation

        #bounds to initialize the agents inside of
        init_space = [[-60,-40],
                    [-60,-40]]

        #bounds to simulate the agents within
        #exiting these bounds will end the episode immediately
        search_space = [[-100,100],
                        [-100,100]]

        # Bounds to intilize the targets and obstacles within
        arena_space = [[-60,60],
                       [-60,60]]

        # Activate Modules
        CohesionModule = True          # Cohesion module makes the agents stay together as a swarm
        CollisionAvoidanceModule = True # Collision module prevents the agents from hitting each other
        OutOfBoundsModule = True        # Boundary module prevents the agents from leaving the search space
        TargetSeekingModule = True      # Target module encourages agents to travel to waypoint
        ObstacleAvoidanceModule = True  # Obstacle module prevents the agents from hitting obstacles

        # These are the weights for each module. they should sum to 1. 
        # If they don't, they will be scaled accordingly during initialization
        # Also, there should be a weight entry for each module
        #module_weights = [0.1, 0.2, 0.1, 0.8, 0.4]
        module_weights = [4.95231076e-06, 5.98819376e-01, 2.59955191e-09, 4.01175657e-01, 1.21312422e-08]
        module_priorities = [0, 1, 0, 0, 1]



        TargetType = TargetPath.Planned
        target_array = np.array([[-40,40],[20,-10],[50,50],[40,-50]])
        #target_array = np.array([[100,0]])
        targets = target_array[0]
        changeTargetOnArrival = True

        # Obstacles to Avoid
        ## [x, y, width, height]
        num_obstacles = 25
        max_obstacle_size = 5

        obstacles = np.array([[-10,-60,30,15], 
                              [-45, -15, 30, 20],
                              [45, -10, 10, 30],
                              [-5, 25, 20, 30]])
        for i in range(0,20):
            temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
            obstacles = np.vstack((obstacles, temp_obstacles))
        
    elif (Arena == Arena.Open): # Open Terrain

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

        TargetType = TargetPath.Planned
        targets = np.array([-40,40],[20,-10],[50,50],[40,-50])
        targets = target_array[0]

        # Obstacles to Avoid
        ## [x, y, width, height]
        obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
        for i in range(0,25):
            temp_obstacles = np.array([random.randint(search_space[0][0], search_space[0][1]),random.randint(search_space[0][0], search_space[0][1]), 1, 1])
            obstacles = np.vstack((obstacles, temp_obstacles))


    ##########################################
    ## STORED VARIABLES - (DO NOT EDIT)
    ##########################################

    cohesionDist = []                     # Distance of all agents to the centroid
    cohesionFactor = []                   # Mean distance of all agents to the centroid per episode

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
    target_histogram_data = [] # Data about what iteration a target was reached.

    episode_iter_num = 0   # Track the current interation of the episode. Used with exploration/exploitation
    agents = list()        # List of agents

##############################################################################
#   Simulation Class
##############################################################################
