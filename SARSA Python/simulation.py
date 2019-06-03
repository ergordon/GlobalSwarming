
##############################################################################
#   Simulation class
##############################################################################

#A class containing high level simulation variables
class Simulation:

    num_agents = 5 #number of agents to simulate

    num_episodes = 200 #number of times to run the training scenario
    episode_length = 100 #number of timesteps in each traning scenario


    #bounds to simulate the agents within
    #exiting these bounds will end the episode immediately
    #[[x1,x2],
    #[y1,y2]]
    search_space = [[-50,50],
                    [-50,50]]

    visualize = False    #whether to show a plot animation of the agent positions

    load_agents = False  #whether to load the agents.pkl file (loads agents exactly as they upon completion of training)
    load_training_data = False #whether to load the agent training data (loads q tables and states into the modules that exist in the agent initialization function)

    exploitation_rise_time = 30
    

##############################################################################
#   Simulation Class
##############################################################################