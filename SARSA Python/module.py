import numpy as np

#Base class that all other modules should inherit from
class Module:
    alpha = 0.1 #learning rate (shared by all modules)

    #variables to be changed for each derived module
    gamma = 0 #discount rate
    


    #constructor
    def __init__(self,parent_agt):

        #TODO include a refernce to the parent agent referencing this object?
        self.parent_agent = parent_agt #the agent that created and is storing this module instance
        self.tracked_agents = [] #list of agents being tracked by this module 
        self.instant_reward = [] #list of instantaneous rewards earned by the agent. 

    def start_tracking(self,agt):
        self.tracked_agents.append(agt)

#module to make the agents swarm together
class CohesionModule(Module):
    
    gamma = 0.01

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [2,1,-1] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [2,4]

    def __init__(self):
        super().__init__()
        
        self.state = np.array([]) #the vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        

    def update_state(self):
        #find the centroid
        centroid = np.array(self.parent_agent.position)
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
            
        print(centroid)
        centroid = centroid / (len(self.tracked_agents)+1)
        self.state = centroid - self.parent_agent.position
        print(self.state) 
    
    def update_state_prime(self):
        #find the centroid
        centroid = self.parent_agent.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 

        centroid = centroid / (len(self.tracked_agents)+1)
        self.state_prime = centroid - self.parent_agent.position
        print(self.state_prime)

    #there is a reward for each state
    #there is only one state for the cohesion module so it is a single number 
    def update_instant_reward(self):
        #im using state_prime to calculate the reward
        #using state may be the correct thing to do, paper is ambiguous, investigate later
        
        #the state is the vector to the swarm centroid
        #use distance squared for range comparisons
        dist_squared = 0
        for i in range(0,len(self.state_prime)):
            dist_squared = dist_squared + self.state_prime[i]^2

        #loop through each range to give the appropriate reward
        rewarded = False
        for i in range(0,len(CohesionModule.ranges_squared)):
            if dist_squared <= CohesionModule.ranges_squared[i]:
                self.instant_reward = CohesionModule.rewards[i]
                rewarded = True    
                break
        
        #not in range, apply last reward (punishment)
        if rewarded == False:
            self.instant_reward = CohesionModule.rewards[-1]


#module to prevent agents from hitting each other
class CollisionModule(Module):
    
    gamma = 0

    def __init__(self):
        super().__init__()
        print(CollisionModule.gamma)

