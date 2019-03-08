import numpy as np

#Base class that all other modules should inherit from
class Module:
    alpha = 0.1 #learning rate (shared by all modules)

    #variables to be changed for each derived module
    gamma = 0 #discount rate
    


    #constructor
    def __init__(self):
        self.instant_reward = 0 #instantaneous reward earned by the agent
        
        #belong in qlearning????
        self.tracked_agents = [] 
        
    def startTracking(self,agt):
        self.tracked_agents.append(agt)
        #TODO update the tracked state?
        #update the tracked states prime???

#module to make the agents swarm together
class CohesionModule(Module):
    
    gamma = 0.01

    def __init__(self):
        super().__init__()
        
        self.state = np.array([]) #the vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        

    def updateState(self,agt):
        #find the centroid
        centroid = np.array(agt.position)
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
            
        print(centroid)
        centroid = centroid /(len(self.tracked_agents)+1)
        self.state = centroid - agt.position
        print(self.state) 
    
    def updateStatePrime(self,agt):
        #find the centroid
        centroid = agt.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 

        centroid = centroid /(len(self.tracked_agents)+1)
        self.state_prime = centroid - agt.position
        print(self.state_prime)

#module to prevent agents from hitting each other
class CollisionModule(Module):
    
    gamma = 0

    def __init__(self):
        super().__init__()
        print(CollisionModule.gamma)

