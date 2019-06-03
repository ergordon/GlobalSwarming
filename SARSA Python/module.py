import numpy as np
from qlearning import Qlearning

from action import Action
import random
from scipy.stats import rv_discrete

import os.path
import sys
import pickle

import time

import matplotlib.pyplot as plt
import simulation as sim


##############################################################################
#   Module Base Class
##############################################################################

#Base class that all other modules should inherit from
class Module:
    
    #constructor
    def __init__(self,parent_agt):

        self.parent_agent = parent_agt #the agent that created and is storing this module instance
        self.tracked_agents = [] #list of agents being tracked by this module 
        self.instant_reward = [] #list of instantaneous rewards earned by the agent. 
        
        self.alpha = 0.1 #learning rate. keep in range [0,1]. can be tuned to affect Q learning

    #add an agent to the list of agents to be tracked by this module
    def start_tracking(self,agt):
        if agt not in self.tracked_agents:
            self.tracked_agents.append(agt)

    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = self.instant_reward
        self.parent_agent.add_total_reward(reward)

    #method for implementing visualization of the module
    #implement should be done in derived class
    def visualize(self):
        pass

    #method for implementing additional functionality for the module
    #will be called once per iteration for each module for each agent
    #implement should be done in derived class
    def auxilariy_functions(self):
        pass

    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):
        sys.exit('get_action_weights not implemented for this module. This function must be implemented for each module in the derived class')

##############################################################################
#   Module Base Class
##############################################################################    

##############################################################################
#   Begin Cohesion Module Class
##############################################################################

#module derived from the base class.
#make the agents swarm together
class CohesionModule(Module):

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [2,1,0,-1,-2] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [8,18,32,50]


    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = Qlearning()    #define a Qleaning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = 12 #the amount of time over which we tranistion from exploration to exploitation 

        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY     #safest not to do anyting for first action
        self.gamma = 0.01                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning


    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        self.Q.update_q(self.state,self.state_prime,self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward)

    #update the state that the agent is currently in
    #for this module, it is the vector pointing from the agent to a tracked agent
    #there is a separate state stored for each tracked agent
    def update_state(self):
        #find the centroid of self and all tracked agents
        centroid = np.array(self.parent_agent.position)
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
        
        centroid = centroid / (len(self.tracked_agents)+1)

        #round to whole numbers for discretization
        self.state = np.round(centroid - self.parent_agent.position,0) 

    #update the state that agent is in. Store it in state_prime because it is called after 
    #executing an action and the Q object needs both the orignal state and the state after exectuion 
    #for this module, it is the vector pointing from the agent to the swarm centroid
    #TODO use the centroid of the agents within a defined range
    def update_state_prime(self):
        #find the centroid
        centroid = self.parent_agent.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 

        centroid = centroid / (len(self.tracked_agents)+1)

         #round to whole numbers for discretization
        self.state_prime = np.round(centroid - self.parent_agent.position, 0)

    #determine the reward for executing the action (not prime) in the state (not prime)
    #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        #the state is the vector to the swarm centroid
        #use distance squared for range comparisons (sqrt is slow)
        dist_squared = 0
        for i in range(0,len(self.state_prime)):
            dist_squared = dist_squared + self.state_prime[i]**2
        
        #tiered reward scheme
        # #loop through each range to give the appropriate reward
        # rewarded = False
        # for i in range(0,len(CohesionModule.ranges_squared)):
        #     if dist_squared <= CohesionModule.ranges_squared[i]:
        #         self.instant_reward = CohesionModule.rewards[i]
        #         rewarded = True    
        #         break
        
        # #not in range, apply last reward (punishment)
        # if rewarded == False:
        #     self.instant_reward = CohesionModule.rewards[-1]

        #continuous reward scheme
        self.instant_reward = 2 - .1*dist_squared

    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):
        
        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        
        #for each possible agent action
        for i in range (0,len(Action)):
            #get the appropiate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qrow = self.Q.fetch_row_by_state(self.state) 
            Qval = Qrow[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < self.exploitation_rise_time):
                T = 1000.0 - (1000.0-0.1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 0.1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308

        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)

        return action_weights


    #select next action for this module with a softmax porabability mass function
    def select_next_action(self):
        
        #create a set of weights for each action
        action_weights = np.zeros(len(Action))
        
        #for each possible agent action
        for i in range (0,len(Action)):
            #get the appropiate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qrow = self.Q.fetch_row_by_state(self.state_prime) 
            Qval = Qrow[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < self.exploitation_rise_time):
                T = 1000.0 - (1000.0-0.1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 0.1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308

        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)

        #use a discrete random variable distribution to select the next action
        x=list(map(int,Action))
        px=action_weights
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        #set state_prime to be the selected next action
        self.action_prime = Action(sample)


##############################################################################
#   End Cohesion Module Class
##############################################################################

##############################################################################
#   Begin Collision Module Class
##############################################################################

#module derived from the base class.
#make the agents swarm together
class CollisionModule(Module):

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vectors from the agent to the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = Qlearning()    #define a Qleaning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = 0 #the amount of time over which we tranistion from exploration to exploitation 

        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY     #safest not to do anyting for first action
        self.gamma = 0.1                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning

        self.collision_count = 0        #number of times this module has recorded a collision (with another agent) for this agent
    
    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function
            
        #set marker size to be the diameter of the range
        mkr_size = np.sqrt(81.0)

        #plot range circle, mkrsize is the radius.
        circle = plt.Circle((self.parent_agent.position[0],self.parent_agent.position[1]), mkr_size, color='purple', alpha=0.1)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.add_artist(circle)

    #for the collision module, this is used to check for and track collisions between agents. 
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        for i in range(0,len(self.state_prime)):
            if(np.array_equal(self.state_prime[i],np.array([0,0]))):
                self.collision_count = self.collision_count + 1

    #add an agent to the list of agents to be tracked by this module
    def start_tracking(self,agt):
        #super().start_tracking(agt) #make sure this works correctly
        #first check if the agent is already being tracked
        if agt not in self.tracked_agents:
            if(len(self.tracked_agents) != 0):
                self.state = np.vstack([self.state,[0,0]])
                self.state_prime = np.vstack([self.state_prime,[0,0]])
                self.instant_reward = np.vstack([self.instant_reward,[0]])
            else:
                self.state = np.zeros((1,2))
                self.state_prime = np.zeros((1,2))
                self.instant_reward = np.zeros((1,1))        
        
        super().start_tracking(agt) 

    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        for i in range(0,len(self.tracked_agents)):
            self.Q.update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])

    #update the state that the agent is currently in
    #for this module, it is the the set of vectors pointing from the agent to each other agent in the swarm
    def update_state(self):
        #each state is the vector from the parent agent to the tracked agent
        for i in range(0,len(self.tracked_agents)):   
            #round to whole numbers for discretization
            self.state[i] = np.round(np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position),0)
            # print('state is')
            # print(self.state[i])

    #update the state that agent is in. Store it in state_prime because it is called after 
    #executing an action and the Q object needs both the orignal state and the state after exectuion 
    #for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
        for i in range(0,len(self.tracked_agents)):  
            #round to whole numbers for discretization  
            self.state_prime[i] = np.round(np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position),0)

    #determine the reward for executing the action (not prime) in the state (not prime)
    #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        for i in range(0,len(self.tracked_agents)):  
            
            #the state is the vector to the tracked agent
            #use distance squared for range comparisons (sqrt is slow)
            dist_squared = 0
            for j in range(0,self.state_prime[i].shape[0]):
                dist_squared = dist_squared + self.state_prime[i,j]**2

            # continuous reward scheme that offeres severe punishements for being very close to other agents
            # the function is always negative but is asymtotic to 0 as dist_squared approaches infinity
            self.instant_reward[i] = 10.0*(dist_squared/(10.0+dist_squared)-1.0)

            # if(dist_squared < 81):
            #     self.instant_reward[i] = -10
            # else:
            #     self.instant_reward[i] = 0


    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward)

    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):

        #create a set of weights for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(self.tracked_agents)):
            action_weights = action_weights + self.Q.fetch_row_by_state(self.state[i])
        
        #for each possible agent action
        for i in range (0,len(action_weights)):
            #get the appropiate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qval = action_weights[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < self.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)

        return action_weights

    #select next action for this module with a softmax porabability mass function
    def select_next_action(self):
        
        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(self.tracked_agents)):
            action_weights = action_weights + self.Q.fetch_row_by_state(self.state_prime[i])
        
        #for each possible agent action
        for i in range (0,len(action_weights)):
            #get the appropiate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qval = action_weights[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < self.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)

        #use a discrete random variable distribution to select the next action
        x=list(map(int,Action))
        px=action_weights
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        #set state_prime to be the selected next action
        self.action_prime = Action(sample)

##############################################################################
#   End Collision Module Class
##############################################################################


##############################################################################
#   Begin Boundary Module Class
##############################################################################

#module derived from the base class.
#make the agents swarm together
class BoundaryModule(Module):

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vectors from the agent to the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = []#Qlearning()    #define a Qleaning object for each module instance   
             
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = 30 #the amount of time over which we tranistion from exploration to exploitation 

        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY    #safest not to do anyting for first action
        self.gamma = 0                     #discount factor. keep in range [0,1]. can be tuned to affect Q learning

        self.collision_count = 0           #number of times this module has recorded a collision (with another agent) for this agent
    
        self.state = np.zeros((len(sim.Simulation.search_space),len(sim.Simulation.search_space[0])))
        self.state_prime = np.zeros((len(sim.Simulation.search_space),len(sim.Simulation.search_space[0])))
        self.instant_reward = np.zeros(len(sim.Simulation.search_space))

        self.action_count = 0.0 #temp debugging variable

        for i in range(len(sim.Simulation.search_space)):
            self.Q.append(Qlearning())   
        

    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function
        pass

    #for the collision module, this is used to check for and track collisions between agents. 
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        # for i in range(0,len(self.state_prime)):
        #     if(np.array_equal(self.state_prime[i],np.array([0,0]))):
        #         self.collision_count = self.collision_count + 1
    
    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        for i in range(0,len(sim.Simulation.search_space)):
            # if(self.instant_reward[i] < 0):
            #     print('search space index is: ')
            #     print(i)
            
            self.Q[i].update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])             

    #update the state that the agent is currently in
    #for this module, it is a vector containing distances from the agent to each boundary
    # Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)
    def update_state(self):
        
        for i in range(0,len(sim.Simulation.search_space)):   
            #round to whole numbers for discretization
            state = np.empty([2,])
            state[0] = np.round(sim.Simulation.search_space[i][1] - self.parent_agent.position[i])
            state[1] = np.round(sim.Simulation.search_space[i][0] - self.parent_agent.position[i])
            self.state[i] = state
            
    # update the state that agent is in. Store it in state_prime because it is called after 
    # executing an action and the Q object needs both the orignal state and the state after exectuion 
    # for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
        for i in range(0,len(sim.Simulation.search_space)):   
            #round to whole numbers for discretization
            state = np.empty([2,])
            state[0] = np.round(sim.Simulation.search_space[i][1] - self.parent_agent.position[i])
            state[1] = np.round(sim.Simulation.search_space[i][0] - self.parent_agent.position[i])
            self.state_prime[i] = state



    #determine the reward for executing the action (not prime) in the state (not prime)
    #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):

        threshold = 4
        
        for i in range(0,len(sim.Simulation.search_space)):  
            
            self.instant_reward[i] = 0

            #handle upper bounds
            if(self.state_prime[i,0] >= threshold):
                pass #only punish if the agent is close to/past the boundary
            else:
                self.instant_reward[i] = self.instant_reward[i] + self.state_prime[i,0] - threshold
                
            #handle lower bounds
            if(self.state_prime[i,1] <= -threshold):
                pass #only punish if the agent is close to/past the boundary
            else:
                self.instant_reward[i] = self.instant_reward[i] - self.state_prime[i,1] - threshold


    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward)

    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):

        #create a set of weights for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(sim.Simulation.search_space)):
            action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state[i])
        
        #for each possible agent action
        for i in range (0,len(action_weights)):
            #get the appropiate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qval = action_weights[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < self.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)

        return action_weights

    #select next action for this module with a softmax porabability mass function
    def select_next_action(self):
        
        self.action_count = self.action_count + 1

        near_bounds = False
        # for i in range (0,len(sim.Simulation.search_space)):
        #     # if(self.instant_reward[i] != 0):
        #     if(self.state_prime[i][0] <= 4.0):
        #         near_bounds = True
        #     if(self.state_prime[i][1] >= -4.0):
        #         near_bounds = True
        #     # if(self.state[i][0] <= 4.0):
        #     #     near_bounds = True
        #     # if(self.state[i][1] >= -4.0):
        #     #     near_bounds = True    


        
        if(near_bounds):
            print('agent near boundary, selecting next action')
            print('action number is')
            print(self.action_count)

        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(sim.Simulation.search_space)):
            # action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state[i]) #maybe all i need is to use state_prime here!!!
            action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state_prime[i]) #maybe all i need is to use state_prime here!!!
            if(near_bounds):
                print('index is')
                print(i)
                print('Q state prime is')
                print(self.state_prime[i])
                print('Q row prime is')
                print(self.Q[i].fetch_row_by_state(self.state_prime[i]))
        


        #for each possible agent action
        for i in range (0,len(action_weights)):
            #get the appropiate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qval = action_weights[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < self.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308

        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)
        else:
            action_weights = np.ones(len(Action))/len(Action)

        if(near_bounds):
            print('action_weights')
            print(action_weights)

        #use a discrete random variable distribution to select the next action
        x=list(map(int,Action))
        px=action_weights
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        #set state_prime to be the selected next action
        # self.action_prime = Action(sample)
        
        indices = np.argwhere(px == np.amax(px))
        if(len(indices) == 1):
            self.action_prime = Action(np.argmax(px))
        else:
            index = random.randint(0,len(indices)-1)
            self.action_prime = Action(indices[index])

        if(near_bounds):
            print('indices are')
            print(indices)

        if(near_bounds):
            print('action prime is')
            print(self.action_prime)

##############################################################################
#   End Boundary Module Class
##############################################################################



# ##############################################################################
# #   Begin Boundary Module Class
# ##############################################################################

# #module derived from the base class.
# #keep the agents within the boundaries of the search space
# class BoundaryModule(Module):

#     #TODO use multiple Q objects and states, one for each bounding axis


#     #class constructor
#     def __init__(self,parent_agt):
#         super().__init__(parent_agt) #inherited class initialization
        
#         self.state = np.array([])   #the perpendicular distance from the agent to each wall. Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case) 
#         self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
#         self.Q = Qlearning()    #define a Qleaning object for each module instance        
        
#         self.init_time = time.time() #store the time at which the agent was initialized
#         #in seconds TODO change the name of this
#         self.exploitation_rise_time = 0 #the amount of time over which we tranistion from exploration to exploitation 

#         self.action = Action.STAY          #safest not to do anyting for first action
#         self.action_prime = Action.STAY     #safest not to do anyting for first action
#         self.gamma = 0.0                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning

#         self.collision_count = 0        #number of times this module has recorded a collision (with the boundary) for this agent
    
#     #visualization for this module. 
#     # draw a transparent circle for each tracked agent for each reward range 
#     def visualize(self):
#         super().visualize() #inherited class function
#         pass

#     #for the boundary module, this is used to check for and track collisions between the agent and the boundary. 
#     def auxilariy_functions(self):
#         super().auxilariy_functions() #inherited class function
        
#         #The state is the perpendicular distance from the agent to each boundary.
#         #It is kept in the following format [+x,-x,+y,-y] (append [+z,-z] for 3D case)
#         #when within all bounds, the state elements should be positive for upper bounds and negative for lower bounds 
#         for i in range(0,len(self.parent_agent.position)):
#             if(self.state_prime[i*2] <= 0):
#                 self.collision_count = self.collision_count + 1
#             if(self.state_prime[i*2 + 1] >= 0):
#                 self.collision_count = self.collision_count + 1

#     #update the Q table for this module
#     def update_q(self):
#         #accessed through the Qlearning object
#         self.Q.update_q(self.state,self.state_prime,self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward)

#     #update the state that the agent is currently in
#     #for this module, it is a vector containing distances from the agent to each boundary
#     # Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)
#     def update_state(self):
        
#         state = np.empty([2*len(self.parent_agent.position),])
#         for i in range(0,len(self.parent_agent.position)):
#             state[2*i] = np.round(sim.Simulation.search_space[i][1] - self.parent_agent.position[i])
#             state[2*i + 1] = np.round(sim.Simulation.search_space[i][0] - self.parent_agent.position[i])
#         # print('updating state')
#         # print('state prime')
#         # print(state)
#         self.state = state

#     #update the state that agent is in. Store it in state_prime because it is called after 
#     #executing an action and the Q object needs both the orignal state and the state after exectuion 
#     #for this module, it is a vector containing distances from the agent to each boundary
#     #Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)
#     def update_state_prime(self):

#         state = np.empty([2*len(self.parent_agent.position),])
#         for i in range(0,len(self.parent_agent.position)):
#             state[2*i] = np.round(sim.Simulation.search_space[i][1] - self.parent_agent.position[i])
#             state[2*i + 1] = np.round(sim.Simulation.search_space[i][0] - self.parent_agent.position[i])
#         # print('updating state prime')
#         # print('state prime is')
#         # print(state)
#         self.state_prime = state

#     #determine the reward for executing the action (not prime) in the state (not prime)
#     #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
#     def update_instant_reward(self):
        
#         threshold = 4 #TODO move this

#         self.instant_reward = 0
#         for i in range(0,len(self.parent_agent.position)):
            
#             #handle upper bounds
#             if(self.state_prime[i*2] >= threshold):
#                 pass #only punish if the agent is close to/past the boundary
#             else:
#                  self.instant_reward = self.instant_reward + self.state_prime[i*2] - threshold
            
#             #handle lower bounds
#             if(self.state_prime[i*2 + 1] <= -threshold):
#                 pass #only punish if the agent is close to/past the boundary
#             else:
#                  self.instant_reward = self.instant_reward - self.state_prime[i*2 + 1] - threshold

#         # print('updating instant reward')
#         # print('state prime is')
#         # print(self.state_prime)
#         print('instant reward is')
#         print(self.instant_reward)

#     #update parent agents total reward based on the module's current instant reward
#     def update_total_reward(self):
#         self.parent_agent.add_total_reward(self.instant_reward)

#     #get a set of action weights for this module to be used in conjuntion with those of other modules 
#     #with the purpose of selecting a single action for the agent to perform 
#     def get_action_weights(self):

#         # #create a set of weights for each action
#         # action_weights = np.zeros(len(Action))
#         # #sum the action tables for every tracked agent
#         # for i in range (0,len(self.tracked_agents)):
#         #     action_weights = action_weights + self.Q.fetch_row_by_state(self.state[i])
#         action_weights = np.copy(self.Q.fetch_row_by_state(self.state))

#         #for each possible agent action
#         for i in range (0,len(action_weights)):
#             #get the appropiate Q value Q table row corresponding to the current state 
#             #and the action being iterated over
#             Qval = action_weights[i]

#             #exploitation vs exploration constant
#             #big T encourages exploration
#             #small T encourages exploitation
#             T = 1
#             #linearly change T to decrease exploration and increase exploitation over time
#             curr_time = time.time()
#             if(curr_time - self.init_time < self.exploitation_rise_time):
#                 T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/self.exploitation_rise_time
#             else:
#                 T = 1

#             #calculate the weight for this action
#             action_weights[i] = np.exp(Qval/T)
            
#             #set the weight to the max float size in case it is beyond pythons max float size
#             if(action_weights[i] == float('inf')):
#                 action_weights[i] = 1.7976931348623157e+308
            
#         #normalize the weights to create probabilities
#         if(np.sum(action_weights) != 0):
#             action_weights = action_weights / np.sum(action_weights)
#         else:
#             action_weights = np.ones(len(Action))/len(Action)

#         return action_weights

#     #select next action for this module with a softmax porabability mass function
#     def select_next_action(self):
        
#         # #create a set of probabilities for each action
#         # action_weights = np.zeros(len(Action))
#         # #sum the action tables for every tracked agent
#         # for i in range (0,len(self.tracked_agents)):
#         #     action_weights = action_weights + self.Q.fetch_row_by_state(self.state[i])
#         action_weights = np.copy(self.Q.fetch_row_by_state(self.state))

#         #for each possible agent action
#         for i in range (0,len(action_weights)):
#             #get the appropiate Q value Q table row corresponding to the current state 
#             #and the action being iterated over
#             Qval = action_weights[i]

#             #exploitation vs exploration constant
#             #big T encourages exploration
#             #small T encourages exploitation
#             T = 1
#             #linearly change T to decrease exploration and increase exploitation over time
#             curr_time = time.time()
#             if(curr_time - self.init_time < self.exploitation_rise_time):
#                 T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/self.exploitation_rise_time
#             else:
#                 T = 1

#             #calculate the weight for this action
#             action_weights[i] = np.exp(Qval/T)
            
#             #set the weight to the max float size in case it is beyond pythons max float size
#             if(action_weights[i] == float('inf')):
#                 action_weights[i] = 1.7976931348623157e+308
            
#         #normalize the weights to create probabilities
#         if(np.sum(action_weights) != 0):
#             action_weights = action_weights / np.sum(action_weights)
#         else:
#             action_weights = np.ones(len(Action))/len(Action)

#         #use a discrete random variable distribution to select the next action
#         x=list(map(int,Action))
#         px=action_weights
#         sample=rv_discrete(values=(x,px)).rvs(size=1)

#         #set state_prime to be the selected next action
#         self.action_prime = Action(sample)

# ##############################################################################
# #   End Boundary Module Class
# ##############################################################################

