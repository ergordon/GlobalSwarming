import numpy as np
from simulation import Simulation
from qlearning import Qlearning

from action import Action
import random
from scipy.stats import rv_discrete

import os.path
import sys
import pickle

import time
import matplotlib.pyplot as plt
import math

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
        self.tracked_agents.append(agt)

    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = self.instant_reward
        self.parent_agent.add_total_reward(reward)

    #method for implementing visualization of the module
    #implement should be done in inherited class
    def visualize(self):
        pass

##############################################################################
#   Module Base Class
##############################################################################    

##############################################################################
#   Begin Cohesion Module Class
##############################################################################

#module inherited from the base class.
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
        self.state_prime = np.array([]) #same as state but for the next step. used for Q-learning before assigning to state
        self.Q = Qlearning()    #define a Q-learning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = 120 #the amount of time over which we transition from exploration to exploitation 

        self.action = Action.STAY          #safest not to do anything for first action
        self.action_prime = Action.STAY     #safest not to do anything for first action
        self.gamma = 0.01                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning


    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        self.Q.update_q(self.state,self.state_prime,self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward)

    #update the state that the agent is currently in
    #for this module, it is the vector pointing from the agent to the swarm centroid
    #TODO use the centroid of the agents within a defined range
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


    #select next action for this module with a softmax probability mass function
    def select_next_action(self):
        
        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        
        #for each possible agent action
        for i in range (0,len(Action)):
            #get the appropriate Q value Q table row corresponding to the current state 
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
            
        #normalize the weights to create probabilities
        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)

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
#   Begin Target Seek Module Class
##############################################################################

#module to encourage agents to track target (Future: moving target?)
class TargetSeekModule(Module):

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [5,2,-1,-2] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [25,225,625]

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vector from the agent to the target
        self.state_prime = np.array([]) #same as state but for the next step. Used for Q-learning before assigning to state
        self.Q = Qlearning()    #define a Q-learning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = Simulation.exploitation_rise_time #the amount of time over which we transition from exploration to exploitation 

        self.action = Action.STAY         # safest not to do anything for first action
        self.action_prime = Action.STAY   # safest not to do anything for first action
        self.gamma = 0.90                 # discount factor. keep in range [0,1]. can be tuned to affect Q learning
        self.target = Simulation.targets  # target location
    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function
        self.target = Simulation.targets  # Retrieve updated target coordinates

        #for each reward tier range
        #TO DO: Make this only run for the first agent and then not run it again. Currently runns for all agents.
        for i in range(0,len(TargetSeekModule.ranges_squared)):
            
            #set marker size to be the diameter of the range
            mkr_size = np.sqrt(TargetSeekModule.ranges_squared[i])

            #plot target
            plt.plot(self.target[0],self.target[1],'bo')

            #plot range circle, mkrsize is the radius.
            circle = plt.Circle((self.target[0],self.target[1]), mkr_size, color='purple', alpha=0.1)
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.add_artist(circle)

    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        self.Q.update_q(self.state,self.state_prime,self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward)

    #update the state that the agent is currently in
    #for this module, it is the vector pointing from the agent to the target
    def update_state(self):
        #round to whole numbers for discretization
        self.state = np.round(self.target - self.parent_agent.position, 0) 
        

    #update the state that agent is in. Store it in state_prime because it is called after 
    #executing an action and the Q object needs both the original state and the state after execution 
    #for this module, it is the vector pointing from the agent to the swarm centroid
    #TODO use the centroid of the agents within a defined range
    def update_state_prime(self): # CHECK THIS FUNCTION IF SWARM DOES NOT BEHAVE AS PLANNED
        #round to whole numbers for discretization
        self.state_prime = np.round(self.target - self.parent_agent.position, 0)
    #determine the reward for executing the action (not prime) in the state (not prime)
    #action (not prime) brings agent from state (not prime) to state_prime, and reward is calculated based on state_prime
    def update_instant_reward(self):
        
        #the state is the vector to the swarm centroid
        #use distance squared for range comparisons (sqrt is slow)
        dist_squared = 0
        for i in range(0,len(self.state_prime)):
            dist_squared = dist_squared + self.state_prime[i]**2

        # tiered reward scheme
        #loop through each range to give the appropriate reward
        rewarded = False
        for i in range(0,len(TargetSeekModule.ranges_squared)):
            if dist_squared <= TargetSeekModule.ranges_squared[i]:
                self.instant_reward = TargetSeekModule.rewards[i]
                rewarded = True    
                break
        
        #not in range, apply last reward (punishment)
        if rewarded == False:
            #self.instant_reward = TargetSeekModule.rewards[-1]
            #self.instant_reward = -5 - dist_squared/100 #EQN1
            #self.instant_reward = -5*((dist_squared/(100+dist_squared))-0.5) #EQN2
            #self.instant_reward = -5*((dist_squared/(10+dist_squared))-0.5) #EQN3
            #self.instant_reward = (10**(100/(dist_squared+100)))-2 #EQN4
            self.instant_reward = -math.log(dist_squared + 10) + 5 #EQN5
            #print(self.instant_reward)
           

    #select next action for this module with a soft max probability mass function
    def select_next_action(self):
        
        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        
        #for each possible agent action
        for i in range (0,len(Action)):
            #get the appropriate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qrow = self.Q.fetch_row_by_state(self.state) 
            Qval = Qrow[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
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

        #use a discrete random variable distribution to select the next action
        x=list(map(int,Action))
        px=action_weights
        sample=rv_discrete(values=(x,px)).rvs(size=1)

        #set state_prime to be the selected next action
        self.action_prime = Action(sample)

##############################################################################
#   End Target Seek Module Class
##############################################################################