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
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = Qlearning()    #define a Qleaning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = 120 #the amount of time over which we tranistion from exploration to exploitation 

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


    #select next action for this module with a softmax porabability mass function
    def select_next_action(self):
        
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

#module inherited from the base class.
#make the agents swarm together
class CollisionModule(Module):

    #rewards (punishements) for keeping distance to other agents. 1st entry is the punishemnt 
    #for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [-100,-1,0] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [16,49]


    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vectors from the agent to the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = Qlearning()    #define a Qleaning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        #in seconds TODO change the name of this
        self.exploitation_rise_time = 120 #the amount of time over which we tranistion from exploration to exploitation 

        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY     #safest not to do anyting for first action
        self.gamma = 0.1                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning


    
    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function
        #for each reward tier range
        for i in range(0,len(CollisionModule.ranges_squared)):
            pass
            #set marker size to be the diameter of the range
            # mkr_size = (np.sqrt(CollisionModule.ranges_squared[i])/2)**2*3.14
            # mkr_size = np.sqrt(20**2*3.14)
            # plt.plot(0,0,'ro',markersize=mkr_size,color='red', alpha=0.1)
            # #plot the marker for each agent
            # for agnt in self.tracked_agents:
            #     plt.plot(agnt.position[0],agnt.position[1],'ro',markersize=mkr_size,color='purple', alpha=0.01)
            

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
            # print('state prime is')
            # print(self.state_prime)
            # print('i is')
            # print(i)
            # print('state prime i is')
            # print(self.state_prime[i])
            # print('shape of state prime[0] is')
            # print(self.state_prime[i].shape[0])

            for j in range(0,self.state_prime[i].shape[0]):
                # print('state is')
                # print(self.state_prime)
                # # print('i is')
                # print(i)
                # print('j is')
                # print(j)
                dist_squared = dist_squared + self.state_prime[i,j]**2
            
            # print('distance squared is:')
            # print(dist_squared)

            #tiered reward scheme
            #loop through each range to give the appropriate reward
            rewarded = False
            for j in range(0,len(CollisionModule.ranges_squared)):
                if dist_squared <= CollisionModule.ranges_squared[j]:
                    self.instant_reward[i] = CollisionModule.rewards[j]
                    rewarded = True    
                    break
            
            #not in range, apply last reward
            if rewarded == False:
                self.instant_reward[i] = CollisionModule.rewards[-1]

            # print('instant reward is')
            # print(self.instant_reward[i])

            #continuous reward scheme
            # print('reward is')
            # print(-(2 - .1*dist_squared))
            # print('reward i is')
            # print(self.instant_reward[i])
            # self.instant_reward[i] = -(2 - .1*dist_squared)

    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward)

    #select next action for this module with a softmax porabability mass function
    def select_next_action(self):
        
        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(self.tracked_agents)):
            # print("Q row is")
            # print(i)
            # print(self.Q.fetch_row_by_state(self.state[i]))
            action_weights = action_weights + self.Q.fetch_row_by_state(self.state[i])

        # print('summed Qrow is')
        # print(action_weights)

        
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
                T = 1000.0 - (1000.0-0.1)*(curr_time - self.init_time)/self.exploitation_rise_time
            else:
                T = 0.1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
            # print('action_weights are')
            # print(action_weights) 

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
