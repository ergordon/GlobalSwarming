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

    def get_q_data(self, collapse):
        pass

    def load_q_data(self,q_states,q_tables):
        pass

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
    rewards = [1,-1] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [200]

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for Q-learning before assigning to state
        self.Q = Qlearning()    #define a Q-learning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        
        self.action = Action.STAY          #safest not to do anything for first action
        self.action_prime = Action.STAY     #safest not to do anything for first action
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
        
        # tiered reward scheme
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

        #continuous reward scheme
        # self.instant_reward = 2 - .1*dist_squared


    #visualization for this module. 
    # draw a transparent circle around the centroid 
    def visualize(self):
        super().visualize() #inherited class function
            
        centroid = self.parent_agent.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
        centroid = centroid / (len(self.tracked_agents)+1)
        
        #set marker size to be the diameter of the range
        mkr_size = np.sqrt(CohesionModule.ranges_squared[-1])

        #plot range circle, mkrsize is the radius.
        circle = plt.Circle((centroid[0],centroid[1]), mkr_size, color='green', alpha=0.1/Simulation.num_agents)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.add_artist(circle)

    def get_module_weight(self):
        
        #first find the squared distance from the centroid
        dist_squared = 0
        for i in range(0,len(self.state_prime)):
            dist_squared = dist_squared + self.state_prime[i]**2

        # #now pass into a weighting function
        # return -2/(1+np.exp(dist_squared/120)) + 1
        
        if dist_squared >= CohesionModule.ranges_squared[0]:
            return 1
        else:
            return 0


    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):
        
        #create a set of probabilities for each action
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
            if(curr_time - self.init_time < Simulation.exploitation_rise_time):
                T = 1000.0 - (1000.0-0.1)*(curr_time - self.init_time)/Simulation.exploitation_rise_time
            else:
                T = 0.1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308

        #normalize the weights to create probabilities
        # if(np.sum(action_weights) != 0):
        #     action_weights = action_weights / np.sum(action_weights)
        # else:
        #     action_weights = np.ones(len(Action))/len(Action)

        return action_weights


    def get_q_data(self,collapse):
        pass


    def load_q_data(self,q_states,q_tables):
        super.load_q_data(q_states,q_tables)

        #TODO have some sanity checks on load

        self.Q.q_table = q_tables
        self.Q.q_states = q_states


##############################################################################
#   End Cohesion Module Class
##############################################################################

##############################################################################
#   Begin Collision Module Class
##############################################################################

#module derived from the base class.
#make the agents swarm together
class CollisionModule(Module):

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [-100,-1,0] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [4,81]

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vectors from the agent to the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = []    #define a Qleaning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        
        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY     #safest not to do anyting for first action
        self.gamma = 0                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning

        self.collision_count = 0        #number of times this module has recorded a collision (with another agent) for this agent
    
    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function

        for i in range(0,len(CollisionModule.ranges_squared)):

            #set marker size to be the diameter of the range
            mkr_size = np.sqrt(CollisionModule.ranges_squared[i])

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

            self.Q.append(Qlearning())          
        
        super().start_tracking(agt) 

    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        for i in range(0,len(self.tracked_agents)):
            self.Q[i].update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])

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
            for j in range(0,self.state_prime[i].shape[0]):
                dist_squared = dist_squared + self.state_prime[i,j]**2

            # tiered reward scheme
            #loop through each range to give the appropriate reward
            rewarded = False
            for k in range(0,len(CollisionModule.ranges_squared)):
                if dist_squared <= CollisionModule.ranges_squared[k]:
                    self.instant_reward[i] = CollisionModule.rewards[k]
                    rewarded = True    
                    break
            
            #not in range, apply last reward (punishment)
            if rewarded == False:
                self.instant_reward[i] = CollisionModule.rewards[-1]


            # # continuous reward scheme that offeres severe punishements for being very close to other agents
            # # the function is always negative but is asymtotic to 0 as dist_squared approaches infinity
            # self.instant_reward[i] = 10.0*(dist_squared/(10.0+dist_squared)-1.0)

            
    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward)

    def get_module_weight(self):
        
        min_dist_squared = 1.7976931348623157e+308

        #iterate over each state, finding the shortest distance squared
        for i in range (0,len(self.tracked_agents)):
            dist_squared = 0
            for j in range(0,len(self.state_prime[i])):
                dist_squared = dist_squared + self.state_prime[i,j]**2

            # print(dist_squared)

            if dist_squared < min_dist_squared:
                min_dist_squared = dist_squared

        if min_dist_squared <= CollisionModule.ranges_squared[-1]:
            return 1
        else:
             return 0
        
        # #now pass into a weighting function
        # if min_dist_squared == 0:
        #     return 1
        # else:
        #     return -1*(min_dist_squared/(1+min_dist_squared)-1)

    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):

        #create a set of weights for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(self.tracked_agents)):
            action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state_prime[i])
        
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
            if(curr_time - self.init_time < Simulation.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/Simulation.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        #normalize the weights to create probabilities
        # if(np.sum(action_weights) != 0):
        #     action_weights = action_weights / np.sum(action_weights)
        # else:
        #     action_weights = np.ones(len(Action))/len(Action)

        return action_weights


    def load_q_data(self,q_states,q_tables):
        super.load_q_data(q_states,q_tables)

        for i in range(0,len(self.tracked_agents)):
            self.Q[i].q_table = q_tables
            self.Q[i].q_states = q_states


##############################################################################
#   End Collision Module Class
##############################################################################

##############################################################################
#   Begin Boundary Module Class
##############################################################################

#module derived from the base class.
#make the agents swarm together
class BoundaryModule(Module):

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward for being out of range
    rewards = [-1,0] 
    #the discrete ranges at which the agent can collect rewards
    ranges = [4]

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vectors from the agent to the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = []    #define a Qleaning object for each module instance   
            
        self.init_time = time.time() #store the time at which the agent was initialized

        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY    #safest not to do anyting for first action
        self.gamma = 0                     #discount factor. keep in range [0,1]. can be tuned to affect Q learning

        self.collision_count = 0           #number of times this module has recorded a collision (with another agent) for this agent
    
        self.state = np.zeros((len(Simulation.search_space)*2,1))
        self.state_prime = np.zeros((len(Simulation.search_space)*2,1))
        self.instant_reward = np.zeros(len(Simulation.search_space)*2)

        for i in range(len(Simulation.search_space)*2):
            self.Q.append(Qlearning())   
        
    #visualization for this module. 
    # draw a dotted line showing the boundary threshold
    def visualize(self):
        super().visualize() #inherited class function

        #TODO make this work automatically with 3 axes
        ax = plt.gca()
        rect = plt.Rectangle((Simulation.search_space[0][0] + BoundaryModule.ranges[0],Simulation.search_space[1][0] + BoundaryModule.ranges[0]),
                            Simulation.search_space[0][1] - Simulation.search_space[0][0] - 2*BoundaryModule.ranges[0],Simulation.search_space[1][1] - Simulation.search_space[1][0] - 2*BoundaryModule.ranges[0],
                            linewidth=2,edgecolor='lightsteelblue',fill=False,linestyle='--')
        ax.add_patch(rect)
        ax.set_aspect('equal')

    #for the collision module, this is used to check for and track collisions between agents. 
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        # for i in range(0,len(self.state_prime)):
        #     if(np.array_equal(self.state_prime[i],np.array([0,0]))):
        #         self.collision_count = self.collision_count + 1
    
    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        for i in range(0,len(Simulation.search_space)*2):
            self.Q[i].update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])             


    #update the state that the agent is currently in
    #for this module, it is a vector containing distances from the agent to each boundary
    # Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)
    def update_state(self):
        for i in range(0,len(Simulation.search_space)):   
            #round to whole numbers for discretization
            self.state[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
            self.state[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])    
            
    # update the state that agent is in. Store it in state_prime because it is called after 
    # executing an action and the Q object needs both the orignal state and the state after exectuion 
    # for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):

        for i in range(0,len(Simulation.search_space)):   
            #round to whole numbers for discretization
            self.state_prime[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
            self.state_prime[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])  

    # each module 
    def get_module_weight(self):
        
        min_dist = 1.7976931348623157e+308

        for i in range(0,len(Simulation.search_space)):  
            if np.abs(self.state_prime[i*2]) < min_dist:
                min_dist = np.abs(self.state_prime[i*2])
            if np.abs(self.state_prime[i*2+1]) < min_dist:
                min_dist = np.abs(self.state_prime[i*2+1])

        if min_dist <= BoundaryModule.ranges[-1]:
            return 1
        else:
            return 0


    #determine the reward for executing the action (not prime) in the state (not prime)
    #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        for i in range(0,len(Simulation.search_space)):  
            
            # self.instant_reward[i*2] = 0
            # self.instant_reward[i*2+1] = 0

            #handle upper bounds
            if(self.state_prime[i*2] >= BoundaryModule.ranges[0]):
                # self.instant_reward[i*2] = self.instant_reward[i*2] + BoundaryModule.rewards[-1]
                self.instant_reward[i*2] = BoundaryModule.rewards[-1]
            else:
                # self.instant_reward[i*2] = self.instant_reward[i*2] + BoundaryModule.rewards[0]
                self.instant_reward[i*2] = BoundaryModule.rewards[0]

            #handle lower bounds
            if(self.state_prime[i*2+1] <= -BoundaryModule.ranges[0]):
                # self.instant_reward[i*2+1] = self.instant_reward[i*2+1] + BoundaryModule.rewards[-1]
                self.instant_reward[i*2+1] = BoundaryModule.rewards[-1]
            else:
                # self.instant_reward[i*2+1] = self.instant_reward[i*2+1] + BoundaryModule.rewards[0]
                self.instant_reward[i*2+1] = BoundaryModule.rewards[0]


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

        for i in range (0,len(Simulation.search_space)*2):
            action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state_prime[i])
        

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
            if(curr_time - self.init_time < Simulation.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/Simulation.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        # #normalize the weights to create probabilities
        # if(np.sum(action_weights) != 0):
        #     action_weights = action_weights / np.sum(action_weights)
        # else:
        #     action_weights = np.ones(len(Action))/len(Action)

        return action_weights


    def load_q_data(self,q_states,q_tables):
        super.load_q_data(q_states,q_tables)

        for i in range(0,len(Simulation.search_space)*2):
            self.Q[i].q_table = q_tables[i]
            self.Q[i].q_states = q_states[i]


##############################################################################
#   End Boundary Module Class
##############################################################################

##############################################################################
#   Begin Target Seek Module Class
##############################################################################

#module to encourage agents to track target (Future: moving target?)
class TargetSeekModule(Module):

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    #rewards = [10,5,3,-2] 
    rewards = [10, -1]
    #the discrete ranges at which the agent can collect rewards
    #ranges_squared = [25,225,625]
    ranges_squared = [25]
    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vector from the agent to the target
        self.state_prime = np.array([]) #same as state but for the next step. Used for Q-learning before assigning to state
        self.Q = Qlearning()    #define a Q-learning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        
        self.action = Action.STAY         # safest not to do anything for first action
        self.action_prime = Action.STAY   # safest not to do anything for first action
        self.gamma = 0.9                 # discount factor. keep in range [0,1]. can be tuned to affect Q learning
        # self.target = Simulation.targets  # target location

    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function

        #for each reward tier range
        #TO DO: Make this only run for the first agent and then not run it again. Currently runns for all agents.
        for i in range(0,len(TargetSeekModule.ranges_squared)):
            
            #set marker size to be the diameter of the range
            mkr_size = np.sqrt(TargetSeekModule.ranges_squared[i])

            #plot target
            plt.plot(Simulation.targets[0],Simulation.targets[1],'bo')

            #plot range circle, mkrsize is the radius.
            circle = plt.Circle((Simulation.targets[0],Simulation.targets[1]), mkr_size, color='purple', alpha=0.1)
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
        self.state = np.round(Simulation.targets - self.parent_agent.position, 0) 
    
    #update the state that agent is in. Store it in state_prime because it is called after 
    #executing an action and the Q object needs both the original state and the state after execution 
    #for this module, it is the vector pointing from the agent to the swarm centroid
    #TODO use the centroid of the agents within a defined range
    def update_state_prime(self):
        #round to whole numbers for discretization
        self.state_prime = np.round(Simulation.targets - self.parent_agent.position, 0)


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
            self.instant_reward = TargetSeekModule.rewards[-1]
            #self.instant_reward = -math.log(dist_squared + 10) + 5 #EQN5

        # self.instant_reward = -math.log(dist_squared + 10) + 5 #EQN5

    def get_module_weight(self):
        
        dist_squared = 0
        for i in range(0,len(self.state_prime)):
            dist_squared = dist_squared + self.state_prime[i]**2

        if dist_squared < TargetSeekModule.ranges_squared[-1]:
            return 0
        else:
            return 0.25
        
        # #now pass into a weighting function
        # return -2/(1+np.exp(dist_squared/4)) + 1

    #select next action for this module with a soft max probability mass function
    def get_action_weights(self):
    
        #create a set of probabilities for each action
        action_weights = np.zeros(len(Action))
        
        Qrow = self.Q.fetch_row_by_state(self.state_prime) 
        
        #for each possible agent action
        for i in range (0,len(Action)):
            #get the appropriate Q value Q table row corresponding to the current state 
            #and the action being iterated over
            Qval = Qrow[i]

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            #linearly change T to decrease exploration and increase exploitation over time
            curr_time = time.time()
            if(curr_time - self.init_time < Simulation.exploitation_rise_time and Simulation.exploitation_rise_time != 0):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/Simulation.exploitation_rise_time
            else:
                T = 1
            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)

            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
    
        return action_weights

    def load_q_data(self,q_states,q_tables):
        super.load_q_data(q_states,q_tables)

        self.Q.q_table = q_tables
        self.Q.q_states = q_states

##############################################################################
#   End Target Seek Module Class
##############################################################################

##############################################################################
#   Begin Obstacle Avoidance Module Class
##############################################################################

#module derived from the base class.
#make the agents swarm together
class ObstacleAvoidanceModule(Module):

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [-100,-10,0] 
    #the discrete ranges at which the agent can collect rewards
    
    #new
    ranges = [2,4]

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #inherited class initialization
        
        self.state = np.array([]) #the vectors from the agent to the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = []    #define a Qleaning object for each module instance        
        
        self.init_time = time.time() #store the time at which the agent was initialized
        
        self.action = Action.STAY          #safest not to do anyting for first action
        self.action_prime = Action.STAY     #safest not to do anyting for first action
        self.gamma = 0.1                   #discount factor. keep in range [0,1]. can be tuned to affect Q learning

        self.collision_count = 0        #number of times this module has recorded a collision (with another agent) for this agent
    
        self.state = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+2))
        self.state_prime = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+2))
        self.instant_reward = np.zeros(len(Simulation.obstacles))

        for i in range(0,len(Simulation.obstacles)):
            self.Q.append(Qlearning())

    #visualization for this module. 
    # draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() #inherited class function
        
        ax = plt.gca()    
        # Create a Obstacle Area
        for i in range(len(Simulation.obstacles)):
            rect = plt.Rectangle((Simulation.obstacles[i][0],Simulation.obstacles[i][1]),
                                Simulation.obstacles[i][2],Simulation.obstacles[i][3],
                                linewidth=2,edgecolor='slategray',facecolor='lightgray')
            ax.add_patch(rect)

        #TODO make this work automatically with 3 axes
        padding = ObstacleAvoidanceModule.ranges[-1]
        # ax = plt.gca()
        for i in range(len(Simulation.obstacles)):
            rect = plt.Rectangle((Simulation.obstacles[i][0]-padding,Simulation.obstacles[i][1]-padding),
                                Simulation.obstacles[i][2]+2*padding,Simulation.obstacles[i][3]+2*padding,
                                linewidth=1,edgecolor='lightgray',facecolor='none')
            ax.add_patch(rect)
        
        ax.set_aspect('equal')

    #for the collision module, this is used to check for and track collisions between agents. 
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        
        #old
        # for i in range(0,len(self.state_prime)):
        #     if(np.array_equal(self.state_prime[i],np.array([0,0]))):
        #         self.collision_count = self.collision_count + 1

        #TODO 
        # new

    #update the Q table for this module
    def update_q(self):
        #accessed through the Qlearning object
        for i in range(0,len(Simulation.obstacles)):
            self.Q[i].update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])             

    #update the state that the agent is currently in
    #for this module, it is the the set of vectors pointing from the agent to each other agent in the swarm
    def update_state(self):

        for i in range(0,len(Simulation.obstacles)): 
            obs_x = Simulation.obstacles[i][0]
            obs_y = Simulation.obstacles[i][1]
            agnt_x = self.parent_agent.position[0]
            agnt_y = self.parent_agent.position[1]
            width = Simulation.obstacles[i][2]
            height = Simulation.obstacles[i][3]

            obs_ctr_x = obs_x + 0.5*width
            obs_ctr_y = obs_y + 0.5*height

            d_mid_x = obs_x + 0.5*width - agnt_x
            d_mid_y = obs_y + 0.5*height - agnt_y
            sdmx = np.sign(d_mid_x)
            sdmy = np.sign(d_mid_y)
            padding = ObstacleAvoidanceModule.ranges[-1]

            state = np.empty([len(Simulation.search_space)+2,])
            
            state[0] = np.round(d_mid_x - sdmx*0.5*width)
            state[1] = np.round(d_mid_y - sdmy*0.5*height)
            
            state[2] = (obs_x - padding <= agnt_x and agnt_x <= obs_x + width + padding and 
                       obs_y - padding <= agnt_y and agnt_y <= obs_y + height + padding)
            
            if agnt_x >= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[3] = 1
            elif agnt_x <= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[3] = 2
            elif agnt_x <= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[3] = 3
            elif agnt_x >= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[3] = 4

            self.state[i] = state

    #update the state that agent is in. Store it in state_prime because it is called after 
    #executing an action and the Q object needs both the orignal state and the state after exectuion 
    #for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
       
        for i in range(0,len(Simulation.obstacles)): 
            obs_x = Simulation.obstacles[i][0]
            obs_y = Simulation.obstacles[i][1]
            agnt_x = self.parent_agent.position[0]
            agnt_y = self.parent_agent.position[1]
            width = Simulation.obstacles[i][2]
            height = Simulation.obstacles[i][3]

            obs_ctr_x = obs_x + width/2
            obs_ctr_y = obs_y + width/2

            d_mid_x = obs_x + 0.5*width - agnt_x
            d_mid_y = obs_y + 0.5*height - agnt_y
            sdmx = np.sign(d_mid_x)
            sdmy = np.sign(d_mid_y)
            padding = ObstacleAvoidanceModule.ranges[-1]

            state = np.empty([len(Simulation.search_space)+2,])
            
            state[0] = np.round(d_mid_x - sdmx*0.5*width)
            state[1] = np.round(d_mid_y - sdmy*0.5*height)
            
            state[2] = (obs_x - padding <= agnt_x and agnt_x <= obs_x + width + padding and
                       obs_y - padding <= agnt_y and agnt_y <= obs_y + height + padding)
            
            if agnt_x >= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[3] = 1
            elif agnt_x <= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[3] = 2
            elif agnt_x <= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[3] = 3
            elif agnt_x >= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[3] = 4

            self.state_prime[i] = state

    #determine the reward for executing the action (not prime) in the state (not prime)
    #action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):

        for i in range(0,len(Simulation.obstacles)):
            obs_x = Simulation.obstacles[i][0]
            obs_y = Simulation.obstacles[i][1]
            agnt_x = self.parent_agent.position[0]
            agnt_y = self.parent_agent.position[1]
            width = Simulation.obstacles[i][2]
            height = Simulation.obstacles[i][3]
            
            if self.state_prime[i,2]:

                for j in range(0,len(ObstacleAvoidanceModule.ranges)):
                    padding = ObstacleAvoidanceModule.ranges[j]
                    if (obs_x - padding <= agnt_x and agnt_x <= obs_x + width + padding and 
                    obs_y - padding <= agnt_y and agnt_y <= obs_y + height + padding):

                        self.instant_reward[i] = ObstacleAvoidanceModule.rewards[j]
                        break

            else:
                self.instant_reward[i] = ObstacleAvoidanceModule.rewards[-1]
       

    #update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward)

    def get_module_weight(self):

        #find if within the largest range of any obstacle
        in_range = False
        for i in range(0,len(Simulation.obstacles)):
            obs_x = Simulation.obstacles[i][0]
            obs_y = Simulation.obstacles[i][1]
            agnt_x = self.parent_agent.position[0]
            agnt_y = self.parent_agent.position[1]
            width = Simulation.obstacles[i][2]
            height = Simulation.obstacles[i][3]

            padding = ObstacleAvoidanceModule.ranges[-1]


            if (obs_x - padding <= agnt_x and agnt_x <= obs_x + width + padding and
                obs_y - padding <= agnt_y and agnt_y <= obs_y + height + padding):
                in_range = True

        if in_range:
            return 1
        else:
            return 0


    #get a set of action weights for this module to be used in conjuntion with those of other modules 
    #with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):

        #create a set of weights for each action
        action_weights = np.zeros(len(Action))
        #sum the action tables for every tracked agent
        for i in range (0,len(Simulation.obstacles)):
            action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state_prime[i])
            
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
            if(curr_time - self.init_time < Simulation.exploitation_rise_time):
                T = 1000.0 - (1000.0-1)*(curr_time - self.init_time)/Simulation.exploitation_rise_time
            else:
                T = 1

            #calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            #set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        #normalize the weights to create probabilities
        # if(np.sum(action_weights) != 0):
        #     action_weights = action_weights / np.sum(action_weights)
        # else:
        #     action_weights = np.ones(len(Action))/len(Action)
        return action_weights


    def load_q_data(self,q_states,q_tables):
        super.load_q_data(q_states,q_tables)
        for i in range(0,len(Simulation.obstacles)):
            self.Q[i].q_table = q_tables
            self.Q[i].q_states = q_states

##############################################################################
#   End Obstacle Avoidance Module Class
##############################################################################