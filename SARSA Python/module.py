import numpy as np
from simulation import Simulation
from simulation import TargetPath
from simulation import Controller
from simulation import Reward
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

        self.parent_agent = parent_agt  # The agent that created and is storing this module instance
        self.tracked_agents = []        # List of agents being tracked by this module 
        self.instant_reward = []        # List of instantaneous rewards earned by the agent. 

        if Simulation.testing == False:
            self.alpha = 0.7                # Learning rate. keep in range [0,1]. can be tuned to affect Q learning
        else:
            self.alpha = 0

        self.mmas_tier = 0

        self.gamma = 0                  # Discount factor

        self.state = np.array([])       # The vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) # Same as state but for the next step. used for Q-learning before assigning to state
        self.Q = np.array([])           # Define a Q-learning object for each module instance        
        self.state_transition = np.array([])

        self.collapsable_Q = False      # Whether or now the Q table array can be collapsed/combined into a single Q table
        self.init_time = time.time()    # Store the time at which the agent was initialized
        self.action = Action.STAY       # Safest not to do anything for first action
        self.action_prime = Action.STAY # Safest not to do anything for first action
  
    def base_round(x, prec=1, base=1): 
        return (base * (np.array(x) / base).round()).round(prec)

    # Add an agent to the list of agents to be tracked by this module
    def start_tracking(self,agt):
        if agt not in self.tracked_agents:
            self.tracked_agents.append(agt)

    # Method for implementing visualization of the module
    # Implement should be done in derived class
    def visualize(self):
        pass

    # Method for implementing additional functionality for the module
    #  will be called once per iteration for each module for each agent
    # Implement should be done in derived class
    def auxiliary_functions(self):
        pass

    def get_T(self):
        num_updates = 0
        for i in range (0,len(self.Q)):
            #TODO think of a better way to do this......
            num_updates = num_updates + self.Q[i].fetch_updates_by_state(self.state_prime[i])

        num_updates = num_updates / len(self.Q) #TODO would this work better as min(q_updates)?
        max_updates = 25

        T = 1 
        if num_updates < max_updates:
            T = 100.0 - (100.0-0.1)*num_updates/max_updates
        else:
            T = 0.1

        return T

    def get_epsilon(self):
        epsilon_sum = 0
        for i in range (0,len(self.Q)):
            #TODO think of a better way to do this......
            epsilon_sum = epsilon_sum + self.Q[i].fetch_epsilon_by_state(self.state_prime[i])

        return epsilon_sum / len(self.Q) #TODO would this work better as min epsilon value??


    # Get a set of action weights for this module to be used in conjuntion with those of other modules 
    #  with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):
        # print('getting action weights')
        # Create a set of weights for each action
        action_weights = np.zeros((len(self.Q),len(Action)))
        
        # get the action tables for every tracked agent
        for i in range (0,len(self.Q)):
            action_weights[i] = self.Q[i].fetch_row_by_state(self.state_prime[i])
            for j in range(0,len(action_weights[i])):
                if(action_weights[i,j] == float('inf')):
                    action_weights[i,j] = 1.7976931348623157e+308
                if(action_weights[i,j] == float('-inf')):
                    action_weights[i,j] = -1.7976931348623157e+308
        # print(action_weights)
        return action_weights


    # Update the Q table for this module
    def update_q(self):
        # Accessed through the Qlearning object
        for i in range(0,len(self.Q)):
            if self.state_transition[i]:
                self.Q[i].update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])

    # Update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward_sum = 0
        for i in range(0,len(self.Q)):
            if self.state_transition[i]:
                reward_sum = reward_sum + self.instant_reward[i]
        # if 1 in self.state_transition:
        # reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward_sum)

    def reset_init(self,e):
        pass

##############################################################################
#   Begin Cohesion Module Class
##############################################################################
class CohesionModule(Module):

    #Rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    # the last entry is the reward (punishment) for being out of range
    rewards = [1,-1] 
    # The discrete ranges at which the agent can collect rewards
    ranges_squared = [100]

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #Inherited class initialization
        
        # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        if (Simulation.RewardType == Reward.Tiered):
            self.gamma = 0.99
        elif (Simulation.RewardType == Reward.Continuous or Simulation.RewardType == Reward.Hybrid):
            self.gamma = 0.0
        

        self.Q = np.empty((1,), dtype=object)
        self.Q[0] = Qlearning()
        self.collapsable_Q = True              # Whether or now the Q table array can be collapsed/combined into a single Q table

        self.state = np.zeros((1,len(Simulation.search_space)))
        self.state_prime = np.zeros((1,len(Simulation.search_space)))
        self.instant_reward = np.zeros((1,))
        self.state_transition = np.zeros((1,))

        self.mmas_tier = 3

    def auxiliary_functions(self):
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        if dist_squared <= CohesionModule.ranges_squared[-1]:
            Simulation.cohesionDist.append(0)
        else:
            Simulation.cohesionDist.append(dist_squared)

    def get_centroid(self):
        # centroid = np.array(self.parent_agent.position)
        # for i in range(0,len(self.tracked_agents)):
        #     centroid = centroid + self.tracked_agents[i].position 
        # centroid = centroid / (len(self.tracked_agents)+1)
        # # centroid = np.array([0,0])
        
        # weight_sum = 0
        # weighted_centroid = np.zeros((len(centroid),))
        
        # state = centroid - self.parent_agent.position
        # weight = 0
        # for j in range(0,len(state)):
        #     weight = weight + state[j]*state[j]
        # inv_weight = 1
        # if weight != 0:
        #     inv_weight = 1/weight
        # weight_sum = weight_sum + inv_weight 
        # weighted_centroid = weighted_centroid + self.parent_agent.position*inv_weight

        # for i in range(0,len(self.tracked_agents)):
        #     weight = 0
        #     state = centroid - self.tracked_agents[i].position
        #     for j in range(0,len(state)):
        #         weight = weight + state[j]*state[j]
        #     inv_weight = 1
        #     if weight != 0:
        #         inv_weight = 1/weight
        #     weight_sum = weight_sum + inv_weight
        #     weighted_centroid = weighted_centroid + self.tracked_agents[i].position*inv_weight

        
        
        
        #set this to be the number of neigbors you want to track!!!
        num_neighbors =  len(self.tracked_agents)
        # print(num_neighbors)

        num_n = min(num_neighbors, len(self.tracked_agents))
        if num_n < len(self.tracked_agents):
            squared_distances = np.zeros((len(self.tracked_agents),))
            for i in range(0,len(self.tracked_agents)):
                dist_squared = 0
                dist_vector = self.tracked_agents[i].position - self.parent_agent.position
                for j in range(0,len(dist_vector)):
                    dist_squared = dist_squared + dist_vector[j]*dist_vector[j]
                squared_distances[i] = dist_squared
           
            sort_index = min(num_neighbors, len(self.tracked_agents)-1)
                
            nearest_indices = np.argpartition(squared_distances, sort_index)[:num_n]
        else:
            nearest_indices = np.arange(len(self.tracked_agents))
        # print('squared_distances')
        # print(squared_distances)
        # print('nearest_indices')
        # print(nearest_indices)
        # print(squared_distances[nearest_indices])
        
        centroid = np.array(self.parent_agent.position)
        for i in range(0,num_n):
            centroid = centroid + self.tracked_agents[nearest_indices[i]].position 
        centroid = centroid / (num_n+1)
        
        return centroid
        

    # Update the state that the agent is currently in
    #  for this module, it is the vector pointing from the agent to a tracked agent
    #  there is a separate state stored for each tracked agent
    def update_state(self):
        
        centroid = self.get_centroid()        

        state = centroid - self.parent_agent.position
        dist_squared = 0
        for i in range(0,len(state)):
            dist_squared = dist_squared + state[i]**2
        
        if dist_squared < 25*25:
            rounding_base = 1
        elif dist_squared < 50*50:
            rounding_base = 5
        else:
            rounding_base = 10

        self.state[0] = Module.base_round(state,2,rounding_base) 

        # Round to whole numbers for discretization
        # self.state[0] = np.round(centroid - self.parent_agent.position,0) 

    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  for this module, it is the vector pointing from the agent to the swarm centroid
    # TODO: use the centroid of the agents within a defined range
    def update_state_prime(self):
        centroid = self.get_centroid()        
        state = centroid - self.parent_agent.position 

        dist_squared = 0
        for i in range(0,len(state)):
            dist_squared = dist_squared + state[i]**2
        
        if dist_squared < 25*25:
            rounding_base = 1
        elif dist_squared < 50*50:
            rounding_base = 5
        else:
            rounding_base = 10


        # print('state')
        # print(state)
        self.state_prime[0] = Module.base_round(state,2,rounding_base) 
        # Round to whole numbers for discretization
        # self.state_prime[0] = np.round(centroid - self.parent_agent.position, 0)
        # print('self.state[0]')
        # print(self.state[0])

    def check_state_transition(self):
        transition = np.zeros((1,))
        # for i in range(0,len(self.state)):
        # print('checking transitions')
        # print('self.state[0]')
        # print(self.state[0])
        # print('self.state_prime[0]')
        # print(self.state_prime[0])
        # print('self.action')
        # print(self.action)


        # transition = np.zeros((len(self.tracked_agents),))
        # for i in range(0,len(self.state)):
        if np.array_equal(self.state[0], self.state_prime[0]):
            if self.action == Action.STAY:
                transition[0] = 1
        else:
            if self.action != Action.STAY:
                transition[0] = 1

        # if self.action == Action.STAY:
        #     transition[0] = 1
        # elif not np.array_equal(self.state[0], self.state_prime[0]): 
        #     transition[0] = 1

        # print('state, state prime')
        # print(self.state[0])
        # print(self.state_prime[0])
        # if np.array_equal(self.state[0], self.state_prime[0]):
        #     if self.action == Action.STAY:
        #         transition[0] = 1
        # else:
        #     if self.action != Action.STAY:
        #         transition[0] = 1

        # print('self.action')
        # print(self.action)
        
        self.state_transition = transition
        # print('self.state_transition')
        # print(self.state_transition)

    # Determine the reward for executing the action (not prime) in the state (not prime)
    # Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        if not self.state_transition[0]:
            self.instant_reward[0] = 0
            return

        # The state is the vector to the swarm centroid
        # NOTE: Use distance squared for range comparisons (sqrt is slow)
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2
        
        # Tiered reward scheme
        #  Loop through each range to give the appropriate reward
        if (Simulation.RewardType == Reward.Tiered):
            rewarded = False
            for i in range(0,len(CohesionModule.ranges_squared)):
                if dist_squared <= CohesionModule.ranges_squared[i]:
                    self.instant_reward[0] = CohesionModule.rewards[i]
                    rewarded = True    
                    break
            # Not in range, apply last reward (punishment)
            if rewarded == False:
                self.instant_reward[0] = CohesionModule.rewards[-1]

        elif (Simulation.RewardType == Reward.Continuous):
            self.instant_reward[0] = 20 - np.sqrt(dist_squared)

        elif (Simulation.RewardType == Reward.Hybrid):
            if dist_squared <= 100:
                self.instant_reward[0] = 10
            else:
                self.instant_reward[0] = 20 - np.sqrt(dist_squared)

    # Visualization for this module. 
    # Draw a transparent circle around the centroid 
    def visualize(self):
        super().visualize() #inherited class function

        # weighted_centroid = weighted_centroid/weight_sum
        # state = weighted_centroid - self.parent_agent.position
        centroid = self.get_centroid()        
        state = centroid - self.parent_agent.position 
        
        # Set marker size to be the diameter of the range
        mkr_size = np.sqrt(CohesionModule.ranges_squared[-1])

        #draw a line from the agent to center of the circle
        plt.plot([self.parent_agent.position[0],centroid[0]],[self.parent_agent.position[1],centroid[1]],'k-')

        # Plot range circle, mkrsize is the radius.
        circle = plt.Circle((centroid[0],centroid[1]), mkr_size, color='green', alpha=0.1/Simulation.num_agents)
        # circle = plt.Circle((weighted_centroid[0],weighted_centroid[1]), mkr_size, color='green', alpha=0.1/Simulation.num_agents)
        # circle = plt.Circle((centroid[0],centroid[1]), mkr_size, color='blue', alpha=0.1)
        ax = plt.gca()
        ax.set_aspect('equal')
        # ax.add_artist(circle)
        ax.add_artist(circle)


    def get_module_weights(self):
        
        module_weights = np.zeros((1,len(self.Q)))
        
        # First find the squared distance from the centroid
        dist_squared = 0
        for j in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,j]**2

        if dist_squared >= CohesionModule.ranges_squared[0]:
            module_weights[0] = 1
        else:
            module_weights[0] = 0

        return module_weights

    def reset_init(self,e):
        Simulation.cohesionDist = []


##############################################################################
#   Begin Collision Module Class
##############################################################################
class CollisionModule(Module):

    # Rewards for being within (or out of) range. 1st entry is the reward 
    #  for being within the range specified by the first entry in ranges_squared
    #  the last entry is the reward (punishment) for being out of range
    rewards = [-100,-1,0] 
    ranges_squared = [4,16] # The discrete ranges at which the agent can collect rewards

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt)     # Inherited class initialization
        
        # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        if (Simulation.RewardType == Reward.Tiered):
            self.gamma = 0.2
        elif (Simulation.RewardType == Reward.Continuous or Simulation.RewardType == Reward.Hybrid):
            self.gamma = 0.0

        #self.collision_count = 0        # Number of times this module has recorded a collision (with another agent) for this agent
        self.collapsable_Q = True        # Whether or now the Q table array can be collapsed/combined into a single Q table
        self.collided = False
        self.resetPoint = 0

        self.mmas_tier = 1

    # Visualization for this module. 
    # Draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() # Inherited class function

        for i in range(0,len(CollisionModule.ranges_squared)):
            # Set marker size to be the diameter of the range
            mkr_size = np.sqrt(CollisionModule.ranges_squared[i])

            #plot range circle, mkrsize is the radius.
            circle = plt.Circle((self.parent_agent.position[0],self.parent_agent.position[1]), mkr_size, color='purple', alpha=0.1)
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.add_artist(circle)

    # For the collision module, this is used to check for and track collisions between agents. 
    def auxiliary_functions(self):
        super().auxiliary_functions() #inherited class function
        if self.collided == True:
            if Simulation.episode_iter_num == self.resetPoint:
                self.collided = False
        for i in range(0,len(self.tracked_agents)): 
            for j in range(0,len(self.state_prime)):
                if(np.array_equal(self.state_prime[i],np.array([0,0]))):
                    if self.collided == False:
                        # Add 0.5 to agent collision because collisions get counted by every agent.
                        Simulation.agent_collision_count = Simulation.agent_collision_count + 0.5
                        self.collided = True
                        self.resetPoint = Simulation.episode_iter_num+5

    # Add an agent to the list of agents to be tracked by this module
    def start_tracking(self,agt):
        # First check if the agent is already being tracked
        if agt not in self.tracked_agents:
            if(len(self.tracked_agents) != 0):
                self.state = np.vstack([self.state,[0,0]])
                self.state_prime = np.vstack([self.state_prime,[0,0]])
                self.instant_reward = np.append(self.instant_reward,0)
                self.state_transition = np.append(self.state_transition, 0)
                self.Q = np.append(self.Q, Qlearning())
            else:
                self.state = np.zeros((1,len(Simulation.search_space)))
                self.state_prime = np.zeros((1,len(Simulation.search_space)))
                self.instant_reward = np.zeros((1,))  
                self.Q = np.empty((1,), dtype=object)
                self.Q[0] = Qlearning()
                self.state_transition = np.zeros((1,))
        
        super().start_tracking(agt) 

    # Update the state that the agent is currently in
    #  for this module, it is the the set of vectors pointing from the agent to each other agent in the swarm
    def update_state(self):
        # Each state is the vector from the parent agent to the tracked agent
        for i in range(0,len(self.tracked_agents)):   
            state_vector = np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position)
            dist_squared = 0
            for j in range(0,len(state_vector)):
                dist_squared = dist_squared + state_vector[j]**2
            
            if dist_squared < 20.0*20.0:
                rounding_base = 1
            elif dist_squared < 30.0*30.0:
                rounding_base = 5
            else:
                rounding_base = 10


            self.state[i] = Module.base_round(state_vector, 1, rounding_base)

            # Round to whole numbers for discretization
            # self.state[i] = np.round(np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position),0)


    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
        # Each state is the vector from the parent agent to the tracked agent
        for i in range(0,len(self.tracked_agents)):   
            state_vector = np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position)
            dist_squared = 0
            for j in range(0,len(state_vector)):
                dist_squared = dist_squared + state_vector[j]**2
            
            if dist_squared < 20.0*20.0:
                rounding_base = 1
            elif dist_squared < 30.0*30.0:
                rounding_base = 5
            else:
                rounding_base = 10
            
            # print('state_vector')
            # print(state_vector)
            self.state_prime[i] = Module.base_round(state_vector, 1, rounding_base)
            # print('self.state_prime[i]')
            # print(self.state_prime[i])

            # Round to whole numbers for discretization
            # self.state[i] = np.round(np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position),0)

    def check_state_transition(self):
        transition = np.zeros((len(self.tracked_agents),))
        for i in range(0,len(self.state)):
            if np.array_equal(self.state[i], self.state_prime[i]):
                if self.action == Action.STAY:
                    transition[i] = 1
            else:
                if self.action != Action.STAY:
                    transition[i] = 1

        # print('transition')
        # print(transition)
        self.state_transition = transition

    # Determine the reward for executing the action (not prime) in the state (not prime)
    # Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        for i in range(0,len(self.tracked_agents)):  
            
            if not self.state_transition[i]:
                self.instant_reward[i] = 0
            else:
                # The state is the vector to the tracked agent
                # Use distance squared for range comparisons (sqrt is slow)
                dist_squared = 0
                for j in range(0,self.state_prime[i].shape[0]):
                    dist_squared = dist_squared + self.state_prime[i,j]**2

                if (Simulation.RewardType == Reward.Tiered):
                    rewarded = False
                    for k in range(0,len(CollisionModule.ranges_squared)):
                        if dist_squared <= CollisionModule.ranges_squared[k]:
                            self.instant_reward[i] = CollisionModule.rewards[k]
                            rewarded = True    
                            break
                    if rewarded == False:
                        self.instant_reward[i] = CollisionModule.rewards[-1]

                if (Simulation.RewardType == Reward.Continuous):
                    self.instant_reward[i] = np.sqrt(dist_squared) - 4.0

                if (Simulation.RewardType == Reward.Hybrid):
                    if dist_squared <= 16:
                        self.instant_reward[i] = np.sqrt(dist_squared) - 4.0
                    else:
                        self.instant_reward[i] = 0

    def get_module_weights(self):
        
        module_weights = np.zeros((len(self.Q),1))
        
        for i in range (0,len(self.Q)):
            # First find the squared distance from the centroid
            dist_squared = 0
            for j in range(0,len(self.state_prime[0])):
                dist_squared = dist_squared + self.state_prime[i,j]**2


            if dist_squared <= CollisionModule.ranges_squared[-1]*1.1:
                module_weights[i] = 1
            else:
                module_weights[i] = 0
                
        return module_weights

    def reset_init(self,e):
        self.collided = False

##############################################################################
#   Begin Boundary Module Class
##############################################################################
class BoundaryModule(Module):

    rewards = [-1,0] # Discrete rewards
    ranges = [4]     # The discrete ranges at which the agent can collect rewards

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) # Inherited class initialization
        
        # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        if (Simulation.RewardType == Reward.Tiered):
            self.gamma = 0.2
        elif (Simulation.RewardType == Reward.Continuous or Simulation.RewardType == Reward.Hybrid):
            self.gamma = 0.0

        self.collision_count = 0           # Number of times this module has recorded a collision (with another agent) for this agent
    
        self.Q = np.empty((len(Simulation.search_space)*2,), dtype=object)
        for i in range(0,len(self.Q)):
            self.Q[i] = Qlearning()
        self.collapsable_Q = False # Whether or now the Q table array can be collapsed/combined into a single Q table
        
        self.state = np.zeros((len(Simulation.search_space)*2,1))
        self.state_prime = np.zeros((len(Simulation.search_space)*2,1))
        self.instant_reward = np.zeros(len(Simulation.search_space)*2)  
        self.state_transition = np.zeros(len(Simulation.search_space)*2)
        self.mmas_tier = 0



    # Visualization for this module. 
    #  Draw a dotted line showing the boundary threshold
    def visualize(self):
        super().visualize() # Inherited class function
        ax = plt.gca()
        rect = plt.Rectangle((Simulation.search_space[0][0] + BoundaryModule.ranges[0],Simulation.search_space[1][0] + BoundaryModule.ranges[0]),
                            Simulation.search_space[0][1] - Simulation.search_space[0][0] - 2*BoundaryModule.ranges[0],Simulation.search_space[1][1] - Simulation.search_space[1][0] - 2*BoundaryModule.ranges[0],
                            linewidth=2,edgecolor='lightsteelblue',fill=False,linestyle='--')
        ax.add_patch(rect)
        ax.set_aspect('equal')

    
    # Update the state that the agent is currently in
    #  For this module, it is a vector containing distances from the agent to each boundary
    #  Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)
    def update_state(self):
        # for i in range(0,len(Simulation.search_space)):   
        #     # Round to whole numbers for discretization
        #     self.state[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
        #     self.state[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])    
            
        for i in range(0,len(Simulation.search_space)):    
            discretization_base = 1
            if self.state[i*2] < 10:
                discretization_base = 1
            else:
                discretization_base = 10
            self.state[i*2] = Module.base_round(Simulation.search_space[i][1] - self.parent_agent.position[i], 2, discretization_base) 

            if(self.state[i*2+1] > -10):
                discretization_base = 1
            else:
                discretization_base = 10
            self.state[i*2+1] = Module.base_round(Simulation.search_space[i][0] - self.parent_agent.position[i], 2, discretization_base)  




    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  Executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  For this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
    
        for i in range(0,len(Simulation.search_space)):    
            state_vector = Simulation.search_space[i][1] - self.parent_agent.position[i]
            discretization_base = 1
            if state_vector < 10.0:
                discretization_base = 1
            else:
                discretization_base = 10.0
            # print('upper bound')
            # print(Simulation.search_space[i][1] - self.parent_agent.position[i])
            # print(discretization_base)
            self.state_prime[i*2] = Module.base_round(state_vector, 2, discretization_base) 

            state_vector = Simulation.search_space[i][0] - self.parent_agent.position[i]
            if(state_vector > -10.0):
                discretization_base = 1
            else:
                discretization_base = 10
            # print('lower bound')
            # print(Simulation.search_space[i][0] - self.parent_agent.position[i])
            self.state_prime[i*2+1] = Module.base_round(state_vector, 2, discretization_base)  


        # print('self.state_prime')
        # print(self.state_prime)

        # for i in range(0,len(Simulation.search_space)):   
        #     # Round to whole numbers for discretization
        #     self.state_prime[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
        #     self.state_prime[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])  

    #boundary module is special, state transition is always true when taking actions perpendicular to the walls.
    #this could be better, but 
    def check_state_transition(self):
        transition = np.zeros((len(Simulation.search_space)*2,))

        for i in range(0,len(Simulation.search_space)):    
            if self.state_prime[i*2] < 10:
                transition[i*2] = 1
            else:
                if np.array_equal(self.state[i*2], self.state_prime[i*2]):
                    if self.action == Action.STAY:
                        transition[i*2] = 1
                else:
                    if self.action != Action.STAY:
                        transition[i*2] = 1

            if self.state_prime[i*2+1] > -10:
                transition[i*2+1] = 1
            else:
                if np.array_equal(self.state[i*2+1], self.state_prime[i*2+1]):
                    if self.action == Action.STAY:
                        transition[i*2+1] = 1
                else:
                    if self.action != Action.STAY:
                        transition[i*2+1] = 1

        # print('self.state_transition')
        # print(self.state_transition)
        
        self.state_transition = transition

    def get_module_weights(self):
        
        module_weights = np.zeros((len(self.Q),1))

        for i in range(0,len(Simulation.search_space)):    
            if(self.state_prime[i*2] <= BoundaryModule.ranges[-1]):
                module_weights[i*2] = 1
            else:
                module_weights[i*2] = 0

            if(self.state_prime[i*2+1] >= -BoundaryModule.ranges[-1]):
                module_weights[i*2+1] = 1
            else:
                module_weights[i*2+1] = 0

        return module_weights

    # Determine the reward for executing the action (not prime) in the state (not prime)
    #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):

        for i in range(0,len(Simulation.search_space)):  
            if (Simulation.RewardType == Reward.Tiered):
                # Handle upper bounds
                if self.state_transition[i*2]:
                    if(self.state_prime[i*2] >= BoundaryModule.ranges[0]):
                        self.instant_reward[i*2] = BoundaryModule.rewards[-1]
                    else:
                        self.instant_reward[i*2] = BoundaryModule.rewards[0]

                # Handle lower bounds
                if self.state_transition[i*2+1]:
                    if(self.state_prime[i*2+1] <= -BoundaryModule.ranges[0]):
                        self.instant_reward[i*2+1] = BoundaryModule.rewards[-1]
                    else:
                        self.instant_reward[i*2+1] = BoundaryModule.rewards[0]

            elif (Simulation.RewardType == Reward.Continuous):
                if self.state_transition[i*2]:
                    # Handle upper bounds
                    self.instant_reward[i*2] = self.state_prime[i*2] - 4.0
                if self.state_transition[i*2+1]:
                    # Handle lower bounds
                    self.instant_reward[i*2+1] = -self.state_prime[i*2+1] - 4.0

            elif (Simulation.RewardType == Reward.Hybrid):
                if self.state_transition[i*2]:
                    # Handle upper bounds
                    if(self.state_prime[i*2] >= 4):
                        self.instant_reward[i*2] = 0
                    else:
                        self.instant_reward[i*2] = self.state_prime[i*2] - 4.0

                if self.state_transition[i*2+1]:
                    # Handle lower bounds
                    if(self.state_prime[i*2+1] <= -4):
                        self.instant_reward[i*2+1] = 0
                    else:
                        self.instant_reward[i*2+1] = -self.state_prime[i*2+1] - 4.0

##############################################################################
#   Begin Target Seek Module Class
##############################################################################
class TargetSeekModule(Module):

    rewards = [10, -1]       # Discrete rewards for a given range
    ranges_squared = [25]    # The discrete ranges at which the agent can collect rewards

    targets_entered = 0

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt)    # Inherited class initialization
        
        # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        if (Simulation.RewardType == Reward.Tiered):
            self.gamma = 0.99
        elif (Simulation.RewardType == Reward.Continuous or Simulation.RewardType == Reward.Hybrid):
            self.gamma = 0.0


        self.alpha = 0.0

        self.Q = np.empty((1,), dtype=object)
        self.Q[0] = Qlearning()
        self.collapsable_Q = True       # Whether or now the Q table array can be collapsed/combined into a single Q table

        self.state = np.zeros((1,len(Simulation.search_space)))
        self.state_prime = np.zeros((1,len(Simulation.search_space)))
        self.instant_reward = np.zeros((1,))
        self.state_transition = np.zeros((1,))

        self.in_target = False           # Bool for tracking if agent is in the target. False = not in target
        #self.targets_entered = 0         # Number of targets entered for a single episode
        
        self.mmas_tier = 2

    # Visualization for this module. 
    # Draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() # Inherited class function

        # For each reward tier range
        for i in range(0,len(TargetSeekModule.ranges_squared)):
            mkr_size = np.sqrt(TargetSeekModule.ranges_squared[i])     # Set marker size to be the diameter of the range
            plt.plot(Simulation.targets[0],Simulation.targets[1],'bo') # Plot target
            circle = plt.Circle((Simulation.targets[0],Simulation.targets[1]), mkr_size, color='purple', alpha=0.1)
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.add_artist(circle)

    #Track number of agents in the target range
    def auxiliary_functions(self):
        super().auxiliary_functions() # Inherited class function

        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        if (dist_squared <= self.ranges_squared[0]):
            if (self.in_target == False):
                Simulation.target_entries_count = Simulation.target_entries_count + 1
                self.in_target = True

        if(Simulation.target_agents_remaining > 0):
            self.in_target = False
            Simulation.target_agents_remaining = Simulation.target_agents_remaining -1

        if (Simulation.changeTargetOnArrival == True):
            if (Simulation.target_entries_count == Simulation.num_agents):
                
                if (Simulation.TargetType == TargetPath.Random):
                    arena_space = Simulation.arena_space
                    Simulation.targets = np.array([random.randint(arena_space[0][0]+5, arena_space[0][1]-5),
                                    random.randint(arena_space[1][0]+5, arena_space[1][1]-5)])

                    Simulation.target_histogram_data.append([TargetSeekModule.targets_entered, Simulation.episode_iter_num])

                elif (Simulation.TargetType == TargetPath.Planned):
                    #self.targets_entered = self.targets_entered + 1
                    if (TargetSeekModule.targets_entered < len(Simulation.target_array)-1):
                        TargetSeekModule.targets_entered = TargetSeekModule.targets_entered + 1
                        #if(self.targets_entered < len(Simulation.target_array)):
                            #self.targets_entered = self.targets_entered + 1
                        Simulation.targets = Simulation.target_array[TargetSeekModule.targets_entered]
                        # print("New Target")
                        # print("Targets Entered ",TargetSeekModule.targets_entered)
                        # print("Target Entered at "+str(time.time() - self.init_time)+" seconds")
                        # print("Target Entered at "+str(Simulation.episode_iter_num)+" iterations")
                        Simulation.target_histogram_data.append([TargetSeekModule.targets_entered, Simulation.episode_iter_num])

                    elif (TargetSeekModule.targets_entered == len(Simulation.target_array)-1):
                        TargetSeekModule.targets_entered = TargetSeekModule.targets_entered + 1
                        # print("Final Target Reached", TargetSeekModule.targets_entered)
                        #if(self.targets_entered == len(Simulation.target_array)):
                        Simulation.target_histogram_data.append([TargetSeekModule.targets_entered, Simulation.episode_iter_num])
                            
                    #print("Target Entered at "+str(time.time() - self.init_time)+" seconds")
                    #print("Target Entered at "+str(Simulation.episode_iter_num)+" iterations")
                    #Simulation.target_histogram_data.append([self.targets_entered, Simulation.episode_iter_num])

                Simulation.target_entries_count = 0
                Simulation.target_agents_remaining = Simulation.num_agents
        
    
    # Update the state that the agent is currently in
    #  For this module, it is the vector pointing from the agent to the target
    def update_state(self):
        # NOTE: Round to whole numbers for discretization
        state_vector = Simulation.targets - self.parent_agent.position
        dist_squared = 0
        for i in range(0,len(state_vector)):
            dist_squared = dist_squared + state_vector[i]**2
        
        if dist_squared < 15.0*15.0:
            rounding_base = 1
        elif dist_squared < 30.0*30.0:
            rounding_base = 5
        else:
            rounding_base = 10
        
        self.state_prime[0] = Module.base_round(state_vector, 1, rounding_base)
        
        # # Round to whole numbers for discretization
        # self.state[0] = np.round(Simulation.targets - self.parent_agent.position, 0) 
    
    # Update the state that agent is in. Store it in state_prime because it is called after 
    # Executing an action and the Q object needs both the original state and the state after execution 
    # For this module, it is the vector pointing from the agent to the target
    def update_state_prime(self):
        # NOTE: Round to whole numbers for discretization
        # decimal_place = 0
        vec = Simulation.targets - self.parent_agent.position
        dist_squared = 0
        for i in range(0,len(vec)):
            dist_squared = dist_squared + vec[i]**2
        
        if dist_squared < 20*20:
            rounding_base = 1
        elif dist_squared < 30*30:
            rounding_base = 5
        else:# dist_squared < 30*30:
            rounding_base = 10
        
        # print(vec)
        # print(np.round(vec,decimal_place))
        # print(Module.base_round(vec, 1, rounding_base))
        self.state_prime[0] = Module.base_round(vec, 1, rounding_base)
        # print(self.state_prime[0])
        # self.state_prime[0] = np.round(Simulation.targets - self.parent_agent.position, 0)


    def check_state_transition(self):
        transition = np.zeros((1,))
        
        if np.array_equal(self.state[0], self.state_prime[0]):
            if self.action == Action.STAY:
                transition[0] = 1
        else:
            if self.action != Action.STAY:
                transition[0] = 1

        self.state_transition = transition


    # Determine the reward for executing the action (not prime) in the state (not prime)
    #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calculated based on state_prime
    def update_instant_reward(self):

        if not self.state_transition[0]:
            self.instant_reward[0] = 0
            return

        # The state is the vector to the swarm centroid
        #  Use distancee squared for range comparisons (sqrt is slow)
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        # Tiered reward scheme
        #  Loop through each range to give the appropriate reward

        if (Simulation.RewardType == Reward.Tiered):
            rewarded = False
            for i in range(0,len(TargetSeekModule.ranges_squared)):
                if dist_squared <= TargetSeekModule.ranges_squared[i]:
                    self.instant_reward[0] = TargetSeekModule.rewards[i]
                    rewarded = True    
                    break
            if rewarded == False:
                self.instant_reward[0] = TargetSeekModule.rewards[-1]

        elif (Simulation.RewardType == Reward.Continuous):
            #self.instant_reward[0] = -2.5*(-math.log(math.sqrt(dist_squared) + 10) + 5)
            #self.instant_reward[0] = -math.log(dist_squared + 10) + 5
            #self.instant_reward[0] = -dist_squared
            #self.instant_reward[0] = -dist_squared*0.02+10
            #self.instant_reward[0] = -dist_squared**(0.5)/10
            self.instant_reward[0] = -2.0/39.0*np.sqrt(dist_squared) + 410.0/39.0

        elif (Simulation.RewardType == Reward.Hybrid):
            if dist_squared <= 25:
                self.instant_reward[0] = -2.0/39.0*5.0 + 410.0/39.0
            else:
                self.instant_reward[0] = -2.0/39.0*np.sqrt(dist_squared) + 410.0/39.0

        
    def get_module_weights(self):
        
        module_weights = np.zeros((1,len(self.Q)))
    
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        if dist_squared < TargetSeekModule.ranges_squared[-1]*0.9:
            module_weights[0] = 0
        else:
            module_weights[0] = 0.35

        return module_weights



    # Reset the initialized variables at the end of an episode.
    def reset_init(self,e):
        self.in_target = False
        Simulation.target_reached_episode_end[e] = self.targets_entered
        TargetSeekModule.targets_entered = 0
        
##############################################################################
#   Begin Obstacle Avoidance Module Class
##############################################################################
class ObstacleAvoidanceModule(Module):
    # Rewards for being within (or out of) range. 1st entry is the reward 
    #  for being within the range specified by the first entry in ranges_squared
    #  the last entry is the reward (punishment) for being out of range
    rewards = [-100,-10,0] 
    # The discrete ranges at which the agent can collect rewards
    ranges = [2,4]

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) # Inherited class initialization
        
        # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        if (Simulation.RewardType == Reward.Tiered):
            self.gamma = 0.5
        elif (Simulation.RewardType == Reward.Continuous or Simulation.RewardType == Reward.Hybrid):
            self.gamma = 0.0

        self.collapsable_Q = True    # Whether or not the Q table array can be collapsed/combined into a single Q table

        self.Q = np.empty((len(Simulation.obstacles),), dtype=object)
        for i in range(0,len(Simulation.obstacles)):
            self.Q[i] = Qlearning()

        # self.state = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+2))
        # self.state_prime = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+2))
        self.state = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+1))
        self.state_prime = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+1))
        self.instant_reward = np.zeros(len(Simulation.obstacles))
        self.state_transition = np.zeros(len(Simulation.obstacles))
        self.mmas_tier = 0


    # Visualization for this module. 
    #  Draw a transparent circle for each tracked agent for each reward range 
    def visualize(self):
        super().visualize() # Inherited class function
        
        ax = plt.gca()    
        # Create a Obstacle Area
        for i in range(len(Simulation.obstacles)):
            rect = plt.Rectangle((Simulation.obstacles[i][0],Simulation.obstacles[i][1]),
                                Simulation.obstacles[i][2],Simulation.obstacles[i][3],
                                linewidth=2,edgecolor='slategray',facecolor='lightgray')
            ax.add_patch(rect)

        padding = ObstacleAvoidanceModule.ranges[-1]
        for i in range(len(Simulation.obstacles)):
            rect = plt.Rectangle((Simulation.obstacles[i][0]-padding,Simulation.obstacles[i][1]-padding),
                                Simulation.obstacles[i][2]+2*padding,Simulation.obstacles[i][3]+2*padding,
                                linewidth=1,edgecolor='lightgray',facecolor='none')
            ax.add_patch(rect)
        
        ax.set_aspect('equal')


    # Track Agent-Obstacle Collisions
    def auxiliary_functions(self):
        super().auxiliary_functions() #inherited class function
        
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
                    if (obs_x <= agnt_x and agnt_x <= obs_x + width and 
                    obs_y <= agnt_y and agnt_y <= obs_y + height):
                        Simulation.obstacle_collision_count =  Simulation.obstacle_collision_count + 1
                        #print("Obstacle Y Collision "+str( Simulation.obstacle_collision_count))
                        break

    # Update the state that the agent is currently in
    #  for this module, it is the the set of vectors pointing from the agent to each other agent in the swarm
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
            if sdmx == 0:
                sdmx = 1
            if sdmy == 0:
                sdmy = 1
            padding = ObstacleAvoidanceModule.ranges[-1]

            state = np.empty([len(Simulation.search_space)+1,])
            
            state[0] = d_mid_x - sdmx*0.5*width
            state[1] = d_mid_y - sdmy*0.5*height
            
            if abs(state[0]) < 10 or abs(state[1]) < 10:
                state[0] = Module.base_round(state[0],2,1)
                state[1] = Module.base_round(state[1],2,1)
            elif abs(state[0]) < 20 or abs(state[1]) < 20:
                state[0] = Module.base_round(state[0],2,5)
                state[1] = Module.base_round(state[1],2,5)
            else:
                state[0] = Module.base_round(state[0],2,10)
                state[1] = Module.base_round(state[1],2,10)
            

            # state[0] = np.round(d_mid_x - sdmx*0.5*width)
            # state[1] = np.round(d_mid_y - sdmy*0.5*height)

            if agnt_x >= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[2] = 1
            elif agnt_x <= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[2] = 2
            elif agnt_x <= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[2] = 3
            elif agnt_x >= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[2] = 4

            self.state[i] = state


    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
       
        for i in range(0,len(Simulation.obstacles)): 
            obs_x = Simulation.obstacles[i][0]
            obs_y = Simulation.obstacles[i][1]
            agnt_x = self.parent_agent.position[0]
            agnt_y = self.parent_agent.position[1]
            width = Simulation.obstacles[i][2]
            height = Simulation.obstacles[i][3]

            obs_ctr_x = obs_x + width/2
            obs_ctr_y = obs_y + height/2

            d_mid_x = obs_x + 0.5*width - agnt_x
            d_mid_y = obs_y + 0.5*height - agnt_y
            sdmx = np.sign(d_mid_x)
            sdmy = np.sign(d_mid_y)
            if sdmx == 0:
                sdmx = 1
            if sdmy == 0:
                sdmy = 1

            padding = ObstacleAvoidanceModule.ranges[-1]

            state = np.empty([len(Simulation.search_space)+1,])
            
            # state[0] = np.round(d_mid_x - sdmx*0.5*width)
            # state[1] = np.round(d_mid_y - sdmy*0.5*height)
            state[0] = d_mid_x - sdmx*0.5*width
            state[1] = d_mid_y - sdmy*0.5*height
            
            # print('state')
            # print(state)

            if abs(state[0]) < 10 or abs(state[1]) < 10:
                state[0] = Module.base_round(state[0],2,1)
                state[1] = Module.base_round(state[1],2,1)
            elif abs(state[0]) < 20 or abs(state[1]) < 20:
                state[0] = Module.base_round(state[0],2,5)
                state[1] = Module.base_round(state[1],2,5)
            else:
                state[0] = Module.base_round(state[0],2,10)
                state[1] = Module.base_round(state[1],2,10)
            

            if agnt_x >= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[2] = 1
            elif agnt_x <= obs_ctr_x and agnt_y >= obs_ctr_y:
                state[2] = 2
            elif agnt_x <= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[2] = 3
            elif agnt_x >= obs_ctr_x and agnt_y <= obs_ctr_y:
                state[2] = 4

            self.state_prime[i] = state
        # print('self.state_prime')
        # print(self.state_prime)


    def check_state_transition(self):
        transition = np.zeros((len(Simulation.obstacles),))
        for i in range(0,len(self.state)):
            if np.array_equal(self.state[i], self.state_prime[i]):
                if self.action == Action.STAY:
                    transition[i] = 1
            else:
                if self.action != Action.STAY:
                    transition[i] = 1

        self.state_transition = transition
        # print('self.state_transition')
        # print(self.state_transition)


    # Determine the reward for executing the action (not prime) in the state (not prime)
    #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        for i in range(0,len(Simulation.obstacles)):

            if not self.state_transition[i]:
                self.instant_reward[i] = 0
            else:
                obs_x = Simulation.obstacles[i][0]
                obs_y = Simulation.obstacles[i][1]
                agnt_x = self.parent_agent.position[0]
                agnt_y = self.parent_agent.position[1]
                width = Simulation.obstacles[i][2]
                height = Simulation.obstacles[i][3]
                
                obs_ctr_x = obs_x + width/2
                obs_ctr_y = obs_y + height/2

                d_mid_x = obs_x + 0.5*width - agnt_x
                d_mid_y = obs_y + 0.5*height - agnt_y
                sdmx = np.sign(d_mid_x)
                sdmy = np.sign(d_mid_y)

                if sdmx == 0:
                    sdmx = 1
                if sdmy == 0:
                    sdmy = 1    

                if (Simulation.RewardType == Reward.Tiered):
                    rewarded = False
                    for j in range(0,len(ObstacleAvoidanceModule.ranges)):
                        padding = ObstacleAvoidanceModule.ranges[j]
                        if (obs_x - padding < agnt_x and agnt_x < obs_x + width + padding and 
                            obs_y - padding < agnt_y and agnt_y < obs_y + height + padding):
                        
                            self.instant_reward[i] = ObstacleAvoidanceModule.rewards[j]
                            rewarded = True
                            break
                    if not rewarded:
                        self.instant_reward[i] = ObstacleAvoidanceModule.rewards[-1]

                elif (Simulation.RewardType == Reward.Continuous):
                    #use a large paddign to simplify logic for rinding fastest escape direction
                    padding = 10000.0
                    min_d_x = d_mid_x - sdmx*(0.5*width+padding)
                    min_d_y = d_mid_y - sdmy*(0.5*height+padding)

                    if(abs(min_d_x) < abs(min_d_y)): #fastest excape is in the x direction
                        self.instant_reward[i] = sdmx*self.state_prime[i,0] - 4
                    else: #fastest escape is in the y direction
                            self.instant_reward[i] = sdmy*self.state_prime[i,1] - 4
                        
                elif (Simulation.RewardType == Reward.Hybrid):
                    #use a large paddign to simplify logic for rinding fastest escape direction
                    padding = 10000.0
                    min_d_x = d_mid_x - sdmx*(0.5*width+padding)
                    min_d_y = d_mid_y - sdmy*(0.5*height+padding)

                    if(abs(min_d_x) < abs(min_d_y)): #fastest excape is in the x direction
                        if sdmx*self.state_prime[i,0] < 4: #agent is within range of the obstacle
                            self.instant_reward[i] = sdmx*self.state_prime[i,0] - 4
                        else: #agent is out of range of obstacle
                            self.instant_reward[i] = 0.0

                    else: #fastest escape is in the y direction
                        if sdmy*self.state_prime[i,1] < 4: #agent is within range of the obstacle
                            self.instant_reward[i] = sdmy*self.state_prime[i,1] - 4
                        else: #agent is out of range of obstacle
                            self.instant_reward[i] = 0.0

        # print(self.instant_reward)


    def get_module_weights(self):

        module_weights = np.zeros((len(self.Q),1))

        #find if within the largest range of any obstacle
        in_range = False
        for i in range(0,len(Simulation.obstacles)):
            obs_x = Simulation.obstacles[i][0]
            obs_y = Simulation.obstacles[i][1]
            agnt_x = self.parent_agent.position[0]
            agnt_y = self.parent_agent.position[1]
            width = Simulation.obstacles[i][2]
            height = Simulation.obstacles[i][3]

            padding = ObstacleAvoidanceModule.ranges[-1]+1

            if (obs_x - padding <= agnt_x and agnt_x <= obs_x + width + padding and
                obs_y - padding <= agnt_y and agnt_y <= obs_y + height + padding):
                module_weights[i] = 1
            else:
                module_weights[i] = 0

        return module_weights