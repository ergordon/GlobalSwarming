import numpy as np
from simulation import Simulation
from simulation import TargetPath
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
        self.alpha = 0.7                # Learning rate. keep in range [0,1]. can be tuned to affect Q learning
        self.gamma = 0                  # Discount factor

        self.state = np.array([])       # The vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) # Same as state but for the next step. used for Q-learning before assigning to state
        self.Q = np.array([])           # Define a Q-learning object for each module instance        

        self.collapsable_Q = False      # Whether or now the Q table array can be collapsed/combined into a single Q table
        self.init_time = time.time()    # Store the time at which the agent was initialized
        self.action = Action.STAY       # Safest not to do anything for first action
        self.action_prime = Action.STAY # Safest not to do anything for first action
  
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
    def auxilariy_functions(self):
        pass

    # Get a set of action weights for this module to be used in conjuntion with those of other modules 
    #  with the purpose of selecting a single action for the agent to perform 
    def get_action_weights(self):
        # print('getting action weights')
        # Create a set of weights for each action
        action_weights = np.zeros(len(Action))
        num_updates = 0
        # Sum the action tables for every tracked agent
        # TODO think about if something like importance functions should be implemented here
        for i in range (0,len(self.Q)):
            # print('state_p is ' + str(self.state_prime[i]))
            #TODO should i multiply by q_updates?
            action_weights = action_weights + self.Q[i].fetch_row_by_state(self.state_prime[i])
            
            #TODO think of a better way to do this......
            num_updates = num_updates + self.Q[i].fetch_updates_by_state(self.state_prime[i])
        num_updates = num_updates / len(self.Q) #TODO would this work better as min(q_updates)?

        # print('Q table entry is ' + str(action_weights))
        # For each possible agent action
        for i in range (0,len(action_weights)):
            # Get the appropiate Q value Q table row corresponding to the current state 
            #  and the action being iterated over
            Qval = action_weights[i]

            max_updates = 10

            T = 1 
            if num_updates < max_updates:
                T = 1000.0 - (1000.0-0.005)*num_updates/max_updates
            else:
                T = 0.005 

            # print("T is: " + str(T))

            # Exploitation vs exploration constant
            #  Big T encourages exploration
            #  Small T encourages exploitation
            # T = 1
            # # NOTE: Linearly change T to decrease exploration and increase exploitation over time
            # curr_time = time.time()
            # if(curr_time - self.init_time < Simulation.exploitation_rise_time):
            #     T = 1000.0 - (1000.0-0.05)*(curr_time - self.init_time)/Simulation.exploitation_rise_time
            # else:
            #     T = 0.05
            # T = 1
            # if Simulation.episode_iter_num/Simulation.num_episodes*100 <= Simulation.exploitation_rise_percent :
            #     T = 1000.0 - (1000.0-0.05)*Simulation.episode_iter_num/Simulation.num_episodes
            # else:
            #     T = 0.05

            
            # Calculate the weight for this action
            action_weights[i] = np.exp(Qval/T)
            
            # Set the weight to the max float size in case it is beyond pythons max float size
            if(action_weights[i] == float('inf')):
                action_weights[i] = 1.7976931348623157e+308
            
        # print('action_weights are')
        # print(action_weights)
        return action_weights


    # Update the Q table for this module
    def update_q(self):
        # Accessed through the Qlearning object
        for i in range(0,len(self.Q)):
            self.Q[i].update_q(self.state[i],self.state_prime[i],self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward[i])

    # Update parent agents total reward based on the module's current instant reward
    def update_total_reward(self):
        reward = sum(self.instant_reward)
        self.parent_agent.add_total_reward(reward)

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
    ranges_squared = [64]

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) #Inherited class initialization
        
        self.gamma = 0.1                      # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        self.Q = np.empty((1,), dtype=object)
        self.Q[0] = Qlearning()
        self.collapsable_Q = True              # Whether or now the Q table array can be collapsed/combined into a single Q table

        self.state = np.zeros((1,len(Simulation.search_space)))
        self.state_prime = np.zeros((1,len(Simulation.search_space)))
        self.instant_reward = np.zeros((1,))

    # Update the state that the agent is currently in
    #  for this module, it is the vector pointing from the agent to a tracked agent
    #  there is a separate state stored for each tracked agent
    def update_state(self):
        # Find the centroid of self and all tracked agents
        centroid = np.array(self.parent_agent.position)
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
        centroid = centroid / (len(self.tracked_agents)+1)
        # Round to whole numbers for discretization
        self.state[0] = np.round(centroid - self.parent_agent.position,0) 

    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  for this module, it is the vector pointing from the agent to the swarm centroid
    # TODO: use the centroid of the agents within a defined range
    def update_state_prime(self):
        # Find the centroid
        centroid = self.parent_agent.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
        centroid = centroid / (len(self.tracked_agents)+1)
         # Round to whole numbers for discretization
        self.state_prime[0] = np.round(centroid - self.parent_agent.position, 0)

    # Determine the reward for executing the action (not prime) in the state (not prime)
    # Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        # The state is the vector to the swarm centroid
        # NOTE: Use distance squared for range comparisons (sqrt is slow)
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2
        
        # # Tiered reward scheme
        # #  Loop through each range to give the appropriate reward
        # rewarded = False
        # for i in range(0,len(CohesionModule.ranges_squared)):
        #     if dist_squared <= CohesionModule.ranges_squared[i]:
        #         self.instant_reward[0] = CohesionModule.rewards[i]
        #         rewarded = True    
        #         break
        
        # # Not in range, apply last reward (punishment)
        # if rewarded == False:
        #    self.instant_reward[0] = CohesionModule.rewards[-1]

        # continuous reward scheme
        self.instant_reward[0] = 2 - .1*dist_squared


    # Visualization for this module. 
    # Draw a transparent circle around the centroid 
    def visualize(self):
        super().visualize() #inherited class function
            
        centroid = self.parent_agent.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
        centroid = centroid / (len(self.tracked_agents)+1)
        
        # Set marker size to be the diameter of the range
        mkr_size = np.sqrt(CohesionModule.ranges_squared[-1])

        # Plot range circle, mkrsize is the radius.
        circle = plt.Circle((centroid[0],centroid[1]), mkr_size, color='green', alpha=0.1/Simulation.num_agents)
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.add_artist(circle)

    def get_module_weight(self):
        
        # First find the squared distance from the centroid
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        # Now pass into a weighting function
        # return -2/(1+np.exp(dist_squared/120)) + 1

        if dist_squared >= CohesionModule.ranges_squared[0]:
            return 1
        else:
            return 0

##############################################################################
#   Begin Collision Module Class
##############################################################################
class CollisionModule(Module):

    # Rewards for being within (or out of) range. 1st entry is the reward 
    #  for being within the range specified by the first entry in ranges_squared
    #  the last entry is the reward (punishment) for being out of range
    rewards = [-100,-1,0] 
    ranges_squared = [4,81] # The discrete ranges at which the agent can collect rewards

    #class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt)     # Inherited class initialization
        
        self.gamma = 0                   # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        #self.collision_count = 0        # Number of times this module has recorded a collision (with another agent) for this agent
        self.collapsable_Q = True        # Whether or now the Q table array can be collapsed/combined into a single Q table
          
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
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        for i in range(0,len(self.state_prime)):
            if(np.array_equal(self.state_prime[i],np.array([0,0]))):
                Simulation.agent_collision_count = Simulation.agent_collision_count + 1
                #print("Agent Collision "+str(Simulation.agent_collision_count))

    # Add an agent to the list of agents to be tracked by this module
    def start_tracking(self,agt):
        # First check if the agent is already being tracked
        if agt not in self.tracked_agents:
            if(len(self.tracked_agents) != 0):
                self.state = np.vstack([self.state,[0,0]])
                self.state_prime = np.vstack([self.state_prime,[0,0]])
                self.instant_reward = np.append(self.instant_reward,0)
                self.Q = np.append(self.Q, Qlearning())
            else:
                self.state = np.zeros((1,len(Simulation.search_space)))
                self.state_prime = np.zeros((1,len(Simulation.search_space)))
                self.instant_reward = np.zeros((1,))  
                self.Q = np.empty((1,), dtype=object)
                self.Q[0] = Qlearning()
        
        super().start_tracking(agt) 

    # Update the state that the agent is currently in
    #  for this module, it is the the set of vectors pointing from the agent to each other agent in the swarm
    def update_state(self):
        # Each state is the vector from the parent agent to the tracked agent
        for i in range(0,len(self.tracked_agents)):   
            # Round to whole numbers for discretization
            self.state[i] = np.round(np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position),0)


    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  for this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
        for i in range(0,len(self.tracked_agents)):  
            # Round to whole numbers for discretization  
            self.state_prime[i] = np.round(np.array(self.tracked_agents[i].position) - np.array(self.parent_agent.position),0)
   
    # Determine the reward for executing the action (not prime) in the state (not prime)
    # Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        
        for i in range(0,len(self.tracked_agents)):  
            
            # The state is the vector to the tracked agent
            # Use distance squared for range comparisons (sqrt is slow)
            dist_squared = 0
            for j in range(0,self.state_prime[i].shape[0]):
                dist_squared = dist_squared + self.state_prime[i,j]**2

            # Tiered reward scheme
            # Loop through each range to give the appropriate reward
            rewarded = False
            for k in range(0,len(CollisionModule.ranges_squared)):
                if dist_squared <= CollisionModule.ranges_squared[k]:
                    self.instant_reward[i] = CollisionModule.rewards[k]
                    rewarded = True    
                    break
            
            # Not in range, apply last reward (punishment)
            if rewarded == False:
                self.instant_reward[i] = CollisionModule.rewards[-1]


            # # continuous reward scheme that offeres severe punishements for being very close to other agents
            # # the function is always negative but is asymtotic to 0 as dist_squared approaches infinity
            # self.instant_reward[i] = 10.0*(dist_squared/(10.0+dist_squared)-1.0)

    def get_module_weight(self):
        
        min_dist_squared = 1.7976931348623157e+308

        # Iterate over each state, finding the shortest distance squared
        for i in range (0,len(self.tracked_agents)):
            dist_squared = 0
            for j in range(0,len(self.state_prime[i])):
                dist_squared = dist_squared + self.state_prime[i,j]**2

            # print(dist_squared)

            if dist_squared < min_dist_squared:
                min_dist_squared = dist_squared

        #TODO consider adding some padding. maybe the agent step size?
        if min_dist_squared <= CollisionModule.ranges_squared[-1]:
            return 1
        else:
             return 0
        
        # #now pass into a weighting function
        # if min_dist_squared == 0:
        #     return 1
        # else:
        #     return -1*(min_dist_squared/(1+min_dist_squared)-1)

    def reset_init(self,e):
        pass

##############################################################################
#   Begin Boundary Module Class
##############################################################################
class BoundaryModule(Module):

    rewards = [-1,0] # Discrete rewards
    ranges = [4]     # The discrete ranges at which the agent can collect rewards

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt) # Inherited class initialization
        
        self.gamma = 0                     # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        self.collision_count = 0           # Number of times this module has recorded a collision (with another agent) for this agent
    
        self.Q = np.empty((len(Simulation.search_space)*2,), dtype=object)
        for i in range(0,len(self.Q)):
            self.Q[i] = Qlearning()
        self.collapsable_Q = False # Whether or now the Q table array can be collapsed/combined into a single Q table
        
        self.state = np.zeros((len(Simulation.search_space)*2,1))
        self.state_prime = np.zeros((len(Simulation.search_space)*2,1))
        self.instant_reward = np.zeros(len(Simulation.search_space)*2)  
        
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

    # Track boundary collisions. 
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        pass
    
    # Update the state that the agent is currently in
    #  For this module, it is a vector containing distances from the agent to each boundary
    #  Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)
    def update_state(self):
        for i in range(0,len(Simulation.search_space)):   
            # Round to whole numbers for discretization
            self.state[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
            self.state[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])    
            
    # Update the state that agent is in. Store it in state_prime because it is called after 
    #  Executing an action and the Q object needs both the orignal state and the state after exectuion 
    #  For this module, it is the set of vectors pointing from the agent to each other tracked agent
    def update_state_prime(self):
        for i in range(0,len(Simulation.search_space)):   
            # Round to whole numbers for discretization
            self.state_prime[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
            self.state_prime[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])  

    # Each module 
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

    # Determine the reward for executing the action (not prime) in the state (not prime)
    #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
    def update_instant_reward(self):
        for i in range(0,len(Simulation.search_space)):  

            # Handle upper bounds
            if(self.state_prime[i*2] >= BoundaryModule.ranges[0]):
                self.instant_reward[i*2] = BoundaryModule.rewards[-1]
            else:
                self.instant_reward[i*2] = BoundaryModule.rewards[0]

            # Handle lower bounds
            if(self.state_prime[i*2+1] <= -BoundaryModule.ranges[0]):
                self.instant_reward[i*2+1] = BoundaryModule.rewards[-1]
            else:
                self.instant_reward[i*2+1] = BoundaryModule.rewards[0]

    def reset_init(self,e):
        pass

##############################################################################
#   Begin Target Seek Module Class
##############################################################################
class TargetSeekModule(Module):

    rewards = [10, -1]       # Discrete rewards for a given range
    ranges_squared = [100]    # The discrete ranges at which the agent can collect rewards

    # Class constructor
    def __init__(self,parent_agt):
        super().__init__(parent_agt)    # Inherited class initialization
        
        self.gamma = 0.2               # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        
        self.Q = np.empty((1,), dtype=object)
        self.Q[0] = Qlearning()
        self.collapsable_Q = True       # Whether or now the Q table array can be collapsed/combined into a single Q table

        self.state = np.zeros((1,len(Simulation.search_space)))
        self.state_prime = np.zeros((1,len(Simulation.search_space)))
        self.instant_reward = np.zeros((1,))

        self.in_target = False           # Bool for tracking if agent is in the target. False = not in target
        self.targets_entered = 0         # Number of targets entered for a single episode 
        
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
    def auxilariy_functions(self):
        super().auxilariy_functions() # Inherited class function

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
                arena_space = Simulation.arena_space
                if (Simulation.TargetType == TargetPath.Random):
                    Simulation.targets = np.array([random.randint(arena_space[0][0]+5, arena_space[0][1]-5),
                                    random.randint(arena_space[1][0]+5, arena_space[1][1]-5)])

                    Simulation.target_histogram_data.append([self.targets_entered, Simulation.episode_iter_num])
                    #print("Target Entered at "+str(time.time() - self.init_time)+" seconds")
                    #print("Target Entered at "+str(Simulation.episode_iter_num)+" iterations")
                elif (Simulation.TargetType == TargetPath.Planned):
                    #self.targets_entered = self.targets_entered + 1
                    if self.targets_entered <= len(Simulation.target_array)-1:
                        self.targets_entered = self.targets_entered + 1
                        if(self.targets_entered < len(Simulation.target_array)):
                            #self.targets_entered = self.targets_entered + 1
                            Simulation.targets = Simulation.target_array[self.targets_entered]
                            # print("Target Entered at "+str(time.time() - self.init_time)+" seconds")
                            # print("Target Entered at "+str(Simulation.episode_iter_num)+" iterations")
                            Simulation.target_histogram_data.append([self.targets_entered, Simulation.episode_iter_num])
                        else:
                            #self.targets_entered = self.targets_entered + 1
                            #if(self.targets_entered == len(Simulation.target_array)):
                            Simulation.target_histogram_data.append([self.targets_entered, Simulation.episode_iter_num])
                            
                    #print("Target Entered at "+str(time.time() - self.init_time)+" seconds")
                    #print("Target Entered at "+str(Simulation.episode_iter_num)+" iterations")
                    #Simulation.target_histogram_data.append([self.targets_entered, Simulation.episode_iter_num])

                Simulation.target_entries_count = 0
                Simulation.target_agents_remaining = Simulation.num_agents
        
    
    # Update the state that the agent is currently in
    #  For this module, it is the vector pointing from the agent to the target
    def update_state(self):
        # Round to whole numbers for discretization
        self.state[0] = np.round(Simulation.targets - self.parent_agent.position, 0) 
    
    # Update the state that agent is in. Store it in state_prime because it is called after 
    # Executing an action and the Q object needs both the original state and the state after execution 
    # For this module, it is the vector pointing from the agent to the target
    def update_state_prime(self):
        # NOTE: Round to whole numbers for discretization
        self.state_prime[0] = np.round(Simulation.targets - self.parent_agent.position, 0)


    # Determine the reward for executing the action (not prime) in the state (not prime)
    #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calculated based on state_prime
    def update_instant_reward(self):
        # The state is the vector to the swarm centroid
        #  Use distance squared for range comparisons (sqrt is slow)
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        # print(dist_squared)
        # Tiered reward scheme
        #  Loop through each range to give the appropriate reward
        rewarded = False
        for i in range(0,len(TargetSeekModule.ranges_squared)):
            if dist_squared <= TargetSeekModule.ranges_squared[i]:
                self.instant_reward[0] = TargetSeekModule.rewards[i]
                
                rewarded = True    
                break

        # # Not in range, apply last reward (punishment)
        if rewarded == False:
            #self.instant_reward[0] = TargetSeekModule.rewards[-1]
            #self.instant_reward[0] = -2.5*(-math.log(math.sqrt(dist_squared) + 10) + 5)
            self.instant_reward[0] = -math.log(dist_squared + 10) + 5
            #self.instant_reward[0] = -dist_squared
            #print(self.instant_reward[0])
            #self.instant_reward[0] = -dist_squared*0.02+10

        # print("instant reward is: " + str(self.instant_reward[0]))

    def get_module_weight(self):
        
        dist_squared = 0
        for i in range(0,len(self.state_prime[0])):
            dist_squared = dist_squared + self.state_prime[0,i]**2

        if dist_squared < TargetSeekModule.ranges_squared[-1]:
            return 0
        else:
            return 0.25
        
        # #now pass into a weighting function
        # return -2/(1+np.exp(dist_squared/4)) + 1



    # Reset the initialized variables at the end of an episode.
    def reset_init(self,e):
        self.in_target = False
        Simulation.target_reached_episode_end[e] = self.targets_entered
        self.targets_entered = 0
        
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
        
        self.gamma = 0.1             # Discount factor. keep in range [0,1]. can be tuned to affect Q learning
        self.collision_count = 0     # Number of times this module has recorded a collision (with another agent) for this agent
        self.collapsable_Q = True    # Whether or not the Q table array can be collapsed/combined into a single Q table

        self.Q = np.empty((len(Simulation.obstacles),), dtype=object)
        for i in range(0,len(Simulation.obstacles)):
            self.Q[i] = Qlearning()

        self.state = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+2))
        self.state_prime = np.zeros((len(Simulation.obstacles),len(Simulation.search_space)+2))
        self.instant_reward = np.zeros(len(Simulation.obstacles))

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
    def auxilariy_functions(self):
        super().auxilariy_functions() #inherited class function
        
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
            padding = ObstacleAvoidanceModule.ranges[-1]

            state = np.empty([len(Simulation.search_space)+2,])
            
            state[0] = np.round(d_mid_x - sdmx*0.5*width)
            state[1] = np.round(d_mid_y - sdmy*0.5*height)
            
            # QUESTION: Can these be combined into a single state?
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

    # Determine the reward for executing the action (not prime) in the state (not prime)
    #  Action (not prime) brings agent from state (not prime) to state_prime, and reward is calulated based on state_prime
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