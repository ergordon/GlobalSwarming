import numpy as np
from qlearning import Qlearning

from action import Action
import random
from scipy.stats import rv_discrete

import os.path
import sys
import pickle

import time

#Base class that all other modules should inherit from
class Module:
    alpha = 0.1 #learning rate (shared by all modules)

    #variables to be changed for each derived module
    gamma = 0 #discount rate
    

    #constructor
    def __init__(self,parent_agt):

        self.parent_agent = parent_agt #the agent that created and is storing this module instance
        self.tracked_agents = [] #list of agents being tracked by this module 
        self.instant_reward = [] #list of instantaneous rewards earned by the agent. 
        
        self.alpha = 0.1


    def start_tracking(self,agt):
        self.tracked_agents.append(agt)

    def update_total_reward(self):
        reward = 0
        # for i in range(0,len(self.instant_reward)):
        #     reward = reward + self.instant_reward[i]
        reward = reward + self.instant_reward

        self.parent_agent.add_total_reward(reward)

    


#module to make the agents swarm together
class CohesionModule(Module):
    

    #rewards for being within (or out of) range. 1st entry is the reward 
    # for being within the range specified by the first entry in ranges_squared
    #the last entry is the reward (punishment) for being out of range
    rewards = [2,1,0,-1,-2] 
    #the discrete ranges at which the agent can collect rewards
    ranges_squared = [8,18,32,50]



    def __init__(self,parent_agt):
        super().__init__(parent_agt)
        
        self.state = np.array([]) #the vector from the agent to the centroid of it and the tracked agents 
        self.state_prime = np.array([]) #same as state but for the next step. used for qlearning before assigning to state
        self.Q = Qlearning()    #define a Qleaning object for each module instance        
        self.q_filename = 'cohesion_q.pkl'

        # check if a q table already exists and load it if it does
        if(os.path.isfile(self.q_filename)):
            print("Q learining data found for table, loading it now")
            
            with open('cohesion_q.pkl', 'rb') as f:
                q_table, q_states = pickle.load(f)

            self.Q.q_table = q_table
            self.Q.q_states = q_states

        self.init_time = time.time()
        self.greedy_rise_time = 120 #in seconds TODO change the name of this

        self.action = Action.STAY
        self.action_prime = Action.STAY
        self.gamma = 0.01


    def update_q(self):
        # for i in range(0,len(self.tracked_agents)):
        #     self.Q.update_q(self.state,self.state_prime,self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward)
        self.Q.update_q(self.state,self.state_prime,self.action,self.action_prime,self.alpha,self.gamma,self.instant_reward)
        
    def update_state(self):
        #find the centroid
        centroid = np.array(self.parent_agent.position)
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 
        
        centroid = centroid / (len(self.tracked_agents)+1)
        # print("swarm centroid is:")
        # print(centroid)
        self.state = np.round(centroid - self.parent_agent.position,0) #round to whole numbers for discretization
        # print("Agent state is:")
        # print(self.state) 
    
    def update_state_prime(self):
        #find the centroid
        centroid = self.parent_agent.position
        for i in range(0,len(self.tracked_agents)):
            centroid = centroid + self.tracked_agents[i].position 

        centroid = centroid / (len(self.tracked_agents)+1)
        self.state_prime = np.round(centroid - self.parent_agent.position, 0) #round to whole numbers for discretization
        # print('state prime is')
        # print(self.state_prime)

    #there is a reward for each state
    #there is only one state for the cohesion module so it is a single number 
    def update_instant_reward(self):
        
        #the state is the vector to the swarm centroid
        #use distance squared for range comparisons
        dist_squared = 0
        for i in range(0,len(self.state_prime)):
            dist_squared = dist_squared + self.state_prime[i]**2
        

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
        self.instant_reward = 2 - .1*dist_squared



    #softmax porabability function to select next action for this module
    def select_next_action(self):
        
        action_weights = np.zeros(len(Action))#, deftype='f')
        
        for i in range (0,len(Action)):
            # print("fetching q row")
            Qrow = self.Q.fetch_row_by_state(self.state) 
            # print("q row is")
            # print(Qrow)
            # print("Q is")
            Qval = Qrow[i]
            # print(Q)

            #exploitation vs exploration constant
            #big T encourages exploration
            #small T encourages exploitation
            T = 1
            curr_time = time.time()
            if(curr_time - self.init_time < self.greedy_rise_time):
                T = 1000.0 - (1000.0-0.1)*(curr_time - self.init_time)/self.greedy_rise_time
            else:
                T = 0.1

            action_weights[i] = np.exp(Qval/T)
            # print(self.Q.fetch_row_by_state(self.state)[self.action])
            # action_weights[i] = self.Q.fetch_row_by_state(self.state)[self.action]
            

        if(np.sum(action_weights) != 0):
            action_weights = action_weights / np.sum(action_weights)

        # print("action_weights are")
        # print(action_weights)
        # print("arg max is ")
        # print(np.argmax(action_weights))
        # print("action is ")
        # print(Action(np.argmax(action_weights)))
        # self.action_prime = Action(np.argmax(action_weights))
    

        x=list(map(int,Action))
        px=action_weights

        # print('x is')
        # print(x)
        # print('action weiths are')
        # print(action_weights)

        sample=rv_discrete(values=(x,px)).rvs(size=1)
        # print('sample is')
        # print(sample)

        
        self.action_prime = Action(sample)

        # best_actions = np.argwhere(action_weights == np.amax(action_weights))
        # print('index of best actions are: ')
        # print(best_actions)
        # self.action_prime = Action(random.choice(best_actions))
        # print('selected action is: ')
        # print(self.action_prime)

#module to prevent agents from hitting each other
class CollisionModule(Module):
    
    gamma = 0

    def __init__(self):
        super().__init__()
        print(CollisionModule.gamma)

