from agent import Agent
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import pickle
import os.path
import copy as cp

##############################################################################
#   data 
##############################################################################

agent_reward_filename = 'agent_rewards.pkl'
agent_rewards = np.array ([])

with open(agent_reward_filename, 'rb') as f:
    agent_rewards = pickle.load(f)


iterations = agent_rewards[0]
agent_rewards = agent_rewards[1]

num_agents = agent_rewards.shape[1]
num_episodes = iterations[-1]+1
#store the iterations and total rewards for each agent for each episode
iterations = np.arange(num_episodes)

#close the visualization plot and create a new plot of each agents total reward over time
f1 = plt.figure(1)
for i in range(0,num_agents):
    plt.plot(iterations,agent_rewards[:,i])

max_reward = np.array([])
min_reward = np.array([])
avg_reward = np.array([])

for i in range(0,num_episodes):
    reward_set = np.array([])
    for j in range(0,num_agents):
        reward_set = np.append(reward_set, agent_rewards[i][j])

    max_reward = np.append(max_reward,np.amax(reward_set))
    min_reward = np.append(min_reward,np.amin(reward_set))
    avg_reward = np.append(avg_reward,np.average(reward_set))
    
f2 = plt.figure(2)
plt.plot(iterations,max_reward,linestyle='--',color='grey')
plt.plot(iterations,min_reward,linestyle='--',color='grey')
plt.plot(iterations,avg_reward)

plt.fill_between(iterations,max_reward,min_reward,color='blue',alpha='0.05')


plt.show()



##############################################################################
#   data
##############################################################################
  
    
