from agent import Agent
import numpy as np
from action import Action
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import sys
import pickle
import os.path


def checkInBounds(position,bounds):
    #TODO make sure position and bound have same number of 
    
    for i in range(0,len(position)):
        if not ( bounds[i][0] <= position[i] <= bounds[i][1]):
            return False

    return True

def ReinitializeAgents(agents,bounds):
    #initialize agent positions
    for i in range(0,len(agents)):
        agents[i].position = np.array([i,i], dtype='f')

    #initialize module state parameters
    for i in range(0,num_agents):
        #loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = agents[i].modules[m].state


##############################################################################
#   Simulation Variables
##############################################################################


num_agents = 4 #number of agents to simulate

num_episodes = 10 #number of times to run the training scenario
episode_length = 200 #number of timesteps in each traning scenario

#bounds to initialize the agents inside of
init_space = [[-0,10],
             [0,10]]
search_space = [[-5,15],
                [-5,15]]

visualize = True


##############################################################################
#   Initialization
##############################################################################

agents = list() #list of agents

if(os.path.isfile('agents.pkl')):
            print("Q learining data found, loading it now")
            
            with open('agents.pkl', 'rb') as f:
                agents = pickle.load(f)

else:
    #initialize agent positions
    for i in range(0,num_agents):
        position = np.array([i,i], dtype='f')
        agents.append(Agent(position))
        print(agents[i].position)

    #initialize module paramters such as who each agent is tracking
    #TODO make it so the tracked agents are based on range and upated every iteration
    for i in range(0,num_agents):
        for j in range(0,num_agents):
            if(i != j):
                #loop through each module
                for m in range(0,len(agents[i].modules)):
                    agents[i].modules[m].start_tracking(agents[j])

    #initialize module state parameters
    for i in range(0,num_agents):
        #loop through each module
        for m in range(0,len(agents[i].modules)):
            agents[i].modules[m].update_state()
            agents[i].modules[m].state_prime = agents[i].modules[m].state



# agents = list() #list of agents

# #initialize agent positions
# for i in range(0,num_agents):
#     position = np.array([i,i], dtype='f')
#     agents.append(Agent(position))
#     print(agents[i].position)

# #initialize module paramters such as who each agent is tracking
# #TODO make it so the tracked agents are based on range and upated every iteration
# for i in range(0,num_agents):
#     for j in range(0,num_agents):
#         if(i != j):
#            #loop through each module
#            for m in range(0,len(agents[i].modules)):
#                 agents[i].modules[m].start_tracking(agents[j])

# #initialize module state parameters
# for i in range(0,num_agents):
#     #loop through each module
#     for m in range(0,len(agents[i].modules)):
#         agents[i].modules[m].update_state()
#         agents[i].modules[m].state_prime = agents[i].modules[m].state

##############################################################################
#   main algorithm
##############################################################################
if(visualize):
    frame_rate = 10
    axis_bounds = [search_space[0][0], search_space[0][1], search_space[1][0], search_space[1][1]]
    print(axis_bounds) 
    plt.axis(axis_bounds)
    plt.draw()
    plt.pause(1/frame_rate)
    plt.clf()
    plt.cla()

for e in range(0,num_episodes):
    print("episode begin")

    for t in range(0,episode_length):
        agent_out_of_bounds = False
        # print("timestep begin")
        # print("agents begin taking actions")
        for agnt in agents:
            #take the action determined in the last step
            #update agent positions on plots
            #TODO use action across multiple modules
            if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                # print('agent position is')
                # print(agnt.position)
                print('agent action is')
                print(agnt.modules[0].action)

            agnt.take_action(agnt.modules[0].action)
            
            # if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                # print('agent updated position is')
                # print(agnt.position)


            #check if any agent went out of search space.
            #terminate episode if so
            if not (checkInBounds(agnt.position,search_space)):
                print("agent left space")
                agent_out_of_bounds = True

            if(visualize):
                plt.plot(agnt.position[0],agnt.position[1],'ro')
                plt.axis(axis_bounds)

            


        # if(agent_out_of_bounds):
        #     break

    
        # print("agents begin updating modules")
        # for agnt in agents:
            for mod in agnt.modules:
                if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                    print("selecting next action")

                mod.select_next_action()
                
                if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                    print('next action is')
                    print(mod.action_prime)

                    print('Updating state prime!')
                #update state_prime
                mod.update_state_prime()
                
                if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                    print('state is')
                    print(mod.state)
                    print('state prime is')
                    print(mod.state_prime)

                    print('Updating instant reward!')                   
                #update instantaneous rewards
                mod.update_instant_reward()
                
                if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                
                    print('instant reward is')
                    print(mod.instant_reward)
                
                    print('Updating total reward!')
                #update the total reward 
                mod.update_total_reward()
                
                if(np.array_equal(agnt.modules[0].state, np.array([-2,-2]))):
                
                    print('total reward is')
                    print(mod.parent_agent.total_reward)

                    print('Updating Q!')
                # print(mod.action)
                # print(mod.action_prime)
                #update Q
                mod.update_q()

                mod.action = mod.action_prime
                mod.state  = mod.state_prime


        if(agent_out_of_bounds):
                break
 
        if(visualize):
            plt.draw()
            plt.pause(1/frame_rate)
            plt.clf()
            plt.cla()


    # print('Qstates')
    # print(agents[0].modules[0].Q.q_states)
    # print('Qtable')
    # print(agents[0].modules[0].Q.q_table)

    ReinitializeAgents(agents,init_space)

    # for agnt in agents:
    #     for mod in agnt.modules:
    #         print(mod.Q.q_states)    

    # for mod in agents[0].modules:       
    #     with open(mod.q_filename,'wb') as f:
    #         pickle.dump([mod.Q.q_table,mod.Q.q_states],f)

    agent_filename = 'agents.pkl'
    with open(agent_filename,'wb') as f:
        pickle.dump(agents,f)


    #     np.savetxt('q_table.txt', mod.Q.q_table)
    #     np.savetxt('q_states.txt', mod.Q.q_states)
    
    

