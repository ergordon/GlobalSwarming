import numpy as np
import module as module
import agent as agent
import pickle
import argparse
import os.path
import time

start = time.time()


##############################################################################
#   Argument Parser
##############################################################################
# EXAMPLE: python plot_data.py --file agent_rewards_DistanceOnly.pkl
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--simName", type=str, default="SimulationResults", required=False,
	help="simName == Name of Simulation or Test")
args = vars(ap.parse_args())

##############################################################################
#   Data 
##############################################################################

path = args["simName"]
agents = np.array ([])

#load the agents saved during the training process 
with open(path+'/agents.pkl', 'rb') as f:
    agents = pickle.load(f)

module_names = [] #list of class names for each module
tables = [] #list of combined q tables
states = [] #list of combined q states

num_agents = len(agents)

#iterate over each module, combining the q tables and q states form each agent
#into a single q table and q states array
for i in range(0,len(agents[0].modules)):

    q_table = np.array([]) #temporary storage for the q table for this module
    q_states = np.array([]) #temporary storage for the q states for this module
    number_experienced = np.array([]) #the number of times each state shows up across all agents
    if(agents[0].modules[i].collapsable_Q):
        #iterate over each agent
        for agnt in agents:
            #iterage over each state for this module for this state
            for Q in agnt.modules[i].Q:
                for j in range(0, Q.q_states.shape[0]):

                    working_state = Q.q_states[j] #the current state being compared
                    working_q_row = Q.q_table[j] #the corresponding qtable entry to the current state

                    #check for any entries in the current q_table for this module
                    if q_states.shape[0] != 0:
                        
                        #check if the working state already exists in the q table for this module 
                        matches = np.equal(q_states,[working_state]).all(1).nonzero()

                        if matches[0].size == 0:
                            #working state not in q states for this module, add it along with the row
                            q_states = np.vstack([q_states, working_state])
                            q_table = np.vstack([q_table, working_q_row])
                            number_experienced = np.vstack([number_experienced, np.array([1])])
                        else:

                            #working state already in q states for this module, 
                            #sum the working q row with the corresponding entry in the q table for this module
                            #incerement the number of times this row has been updated
                            matching_index = matches[0][0] 
                            q_table[matching_index] = np.add(q_table[matching_index], working_q_row)
                            number_experienced[matching_index] = np.add(number_experienced[matching_index], np.array([1]))
                    
                    else: #q_states.shape[0] != 0:
                        #no entries found yet, initialize with current values
                        q_states = working_state
                        q_table = working_q_row
                        number_experienced = np.array([1])  

        

        #sanity check for duplicate q state entries
        for d in range(0,q_states.shape[0]):
            test_state = q_states[d]
            for e in range(0,q_states.shape[0]):
                if d != e:
                    if np.equal(q_states[e],test_state).all():
                        print('duplicate state found from single agent, there is most likely an error in the Qlearning class')
                        print(test_state)
                        print(q_states[e])

        #average the q rows based on the number of times they were updated
        for d in range(0,q_states.shape[0]):
            q_table[d] = np.divide(q_table[d],number_experienced[d])
    else:
        q_states = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        q_table = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        number_experienced = np.empty((len(agents[0].modules[i].Q),), dtype=object)

        for q in range(0,len(agents[0].modules[i].Q)):
            for agnt in agents:
                Q = agnt.modules[i].Q[q]
                for j in range(0, Q.q_states.shape[0]):
                    working_state = Q.q_states[j] #the current state being compared
                    working_q_row = Q.q_table[j] #the corresponding qtable entry to the current state


                    #check for any entries in the current q_table for this module for this Q
                    if q_states[q] is not None:
                        
                        #check if the working state already exists in the q table for this module 
                        matches = np.equal(q_states[q],[working_state]).all(1).nonzero()

                        if matches[0].size == 0:
                            #working state not in q states for this module, add it along with the row
                            q_states[q] = np.vstack([q_states[q], working_state])
                            q_table[q] = np.vstack([q_table[q], working_q_row])
                            number_experienced[q] = np.vstack([number_experienced[q], np.array([1])])
                        else:
                            #working state already in q states for this module, 
                            #sum the working q row with the corresponding entry in the q table for this module
                            #incerement the number of times this row has been updated
                            matching_index = matches[0][0] 
                            q_table[q][matching_index] = np.add(q_table[q][matching_index], working_q_row)
                            number_experienced[q][matching_index] = np.add(number_experienced[q][matching_index], np.array([1]))
                    
                    else: #q_states[q] is None
                        #no entries found yet, initialize with current values
                        q_states[q] = working_state
                        q_table[q] = working_q_row
                        number_experienced[q] = np.array([1])     

            # sanity check for duplicate q state entries
            for d in range(0,q_states[q].shape[0]):
                test_state = q_states[q][d]
                for e in range(0,q_states[q].shape[0]):
                    if d != e:
                        if np.equal(q_states[q][e],test_state).all():
                            print('duplicate state found from single agent, there is most likely an error in the Qlearning class')
                            print(test_state)
                            print(q_states[q][e])

            #average the q rows based on the number of times they were updated
            for d in range(0,q_states[q].shape[0]):
                q_table[q][d] = np.divide(q_table[q][d],number_experienced[q][d])

        print(q_table)

    #store the results in lists
    module_names.append(agents[0].modules[i].__class__.__name__)
    tables.append(q_table)
    states.append(q_states)

for i in range(0,len(module_names)):
    save_data_filename = module_names[i] + '_training_data.pkl'
    with open(os.path.join(path, save_data_filename),'wb') as f:
        pickle.dump([module_names[i], tables[i], states[i]],f)  

end = time.time()

duration = end - start

print('operation took')
print(duration)