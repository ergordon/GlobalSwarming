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

    q_table = np.array([])
    q_states = np.array([])
    q_table_dict = {} #temporary storage for the q table for this module
    number_experienced = {} #the number of times each state shows up across all agents

    if(agents[0].modules[i].collapsable_Q):
        #directly copy first agent's data.
        #iterage over each state for this module for this state
        for a in range(0,len(agents)):
            for q in range(0,len(agents[a].modules[i].Q)):
                Q = agents[a].modules[i].Q[q]

                for s in range(0,Q.q_states.shape[0]):

                    working_state = tuple(Q.q_states[s]) #the current state being compared
                    working_q_row = Q.q_table[s] #the corresponding qtable entry to the current state

                    if a == 0 and q == 0:
                        q_table_dict.update({working_state:working_q_row})
                        number_experienced.update({working_state:1})
                    else:
                        if working_state in q_table_dict:
                            q_table_dict[working_state] = q_table_dict[working_state] + working_q_row
                            number_experienced[working_state] = number_experienced[working_state] + 1 
                        else:
                            q_table_dict.update({working_state:working_q_row})
                            number_experienced.update({working_state:1})
        
        #average the q rows based on the number of times they were updated
        #then put back into numpy array because thats how the algorithm expects it
        for q_state in q_table_dict:
            q_table_dict[q_state] = q_table_dict[q_state]/number_experienced[q_state]
            #check for any entries in the current q_table for this module for this Q
            if q_states.shape[0] != 0:
                
                #working state not in q states for this module, add it along with the row
                q_states = np.vstack([q_states, np.asarray(q_state)])
                q_table = np.vstack([q_table, q_table_dict[q_state]])
                
            else: #q_states[q] is None
                #no entries found yet, initialize with current values
                q_states = np.asarray(q_state)
                q_table = q_table_dict[q_state]
        
    else: #Q not collapsable
        q_states = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        q_table = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        
        for q in range(0,len(agents[0].modules[i].Q)):

            q_table_dict = {} #temporary storage for the q table for this module
            number_experienced = {} #the number of times each state shows up across all agents

            
            #handle 1st agent
            Q = agents[0].modules[i].Q[q]
            for j in range(0, Q.q_states.shape[0]):
                working_state = tuple(Q.q_states[j]) #the current state being compared
                working_q_row = Q.q_table[j] #the corresponding qtable entry to the current state

                q_table_dict.update({working_state:working_q_row})
                number_experienced.update({working_state:1})

            #then handle the rest
            for agnt in agents[1:]:
                Q = agnt.modules[i].Q[q]
                # print(Q.q_table)
                for j in range(0, Q.q_states.shape[0]):
                    working_state = tuple(Q.q_states[j]) #the current state being compared
                    working_q_row = Q.q_table[j] #the corresponding qtable entry to the current state
                    # print(working_q_row)
                    if working_state in q_table_dict:
                        q_table_dict[working_state] = q_table_dict[working_state] + working_q_row
                        number_experienced[working_state] = number_experienced[working_state] + 1 
                    else:
                        q_table_dict.update({working_state:working_q_row})
                        number_experienced.update({working_state:1})

            #average the q rows based on the number of times they were updated
            #then put back into numpy array because thats how the algorithm expects it
            for q_state in q_table_dict:
                # print(q_table_dict[q_state])
                q_table_dict[q_state] = q_table_dict[q_state]/number_experienced[q_state]
                #check for any entries in the current q_table for this module for this Q
                if q_states[q] is not None:
                    
                    #working state not in q states for this module, add it along with the row
                    q_states[q] = np.vstack([q_states[q], np.asarray(q_state)])
                    q_table[q] = np.vstack([q_table[q], q_table_dict[q_state]])
                    
                else: #q_states[q] is None
                    #no entries found yet, initialize with current values
                    q_states[q] = np.asarray(q_state)
                    q_table[q] = q_table_dict[q_state]

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