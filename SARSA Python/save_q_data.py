import numpy as np
import module as module
import agent as agent
import pickle


#load the agents saved during the training process 
with open('agents.pkl', 'rb') as f:
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

    #iterate over each agent
    for j in range(0,num_agents):
        #iterage over each state for this module for this state
        for k in range(0, agents[j].modules[i].Q.q_states.shape[0]):

            working_state = agents[j].modules[i].Q.q_states[k] #the current state being compared
            working_q_row = agents[j].modules[i].Q.q_table[k] #the corresponding qtable entry to the current state

            #check for any entries in the current q_table for this module
            if q_states.shape[0] != 0:
                
                #TODO: implement this in the qlearning class for row fetches
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

    
    # #sanity check for duplicate q state entries
    # for d in range(0,q_states.shape[0]):
    #     test_state = q_states[d]
    #     for e in range(0,q_states.shape[0]):
    #         if d != e:
    #             if np.equal(q_states[e],test_state).all():
    #                 print('duplicate found oh no!!!!')
    #                 print(test_state)
    #                 print(q_states[e])

    #average the q rows based on the number of times they were updated
    for j in range(0,q_states.shape[0]):
        q_table[j] = np.divide(q_table[j],number_experienced[j])

    #store the results in lists
    module_names.append(agents[0].modules[i].__class__.__name__)
    tables.append(q_table)
    states.append(q_states)

#save the results to disk
save_data_filename = 'training_data.pkl'
with open(save_data_filename,'wb') as f:
    pickle.dump([module_names, tables,states],f)  
