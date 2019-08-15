import numpy as np
import module as module
import agent as agent
import pickle
import argparse
import os.path
import time
import warnings

warnings.filterwarnings("error")
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
data = [] #list of combined q data
updates = [] #list of combined q updates

num_agents = len(agents)

max_updates = 50 #assuming that the agent is fully trained for a given state after updating it this many times

#iterate over each module, combining the q tables and q states form each agent
#into a single q table and q states array
for i in range(0,len(agents[0].modules)):

    q_data = np.empty((1,), dtype=object)
    q_updates = np.empty((1,), dtype=object)
    q_data_dict = {} # temporary storage for the q table for this module
    q_updates_dict = {} 
    q_updates_sum = {}

    if(agents[0].modules[i].collapsable_Q):
        
        #directly copy first agent's data.
        #iterage over each state for this module for this state
        for a in range(0,len(agents)):
            for q in range(0,len(agents[a].modules[i].Q)):
                Q = agents[a].modules[i].Q[q]

                for working_state, working_q_row in Q.q_data.items():
                    working_updates = min(Q.q_updates[working_state],max_updates)
                    
                    if working_state in q_data_dict:
                        q_data_dict[working_state] = q_data_dict[working_state] + working_q_row*working_updates
                        #it'd be nice to increment/decrement the updates count based on the relative sizes
                        q_updates_dict[working_state] = max(q_updates_dict[working_state],working_updates) #TODO consider other ways of doing this
                        q_updates_sum[working_state] = q_updates_sum[working_state] + working_updates 
                    else:
                        q_data_dict.update({working_state:working_q_row*working_updates})
                        q_updates_dict.update({working_state:working_updates})
                        q_updates_sum.update({working_state:working_updates})
        
        #average the q rows based on the number of times they were updated
        for working_state in q_data_dict:
            if q_updates_sum[working_state] != 0:
                q_data_dict[working_state] = q_data_dict[working_state]/q_updates_sum[working_state]
            
        q_data[0] = q_data_dict
        q_updates[0] = q_updates_dict


    else: #Q not collapsable
        q_data = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        q_updates = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        
        
        for q in range(0,len(agents[0].modules[i].Q)):

            q_data_dict = {} # temporary storage for the q table for this module
            q_updates_dict = {} 
            q_updates_sum = {}
            
            #handle 1st agent
            Q = agents[0].modules[i].Q[q]
            # for j in range(0, Q.q_states.shape[0]):
            for working_state, working_q_row in Q.q_data.items():
                working_updates = min(Q.q_updates[working_state],max_updates)
                q_data_dict.update({working_state:working_q_row*working_updates})
                q_updates_dict.update({working_state:working_updates})
                q_updates_sum.update({working_state:working_updates})
                

            #then handle the rest
            for agnt in agents[1:]:
                Q = agnt.modules[i].Q[q]
                
                for working_state, working_q_row in Q.q_data.items():
                    
                    working_updates = min(Q.q_updates[working_state],max_updates)

                    if working_state in q_data_dict:
                        q_data_dict[working_state] = q_data_dict[working_state] + working_q_row*working_updates
                        q_updates_dict[working_state] = max(q_updates_dict[working_state],working_updates) #TODO consider other ways of doing this
                        q_updates_sum[working_state] = q_updates_sum[working_state] + working_updates 
                    else:
                        q_data_dict.update({working_state:working_q_row*working_updates})
                        q_updates_dict.update({working_state:working_updates})
                        q_updates_sum.update({working_state:working_updates})

            #average the q rows based on the number of times they were updated
            for working_state in q_data_dict:
                if q_updates_sum[working_state] != 0:
                    q_data_dict[working_state] = q_data_dict[working_state]/q_updates_sum[working_state]
            
            q_data[q] = q_data_dict
            q_updates[q] = q_updates_dict

    #store the results in lists
    module_names.append(agents[0].modules[i].__class__.__name__)
    data.append(q_data)
    updates.append(q_updates)

for i in range(0,len(module_names)):
    save_data_filename = module_names[i] + '_training_data.pkl'
    with open(os.path.join(path, save_data_filename),'wb') as f:
        pickle.dump([module_names[i], data[i], updates[i]],f)  

end = time.time()
duration = end - start

print('operation took')
print(duration)

print("Currently found "+str(len(data[0][0]))+" states")