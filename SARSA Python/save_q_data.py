import numpy as np
import module as module
import agent as agent
import pickle
import argparse
import os.path
import time
import warnings

import collections
import numpy as np
import scipy.stats as stat
from scipy.stats import iqr



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



def sd_outlier(x, axis = None, bar = 1.5, side = 'both'):

    if np.all(x == x[0]):
        return np.zeros((len(x),), dtype=bool)

    assert side in ['gt', 'lt', 'both'], 'Side should be `gt`, `lt` or `both`.'
    
    d_z = stat.zscore(x, axis = axis)

    if side == 'gt':
        return d_z > bar
    elif side == 'lt':
        return d_z < -bar
    elif side == 'both':
        return np.abs(d_z) > bar



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

    if(agents[0].modules[i].collapsable_Q):

        q_data = np.empty((1,), dtype=object)
        q_updates = np.empty((1,), dtype=object)
        q_data_dict = {} # temporary storage for the q table for this module
        q_data_filtering_dict = {}
        q_updates_dict = {} 
        q_updates_sum = {}

        for a in range(0,len(agents)):
            for q in range(0,len(agents[a].modules[i].Q)):
                Q = agents[a].modules[i].Q[q]
                
                for working_state, working_q_row in Q.q_data.items():
                    working_updates = min(Q.q_updates[working_state],max_updates)

                    if working_updates != 0:

                        if working_state in q_data_filtering_dict:
                            for u in range(0,working_updates):
                                q_data_filtering_dict[working_state].append(working_q_row)
                            
                            q_updates_dict[working_state] = max(q_updates_dict[working_state],working_updates) #TODO consider other ways of doing this

                        else:
                            q_row_list = []
                            

                            for u in range(0,working_updates):
                                q_row_list.append(working_q_row)
                            q_data_filtering_dict.update({working_state:q_row_list})
                            q_updates_dict.update({working_state:working_updates})
        

        for working_state, working_q_row_list in q_data_filtering_dict.items():
            filtered_q_row = np.zeros((len(working_q_row_list[0]),)) 
            for a in range(0,len(working_q_row_list[0])):
                
                #collect all of the q_row values that occupy index a into a np array so it can be filtered
                q_column = np.zeros((len(working_q_row_list),)) 
                for l in range(0,len(working_q_row_list)):
                    q_column[l] = working_q_row_list[l][a]
                
                
                outlier_bools = sd_outlier(q_column)
                filtered_column = np.extract(np.logical_not(outlier_bools), q_column)
                filtered_q_row[a] = np.mean(filtered_column)
                    
            q_data_dict.update({working_state:filtered_q_row})
    
        q_data[0] = q_data_dict
        q_updates[0] = q_updates_dict

    else: #Q not collapsable
        q_data = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        q_updates = np.empty((len(agents[0].modules[i].Q),), dtype=object)
        
        for q in range(0,len(agents[0].modules[i].Q)):

            q_data_dict = {} # temporary storage for the q table for this module
            q_data_filtering_dict = {}
            q_updates_dict = {} 
            
            for agnt in agents:
                Q = agnt.modules[i].Q[q]
                
                for working_state, working_q_row in Q.q_data.items():
                    working_updates = min(Q.q_updates[working_state],max_updates)

                    if working_updates != 0:
                        if working_state in q_data_filtering_dict:
                            for u in range(0,working_updates):  
                                q_data_filtering_dict[working_state].append(working_q_row)
                            q_updates_dict[working_state] = max(q_updates_dict[working_state],working_updates)
                        else:
                            q_row_list = []
                            for u in range(0,working_updates):
                                q_row_list.append(working_q_row)
                            q_data_filtering_dict.update({working_state:q_row_list})
                            q_updates_dict.update({working_state:working_updates})

            for working_state, working_q_row_list in q_data_filtering_dict.items():
                filtered_q_row = np.zeros((len(working_q_row_list[0]),)) 
                for a in range(0,len(working_q_row_list[0])):
                    
                    #collect all of the q_row values that occupy index a into a np array so it can be filtered
                    q_column = np.zeros((len(working_q_row_list),)) 
                    for l in range(0,len(working_q_row_list)):
                        q_column[l] = working_q_row_list[l][a]
                    
                    outlier_bools = sd_outlier(q_column)
                    filtered_column = np.extract(np.logical_not(outlier_bools), q_column)
                    filtered_q_row[a] = np.mean(filtered_column)

                q_data_dict.update({working_state:filtered_q_row})

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

for i in range(0,len(data)):
    num_states = 0
    for j in range(0,len(data[i])):
        # for k in range data[i][j]:
        num_states = num_states + len(data[i][j])

    print(module_names[i] + ' found ' + str(num_states) + ' states')