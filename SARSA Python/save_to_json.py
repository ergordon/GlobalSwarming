import json
from os import listdir
from os.path import isfile, join
import argparse
import pickle
import numpy as np

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

files_in_path = [f for f in listdir(path) if isfile(join(path, f))]
print(files_in_path)
print("")

for i in range(0, len(files_in_path)):
    if "_training_data.pkl" in files_in_path[i]:
        print(files_in_path[i] + "found. Converting to json")

        with open(path + "/" + files_in_path[i], 'rb') as f:
            [module_name, data, updates] = pickle.load(f) 
            
            json_string = '{ "data" : ['
            # data = {'apple':'yummy','kimchi':'gross'}
            for j in range(0, len(data)):
                
                # dump the key:value pairs into a json file  
                values = [{"key": k, "value": v.tolist()} for k, v in data[j].items()]
                # print(values) 
                json_string = json_string + json.dumps(values)
                    
                if j < len(data) - 1:
                    json_string = json_string + ","

            json_string = json_string + "]}"
            # print(json_string)
            
            save_file = open(path+"/" + module_name + "_training_data.json","w")
            save_file.write(json_string)
            save_file.close()        



