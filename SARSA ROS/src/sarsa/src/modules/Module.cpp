#include "Module.h"
#include <map>
#include <iostream>
#include <json/json.h>
#include <fstream> 
#include <sstream>
#include <typeinfo>
#define quote(x) #x

namespace sarsa_ros{

    Module::Module(){
        
    }

    Module::~Module(){}

    void Module::UpdateState(){
        std::cout << "Module base class UpdateState() called, implement in derived class and call by pointer." << std::endl;
    };

    std::vector<double> Module::GetActionWeights(){

        std::cout << "module is getting action weights" << std::endl;

        std::vector<double> action_weights(5); //TODO change to length of action enumerator
        int Qindex = 0;
        auto s_it = state.begin();
        for(auto q_it = Q.begin(); q_it != Q.end(); q_it++)
        {
            QLearning q_obj = *q_it; 
            std::vector<double> s_vect = *s_it;
            std::vector<double> aw = q_obj.FetchRowByState(s_vect);
            
            std::cout << "fetched weights are: ";
            for(size_t i=0; i<aw.size(); i++){
                std::cout << aw[i] << ", ";
            }
            std::cout << std::endl;


            for(size_t i=0; i<action_weights.size(); i++){
                action_weights[i] += aw[i];
            }

            s_it++;
        }

        std::cout << "now action weights are: ";
        for(size_t i=0; i<action_weights.size(); i++){
            std::cout << action_weights[i] << ", ";
        }
        std::cout << std::endl;

        // std::cout << "Qsize is"

        // for(size_t i=0; i<action_weights.size(); i++){
        //     action_weights[i] /= (double)Q.size(); //TODO cast to double???
        //     //should be dividing by number of Q?
        // }

        std::cout << "Q size is: " << Q.size() << std::endl;

        // std::cout << "now action weights are: ";
        // for(size_t i=0; i<action_weights.size(); i++){
        //     std::cout << action_weights[i] << ", ";
        // }
        // std::cout << std::endl;

        //TODO have some +/- inifinity checks before return?

        return action_weights;

    }

    void Module::LoadQData(){

        
        //TODO get filename from class name...
        std::cout << "file path is: " << training_filepath << std::endl;

        std::cout << "loading json from disk" << std::endl;
        
        Json::Value training_json;
        Json::Reader reader;
        
        //TODO consider a try catch block???
        std::ifstream json_file(training_filepath);
        if(!json_file){
            std::cout << "json file is not open" << std::endl;
        }
        
        if(json_file.peek() == std::ifstream::traits_type::eof()){
            std::cout << "json file is empty" << std::endl;
        }
        

        std::stringstream buffer;
        buffer << json_file.rdbuf();
        std::string json_string = buffer.str();
        
        //parse the json

        if(!reader.parse(json_string, training_json)) {
            std::cout << reader.getFormattedErrorMessages() << std::endl;
        }else{
            std::cout << "json successfully parsed, working with it" << std::endl;

            //store the maps in a list

            for(size_t j=0; j<training_json["data"].size(); j++){
                //create a map for each
                std::map<std::vector<double>, std::vector<double>> Q_data;

                for(size_t k=0; k<training_json["data"][(int)j].size(); k++){

                    Json::Value key_value_pair = training_json["data"][(int)j][(int)k];

                    // std::cout << "state is: ";
                    std::vector<double> temp_state;
                    for(size_t m=0; m<key_value_pair["key"].size(); m++){
                        temp_state.push_back(key_value_pair["key"][(int)m].asDouble());
                        // std::cout << temp_state[(int)m] << ", ";
                    }
                    // std::cout << std::endl;


                    // std::cout << "q row is: ";
                    std::vector<double> temp_Q_row;
                    for(size_t m=0; m<key_value_pair["value"].size(); m++){
                        temp_Q_row.push_back(key_value_pair["value"][(int)m].asDouble());
                        // std::cout << temp_Q_row[(int)m] << ", ";
                    }
                    // std::cout << std::endl;

                    // std::cout << "json pair is: " <<  training_json["data"][(int)j][(int)k] << std::endl;

                    Q_data.insert({temp_state,temp_Q_row});


                }

                //add Q_data to list
                Q.push_back(Q_data);

            }
            
        }

    }

}