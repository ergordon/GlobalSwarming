#include "SpeckBucciMMAS.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>


namespace sarsa_ros{

    
    SpeckBucciMMAS::SpeckBucciMMAS(std::list<Module*>& _modules, std::list<double>& weights) : MMAS(_modules), module_weights(weights){
        
        // PrintModuleWeights();

    }

    SpeckBucciMMAS::~SpeckBucciMMAS(){}


    void SpeckBucciMMAS::PrintModuleWeights(){
        std::cout << "contents of module_weights reference is: ";
        for (auto w_it = module_weights.begin(); w_it != module_weights.end(); w_it++){
            std::cout << *w_it << ", ";
        }
        std::cout << std::endl;
    }

    action SpeckBucciMMAS::SelectNextAction(){
        
        std::vector<double> action_weights(5);


        auto weight_it = module_weights.begin();    
        for (std::list<Module*>::iterator mod_it = modules.begin(); mod_it != modules.end(); mod_it++){
            
            //get that module's action weights
            std::vector<double> mod_aw = (*mod_it)->GetActionWeights();
            for(size_t i=0; i<mod_aw.size(); i++){
                action_weights[i] += mod_aw[i]*(*weight_it);
            }

            weight_it++;
        }


        std::cout << "action_weights are: ";
        for(size_t i=0; i<action_weights.size(); i++){
            std::cout << action_weights[i] << ", ";
        }
        std::cout << std::endl;


        //chose an action
        //first get largest number
        double q_max = *std::max_element(std::begin(action_weights), std::end(action_weights));
        std::cout << "max action weight is: " << q_max << std::endl;


        //here action weights should take on a different meaning....
        // //next get indices for all max occurences
        // std::vector<double> action_weights;
        // std::vector<int> max_indices; 
        // for(size_t i=0; i<Q_row.size(); i++){
        //     if(Q_row[i] == q_max){
        //         max_indices.push_back((int)i);
        //         action_weights.push_back(Q_row[i]);
        //     }
        // } 



        //next get indices for all max occurences
        std::vector<int> max_indices; 
        std::vector<double> max_weights;
        for(size_t i=0; i<action_weights.size(); i++){
            double tolerance = 0.0000001;
            if(action_weights[i] >= q_max - tolerance && action_weights[i] <= q_max + tolerance){
                max_indices.push_back((int)i);
                max_weights.push_back(q_max);
            }
        }    
        
        std::cout << "max weights are: ";
        for(size_t i=0; i<max_weights.size(); i++){
            std::cout << max_weights[i];
            std::cout << ", ";
        }
        std::cout << std::endl;
        double max_sum = 0;
        for(size_t i=0; i<max_weights.size(); i++){
            max_sum += max_weights[i];
        }

        if(max_sum >= -0.00000001 && max_sum <= 0.00000001 ){
            double p = 1.0/(double)max_weights.size();    
            for(size_t i=0; i<max_weights.size(); i++){
                max_weights[i] = p;
            }
        }
        
        //i dont need to normalize probabilities?


        //sample a probability distribution to find which action to take
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        //std::default_random_engine generator;
        std::discrete_distribution<> distribution(max_weights.begin(),max_weights.end()); 
        std::vector<double> p = distribution.probabilities();
        
        std::cout << "probabilities are: ";
        for(size_t i=0; i<p.size(); i++){
            std::cout << p[i];
            std::cout << ", ";
        }
        std::cout << std::endl;
        
        int sample = distribution(generator);
        std::cout << "the sampled number is: " << sample << std::endl;
        
        //set a waypoint corresponding to that action
        int action_index = max_indices[sample];
        std::cout << "the action index is: " << action_index << std::endl;
        


        switch(action_index){
            case 0:
                // std::cout << "returned +X" << std::endl;
                return move_plus_x;
                break;
            case 1:
                // std::cout << "returned -X" << std::endl;
                return move_minus_x;
                break;
            case 2:
                // std::cout << "returned +Y" << std::endl;
                return move_plus_y;
                break;
            case 3:
                // std::cout << "returned -Y" << std::endl;
                return move_minus_y;
                break;
            case 4:
                // std::cout << "returned stay" << std::endl;
                return stay;
                break;
            default:
                // std::cout << "returned default(stay)" << std::endl;
                return stay;
                break;
        }

    }

}