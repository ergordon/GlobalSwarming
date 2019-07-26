#include "QLearning.h"
#include <iostream>

namespace sarsa_ros{


    QLearning::QLearning(std::map<std::vector<double>, std::vector<double>> q_data_in){
        q_data = q_data_in;
    }

    std::vector<double> QLearning::FetchRowByState(std::vector<double> state){

        std::vector<double> q_row;
        if(q_data.count(state) == 1){
            q_row = q_data[state];

            // std::cout << "state found, Q row is: ";
            // for (int i=0; i<5; i++){
            //     std::cout << q_row[i];
            //     std::cout << ", ";
            // }
            // std::cout << std::endl;      


        }else{
            //TODO: how will we handle states that arent in the dictionary?
            //initially, I think I will look for the state closest to what we have
            //...how do I to that?
            // std::cout << "state not found, figure it out" << std::endl;

            q_row = {0.0, 0.0, 0.0, 0.0, 0.0};
        }

        return q_row;
    }






}