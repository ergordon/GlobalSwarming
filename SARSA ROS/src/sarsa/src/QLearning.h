#ifndef SARSA_QLEARNING_H
#define SARSA_QLEARNING_H

#include <map>
#include <vector>

namespace sarsa_ros{
    class QLearning {
    
    private:
        std::map<std::vector<double>, std::vector<double>> q_data;
        

    public:
        QLearning(std::map<std::vector<double>, std::vector<double>> q_data_in);
        std::vector<double> FetchRowByState(std::vector<double> state);

    };
}


#endif