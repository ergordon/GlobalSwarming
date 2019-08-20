#ifndef SARSA_MODULE_H
#define SARSA_MODULE_H

#include <list>
#include "QLearning.h"

namespace sarsa_ros{
    class Module {
    
    protected:
        std::list<QLearning> Q;
        std::list<std::vector<double>> state;
        std::string training_filepath;

    public:
        Module();
        virtual ~Module();
        std::vector<double> GetActionWeights();        
        float GetModuleWeight();
        virtual void UpdateState();
        void LoadQData();

    };
}

#endif