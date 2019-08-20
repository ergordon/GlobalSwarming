#ifndef SARSA_SPECKBUCCIMMAS_H
#define SARSA_SPECKBUCCIMMAS_H

#include "MMAS.h"
#include "Module.h"
#include <list>


namespace sarsa_ros{
    class SpeckBucciMMAS : public MMAS {
    

    private:
        std::list<double>& module_weights;

    public:

        //inputs will be the reference to the modules and a list of fixed module weights
        SpeckBucciMMAS(std::list<Module*>& _modules, std::list<double>& weights);
        ~SpeckBucciMMAS();

        void PrintModuleWeights();
        virtual action SelectNextAction();                 
    };
}


#endif