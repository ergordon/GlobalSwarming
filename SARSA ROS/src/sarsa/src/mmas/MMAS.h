#ifndef SARSA_MMAS_H
#define SARSA_MMAS_H

#include <list>
#include "Module.h"
#include "action.h"

namespace sarsa_ros{
    class MMAS {
    

    protected:
        
        std::list<Module*>& modules;

    public:

        //inputs should be a reference to the modules list
        MMAS(std::list<Module*>& _modules);
        virtual ~MMAS();

        virtual action SelectNextAction();               
    };
}


#endif