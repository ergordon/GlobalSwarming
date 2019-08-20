#include "MMAS.h"
#include <iostream>

namespace sarsa_ros{



    MMAS::MMAS(std::list<Module*>& _modules):modules(_modules){
        
    }

    MMAS::~MMAS(){}

    action MMAS::SelectNextAction(){
        std::cout << "MMAS base class action selection" << std::endl;

        return action::move_minus_x;
    }

}