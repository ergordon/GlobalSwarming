#include <iostream>
#include "OffboardExampleSingle.h"

using namespace offboard_single;


int main(int argc, char **argv)
{
    
    std::cout << "Single UAV offboard example started" << std::endl;


    //need to initialize the modules
    Initialize(argc, argv);
    
    while(ros::ok()){

        Update();
 
    }
    

    return EXIT_SUCCESS;
}