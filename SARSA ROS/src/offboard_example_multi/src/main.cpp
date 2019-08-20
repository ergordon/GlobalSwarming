#include <iostream>
#include "OffboardExampleMulti.h"

using namespace offboard_multi;


int main(int argc, char **argv)
{
    
    std::cout << "Multiple UAV offboard example started" << std::endl;
    ParseOptions(argc, argv);

    //need to initialize the modules
    Initialize(argc, argv);
    
    while(ros::ok()){
        
        Update();
        
    }
    

    return EXIT_SUCCESS;
}