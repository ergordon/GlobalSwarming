#include <iostream>
#include "gps_pos.h"

using namespace gps_pos;


int main(int argc, char **argv)
{
    
    std::cout << "gps position node started" << std::endl;
    ParseOptions(argc, argv);

    //need to initialize the modules
    Initialize(argc, argv);
    
    while(ros::ok()){

        Update();
 
    }
    

    return EXIT_SUCCESS;
}