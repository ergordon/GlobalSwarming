#include "main.h"
#include <iostream>
#include "SARSAControlPolicy.h"

using namespace sarsa_ros;

int main(int argc, char **argv)
{
    
    std::cout << "SARSA Control Policy Started" << std::endl;


    //need to load the q_data .json files

    //need to initialize the modules
    Initialize(argc, argv);
    
    while(ros::ok()){

        Update();
 
    }
    

    return EXIT_SUCCESS;
}