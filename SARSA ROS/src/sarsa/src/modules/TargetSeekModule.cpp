#include "TargetSeekModule.h"


namespace sarsa_ros{


    TargetSeekModule::TargetSeekModule(geometry_msgs::PoseStamped& pose,  std::vector<double>& target): Module(), current_pose(pose), target_position(target){
        
        //initialize states
        std::vector<double> s{0,0};
        state.push_back(s);

        training_filepath = "/media/kwanchangnim/LinuxDrive/GlobalSwarming/SARSA ROS/src/sarsa/src/TargetSeekModule_training_data.json";
        std::cout << training_filepath << std::endl;

        LoadQData();
    }
    
    TargetSeekModule::~TargetSeekModule(){}

    void TargetSeekModule::UpdateState(){
        
        auto it = state.begin();
        // auto tmp = *it;
        // std::cout << "state was: ";
        // for (int i=0; i<2; i++){
        //     std::cout << tmp[i] << ", ";
        // }
        // std::cout << std::endl;

        //get current state

        std::cout << "target_position is: " << target_position[0] << ", " << target_position[1] << std::endl;
        std::cout << "quadrotor position is: " << current_pose.pose.position.x << ", " << current_pose.pose.position.y << std::endl;
        
        std::vector<double> target_state{0,0};        
        target_state[0] = round(target_position[0] - current_pose.pose.position.x);
        target_state[1] = round(target_position[1] - current_pose.pose.position.y);

        std::cout << "target state is: " << target_state[0] << ", " << target_state[1] << std::endl;

        // auto it = state.begin();
        *it = target_state;

        // it = state.begin();
        // tmp = *it;
        // std::cout << "state is: ";
        // for (int i=0; i<2; i++){
        //     std::cout << tmp[i] << ", ";
        // }
        // std::cout << std::endl;
    }

    

}