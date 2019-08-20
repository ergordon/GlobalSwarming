#include "BoundaryModule.h"


namespace sarsa_ros{


    BoundaryModule::BoundaryModule(geometry_msgs::PoseStamped& pose,  std::vector<double>& _bounds): Module(), current_pose(pose), bounds(_bounds){
        
        //initialize states
        for(size_t i=0; i<bounds.size(); i++){
            std::vector<double> s{0};
            state.push_back(s);
        }

        training_filepath = "/media/kwanchangnim/LinuxDrive/GlobalSwarming/SARSA ROS/src/sarsa/src/BoundaryModule_training_data.json";
        std::cout << training_filepath << std::endl;

        LoadQData();

    }
    
    BoundaryModule::~BoundaryModule(){}
    

    void BoundaryModule::UpdateState(){
        
        //  For this module, it is sa vector containing distances from the agent to each boundary
        //  Ordering is [+x,-x,+y,-y] (append [+z,-z] for 3D case)

        //Boundary vector {-x, +x, -y, +y}
        //sort bounds to make iteration easier
        std::vector<double> s_bounds;
        for(size_t i=0; i<bounds.size(); i+=2){
            s_bounds.push_back(bounds[i+1]);
            s_bounds.push_back(bounds[i]);
        }
        
        std::vector<double> s_position;
        s_position.push_back(current_pose.pose.position.x);
        s_position.push_back(current_pose.pose.position.x);
        s_position.push_back(current_pose.pose.position.y);
        s_position.push_back(current_pose.pose.position.y);
        
        std::cout << "position is: " << current_pose.pose.position.x << ", " << current_pose.pose.position.y << std::endl;
        std::cout << "bounds are: ";
        for(size_t i=0; i<bounds.size(); i++){
            std::cout << bounds[i] << ", ";
        }
        std::cout << std::endl;



        size_t i=0;
        for(auto it=state.begin(); it!=state.end(); it++){
            std::vector<double> s;
            s.push_back(round(s_bounds[i] - s_position[i]));
            *it = s;
            i++;
        } 
        
        // for i in range(0,len(Simulation.search_space)):   
        //     # Round to whole numbers for discretization
        //     self.state[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
        //     self.state[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])    
        




        std::cout << "state is: <";
        
        for(auto it=state.begin(); it!=state.end(); it++){
            auto tmp = *it;
            for (size_t i=0; i<tmp.size(); i++){
                std::cout << tmp[i] << ", ";
            }
        }
        std::cout << ">" << std::endl;

        // //distance from 
        // std::vector<double> current_position{current_pose.pose.position.x, current_pose.pose.position.y};
        // int i = 0;
        // auto it = state.begin();
        // // for(size_t i=0; i<s_bounds.size(); i+=2){
        // //     std::vector<double> s1{s_bounds[i]-current_position[i]};
        // //     std::vector<double> s2{s_bounds[i+1]-current_position[i]};
        // // }
        // while(it!=state.end()){
        //     std::cout << "it n n+1" << std::endl;
        //     std::vector<double> s1{s_bounds[i]-current_position[i]};
        //     std::vector<double> s2{s_bounds[i+1]-current_position[i]};
        //     // s.push__back(bounds[i] - current_pose.pose.position.x);
        //     *it = s1;
        //     *it++;
        //     *it = s2; 
        //     i+=2;

        // }
        
        // std::vector<double> current_position{current_pose.pose.position.x, current_pose.pose.position.y};
        // auto it = state.begin();
        // std::vector<double> s{0.0};
        // *it = s;
        // it++;
        // *it = s;
        // it++;
        // *it = s;
        // it++;
        // *it = s;
        


        // for i in range(0,len(Simulation.search_space)):   
        //     # Round to whole numbers for discretization
        //     self.state[i*2] = np.round(Simulation.search_space[i][1] - self.parent_agent.position[i]) 
        //     self.state[i*2+1] = np.round(Simulation.search_space[i][0] - self.parent_agent.position[i])    
        
        // auto it = state.begin();
        // for(size_t i=0; i<bounds.size(); i++){
        //     it =     
        // }




        // int i = 0;
        // for(auto it=state.begin(); it!=state.end(); state++ ){




        //     i++;
        // }

        // auto tmp = *it;
        // std::cout << "state was: ";
        // for (int i=0; i<2; i++){
        //     std::cout << tmp[i] << ", ";
        // }
        // std::cout << std::endl;

        //get current state

        // std::cout << "target_position is: " << target_position[0] << ", " << target_position[1] << std::endl;
        // std::cout << "quadrotor position is: " << current_pose.pose.position.x << ", " << current_pose.pose.position.y << std::endl;

        // std::vector<double> target_state{0,0};        
        // target_state[0] = round(target_position[0] - current_pose.pose.position.x);
        // target_state[1] = round(target_position[1] - current_pose.pose.position.y);

        // std::cout << "target state is: " << target_state[0] << ", " << target_state[1] << std::endl;

        // // auto it = state.begin();
        // *it = target_state;

        // // it = state.begin();
        // // tmp = *it;
        // // std::cout << "state is: ";
        // // for (int i=0; i<2; i++){
        // //     std::cout << tmp[i] << ", ";
        // // }
        // // std::cout << std::endl;

        // it++;
    }

    

}