/**
 * @file offb_node.cpp
 * @brief Offboard control example node, written with MAVROS version 0.19.x, PX4 Pro Flight
 * Stack and tested in Gazebo SITL
 */

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <fstream>

#include "std_msgs/String.h"


#include <sstream>

#include <map>
#include <math.h>
#include <array>
#include <iostream>
#include <string>
#include <random>
#include <chrono>
#include <list>
#include <vector>

#include <json/json.h>
#include <fstream>


mavros_msgs::State current_state; /* current state. */
geometry_msgs::PoseStamped current_pose; /* current state of the drone. */

void state_cb(const mavros_msgs::State::ConstPtr& msg){
    current_state = *msg;
}

void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
    current_pose = *msg;
}


int main(int argc, char **argv)
{
    std::cout << "we got something..." << std::endl;

    ros::init(argc, argv, "target_seek_example");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb); /* subscriber for current state.  */
    ros::Subscriber pose_sub = nh.subscribe<geometry_msgs::PoseStamped>
            ("mavros/local_position/pose", 10, pose_cb); /* subscriber for current pose. */
    ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
            ("mavros/setpoint_position/local", 10); /* receives its target location from this publisher. */
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming"); /* arming service. */
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode"); /* set mode service. */
    ros::ServiceClient landing_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/land"); /* landing service. */

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = 0;
    pose.pose.position.y = 0;
    pose.pose.position.z = 1;


    // target positions of desired waypoints.
    std::vector<double> target_position = {20.0, -20.0};
    std::vector<double> target_state = {0.0, 0.0};
    std::list<std::map<std::vector<double>, std::vector<double>>> Q;
    
    
    //load the Q data
    std::cout << "loading json from disk" << std::endl;
    std::string file_path = "/media/kwanchangnim/LinuxDrive/GlobalSwarming/SARSA ROS/src/sarsa/src/TargetSeekModule_training_data.json";

    Json::Value training_json;
    Json::Reader reader;
    
    std::ifstream json_file(file_path);
    if(!json_file){
        std::cout << "json file is not open" << std::endl;
    }
    
    if(json_file.peek() == std::ifstream::traits_type::eof()){
        std::cout << "json file is empty" << std::endl;
    }
    

    std::stringstream buffer;
    buffer << json_file.rdbuf();
    std::string json_string = buffer.str();
    
    //parse the json

    if(!reader.parse(json_string, training_json)) {
        std::cout << reader.getFormattedErrorMessages() << std::endl;
    }else{
        std::cout << "json successfully parsed, working with it" << std::endl;

        // //TODO, how to auto-handle collapsable Q???? maybe have json variable stored when saving? YES!
        // bool collapsable_q = false;
        // if(collapsable_q){

        // } else {
            
        // } //probably dont even need to do this? can infer collapsability training_json["data"].size()



        //store the maps in a list

        for(size_t j=0; j<training_json["data"].size(); j++){
            //create a map for each
            std::map<std::vector<double>, std::vector<double>> Q_data;

            for(size_t k=0; k<training_json["data"][(int)j].size(); k++){

                Json::Value key_value_pair = training_json["data"][(int)j][(int)k];

                std::cout << "state is: ";
                std::vector<double> temp_state;
                for(size_t m=0; m<key_value_pair["key"].size(); m++){
                    temp_state.push_back(key_value_pair["key"][(int)m].asDouble());
                    std::cout << temp_state[(int)m] << ", ";
                }
                std::cout << std::endl;


                std::cout << "q row is: ";
                std::vector<double> temp_Q_row;
                for(size_t m=0; m<key_value_pair["value"].size(); m++){
                    temp_Q_row.push_back(key_value_pair["value"][(int)m].asDouble());
                    std::cout << temp_Q_row[(int)m] << ", ";
                }
                std::cout << std::endl;

                // std::cout << "json pair is: " <<  training_json["data"][(int)j][(int)k] << std::endl;

                Q_data.insert({temp_state,temp_Q_row});


            }

            //add Q_data to list
            Q.push_back(Q_data);

        }
        // Read and modify the json data
        // std::cout << "data size: " << training_json["data"].size() << std::endl;
        // std::cout << "Contains data? " << training_json.isMember("data") << std::endl;
    }



    
    //connect to the flight computer
    std::cout << "connecting to the FCU" << std::endl;

    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    //NOTE i'm not sure why we are doing this.
    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i){
        local_pos_pub.publish(pose);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();


    
    while(ros::ok()){ // && cycle_idx <= total_cycles){
        
        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }

        //TODO here is where we need module specific classes. lets write this to work better later...

        //get current state
        target_state[0] = round(target_position[0] - current_pose.pose.position.x);
        target_state[1] = round(target_position[1] - current_pose.pose.position.y);


        std::vector<double> Q_row;
        // look it up in the Q-table
        for (auto it = Q.begin(); it != Q.end(); it++)
        {
            std::map<std::vector<double>, std::vector<double>> Q_data = *it; 
            if(Q_data.count(target_state) == 1){
                Q_row = Q_data[target_state];

                // std::cout << "state found, Q row is: ";
                // for (int i=0; i<5; i++){
                //     std::cout << Q_row[i];
                //     std::cout << ", ";
                // }
                // std::cout << std::endl;    
                // std::cout << "state found" << std::endl;    


            }else{
                //TODO: how will we handle states that arent in the dictionary?
                //initially, I think I will look for the state closest to what we have
                //...how do I to that?
                // std::cout << "state not found, figure it out" << std::endl;

                Q_row = {0.0, 0.0, 0.0, 0.0, 0.0};
            }


            // // Access the object through iterator
            // int id = it->id;
            // std::string name = it->name;
    
            // //Print the contents
            // std::cout << id << " :: " << name << std::endl;
    
        }



        //chose an action
        //first get largest number
        double q_max = *std::max_element(std::begin(Q_row), std::end(Q_row));
        std::cout << "max q is: " << q_max << std::endl;

        //next get indices for all max occurences
        std::vector<double> action_weights;
        std::vector<int> max_indices; 
        for(size_t i=0; i<Q_row.size(); i++){
            if(Q_row[i] == q_max){
                max_indices.push_back((int)i);
                action_weights.push_back(Q_row[i]);
            }
        }    
        
        std::cout << "action weights are: ";
        for(size_t i=0; i<action_weights.size(); i++){
            std::cout << action_weights[i];
            std::cout << ", ";
        }
        std::cout << std::endl;
        
        //sample a probability distribution to find which action to take
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator (seed);
        //std::default_random_engine generator;
        std::discrete_distribution<> distribution(action_weights.begin(),action_weights.end()); 
        std::vector<double> p = distribution.probabilities();
        std::cout << "probabilities are: ";
        for(size_t i=0; i<p.size(); i++){
            std::cout << p[i];
            std::cout << ", ";
        }
        std::cout << std::endl;
        
        int sample = distribution(generator);
        std::cout << "the sampled number is: " << sample << std::endl;
        
        //set a waypoint corresponding to that action
        int action_index = max_indices[sample];


        //TODO include continuous steering!!!
        double step_size = 1.0;
        std::array<double,2> set_point;
        switch(action_index){
            case 0:
                std::cout << "moving +X" << std::endl;
                set_point[0] = current_pose.pose.position.x + step_size;
                set_point[1] = current_pose.pose.position.y;
                break;
            case 1:
                std::cout << "moving -X" << std::endl;
                set_point[0] = current_pose.pose.position.x - step_size;
                set_point[1] = current_pose.pose.position.y;
                break;
            case 2:
                std::cout << "moving +Y" << std::endl;
                set_point[0] = current_pose.pose.position.x;
                set_point[1] = current_pose.pose.position.y + step_size;
                break;
            case 3:
                std::cout << "moving -Y" << std::endl;
                set_point[0] = current_pose.pose.position.x;
                set_point[1] = current_pose.pose.position.y - step_size;
                break;
            case 4:
                std::cout << "staying" << std::endl;
                set_point[0] = current_pose.pose.position.x;
                set_point[1] = current_pose.pose.position.y;
                break;
            default:
                std::cout << "undefined action index ...doing nothing." << std::endl;
                set_point[0] = current_pose.pose.position.x;
                set_point[1] = current_pose.pose.position.y;
                break;
        }


        //now publish the set_point!!!!
        std::cout << "publishing setpoint: ";
        for(size_t i=0; i<set_point.size(); i++){
            std::cout << set_point[i] << ", ";
        }
        std::cout << std::endl;
        

        pose.pose.position.x = set_point[0];
        pose.pose.position.y = set_point[1];
        pose.pose.position.z = 1.0;

        
        local_pos_pub.publish(pose);

        ros::spinOnce();
        rate.sleep();

	
	// std::ofstream out("ibqr_waypoint_test.csv", std::ofstream::out | std::ofstream::app);
	// out << pose.pose.position.x << ",";
	// out << pose.pose.position.y << ",";
	// out << pose.pose.position.z << ",";
	// out << current_pose.pose.position.x << ",";
	// out << current_pose.pose.position.y << ",";
	// out << current_pose.pose.position.z;
	// out << std::endl;


	// std::cout << pose.pose.position.x << ",";
    // std::cout << pose.pose.position.y << ",";
    // std::cout << pose.pose.position.z << ",";
    // std::cout << current_pose.pose.position.x << ",";
    // std::cout << current_pose.pose.position.y << ",";
    // std::cout << current_pose.pose.position.z;
    // std::cout << std::endl;
    }

    // land.
    mavros_msgs::CommandBool land_cmd;
    land_cmd.request.value = true;
    while(ros::ok()){
        if(landing_client.call(land_cmd)){
            ROS_INFO("Vehicle landing.");
            break;
        }
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
