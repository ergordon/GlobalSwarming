#include "SARSAControlPolicy.h"
#include <typeinfo>
#include <iostream>
#include <json/json.h>
#include <fstream>
#include "TargetSeekModule.h"
#include "BoundaryModule.h"
#include <memory>
#include "action.h"
#include "MMAS.h"
#include "SpeckBucciMMAS.h"

namespace sarsa_ros{

    //TODO this only worked (got include errors) by making static.
    // is there a better way to structure this????
    std::list<Module*> modules;
    std::list<double> module_weights{0.5,0.5};
    std::vector<double> target_position = {20.0, -20.0};
    std::vector<double> bounds = {-6, 6, -6, -6}; //{-x, +x, -y, +y}
    
    //should these be static?
    action current_action = stay;
    MMAS* action_selector;

    geometry_msgs::PoseStamped current_pose; /* current state of the drone. */
    mavros_msgs::State current_state; /* current state. */
    mavros_msgs::SetMode offb_set_mode;
    mavros_msgs::CommandBool arm_cmd;

    ros::Subscriber state_sub;
    ros::Subscriber pose_sub;
    ros::Publisher local_pos_pub;
    ros::ServiceClient arming_client;
    ros::ServiceClient set_mode_client;
    ros::ServiceClient landing_client;

    double ros_rate = 10;
    ros::Time last_mode_request;
    ros::Time last_arm_request;
    

    void Initialize(int argc, char **argv){

        //initialize modules (should this go after ROS initialization?)
        modules.emplace_back(new TargetSeekModule(current_pose, target_position));
        modules.emplace_back(new BoundaryModule(current_pose, bounds));
        action_selector = new SpeckBucciMMAS(modules, module_weights);

        //initialize ROS
        ros::init(argc, argv, "SARSAControlPolicy");
        ros::NodeHandle nh;
        
        //initialize ROS commmunications
        state_sub = nh.subscribe<mavros_msgs::State>
                ("mavros/state", 10, state_cb); /* subscriber for current state.  */
        pose_sub = nh.subscribe<geometry_msgs::PoseStamped>
                ("mavros/local_position/pose", 10, pose_cb); /* subscriber for current pose. */
        local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>
                ("mavros/setpoint_position/local", 10); /* receives its target location from this publisher. */
        arming_client = nh.serviceClient<mavros_msgs::CommandBool>
                ("mavros/cmd/arming"); /* arming service. */
        set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
                ("mavros/set_mode"); /* set mode service. */
        landing_client = nh.serviceClient<mavros_msgs::CommandBool>
                ("mavros/cmd/land"); /* landing service. */
        
        FCUConnect();

        //change to offboard flight mode
        offb_set_mode.request.custom_mode = "OFFBOARD";

        //arm the quadrotor
        arm_cmd.request.value = true;

        last_mode_request = ros::Time::now();
        last_arm_request = ros::Time::now();
    }

    void Update(){

        SetModeOffboard(); 
        SetArm();

        if(current_state.armed && current_state.mode == "OFFBOARD"){

            UpdateModuleStates();

            SelectNextAction();

            TakeNextAction();

        }else{ 
            //send a 'dummy' pose to prevent the FCU from timing out and dropping out of offboard mode 
            geometry_msgs::PoseStamped pose;
            pose.pose.position.x = 0;
            pose.pose.position.y = 0;
            pose.pose.position.z = 1;
            local_pos_pub.publish(pose); //TODO replace with current_pose???
            
        }

        ros::spinOnce();
        ros::Rate rate(ros_rate);
        rate.sleep();    
    }


    void SelectNextAction(){
        //instantiate action selector object
        std::cout << "Selecting next action." << std::endl;
        
        current_action = action_selector->SelectNextAction();
        
    }

    void TakeNextAction(){
        std::cout << "Taking next action." << std::endl;

        //set a waypoint one meter away from the quadrotors current position in the direction indicated by the action.
        //TODO include continuous steering!!!
        double step_size = 1.0;
        std::array<double,2> set_point;
        switch(current_action){
            case move_plus_x:
                std::cout << "moving +X" << std::endl;
                set_point[0] = current_pose.pose.position.x + step_size;
                set_point[1] = current_pose.pose.position.y;
                break;
            case move_minus_x:
                std::cout << "moving -X" << std::endl;
                set_point[0] = current_pose.pose.position.x - step_size;
                set_point[1] = current_pose.pose.position.y;
                break;
            case move_plus_y:
                std::cout << "moving +Y" << std::endl;
                set_point[0] = current_pose.pose.position.x;
                set_point[1] = current_pose.pose.position.y + step_size;
                break;
            case move_minus_y:
                std::cout << "moving -Y" << std::endl;
                set_point[0] = current_pose.pose.position.x;
                set_point[1] = current_pose.pose.position.y - step_size;
                break;
            case stay:
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
        
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = set_point[0];
        pose.pose.position.y = set_point[1];
        pose.pose.position.z = 1.0;

        local_pos_pub.publish(pose);

    }


    void FCUConnect(){
        
        ros::Rate rate(ros_rate);

        //connect to the flight computer
        std::cout << "connecting to the FCU" << std::endl;


        // wait for FCU connection
        while(ros::ok() && !current_state.connected){
                ros::spinOnce();
                rate.sleep();
        }

        std::cout << "FCU connected" << std::endl;


        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = 0;
        pose.pose.position.y = 0;
        pose.pose.position.z = 1;
        
        //send a few setpoints before starting
        //need to to this to prep FCU for OFFBOARD mode  
        for(int i = 100; ros::ok() && i > 0; --i){
                local_pos_pub.publish(pose);
                ros::spinOnce();
                rate.sleep();
        }
    }

    void SetModeOffboard(){
    
        //TODO put into position mode before going into offboard mode.
        //offboard mode goes back into it's most recent mode as a failsafe.

        std::cout << "function start" << std::endl;
        std::cout << "current state mode is: " <<  current_state.mode << std::endl;
        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_mode_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
                std::cout << "current state mode is: " <<  current_state.mode << std::endl;


            }
            last_mode_request = ros::Time::now();
        }
    }

    void SetArm(){
        if( !current_state.armed &&
            current_state.mode == "OFFBOARD" &&
            (ros::Time::now() - last_arm_request > ros::Duration(5.0))){
            if( arming_client.call(arm_cmd) &&
                arm_cmd.response.success){
                ROS_INFO("Vehicle armed");
            }
            last_arm_request = ros::Time::now();
        }   
    }


    

    void UpdateModuleStates(){

        std::cout << "updating module states" << std::endl;

        for (std::list<Module*>::iterator mod_it = modules.begin(); mod_it != modules.end(); mod_it++){
            
            (*mod_it)->UpdateState();
        }


    }


    void state_cb(const mavros_msgs::State::ConstPtr& msg){
        current_state = *msg;
        std::cout << "mode is: " << current_state.mode << std::endl;
    }

    void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
        current_pose = *msg;
    }

}