#include "gps_pos.h"
// #include "PoseWrapper.h"
#include <typeinfo>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <map>


#include <ros/ros.h>
#include <mavros/frame_tf.h>
#include <eigen_conversions/eigen_msg.h>
#include <GeographicLib/Geocentric.hpp>


#include <angles/angles.h>
#include <mavros/mavros_plugin.h>

#include <std_msgs/Float64.h>
#include <sensor_msgs/NavSatStatus.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <geographic_msgs/GeoPointStamped.h>



namespace gps_pos{

    int swarm_size = 1;
    int swarm_index = 0;

    geometry_msgs::PoseStamped set_pose_local; /* set point pose to be sent to the flight computer */
    geometry_msgs::PoseStamped current_pose_local; /* current pose of the drone as received by the mavros subscriber */
    
    sensor_msgs::NavSatFix current_position_global; /* current position of the agent as read by the satellite */
    nav_msgs::Odometry current_position_local;
    mavros_msgs::HomePosition current_home_position;

    double relative_alt;

    mavros_msgs::State current_state; /* current state. */
    mavros_msgs::SetMode offb_set_mode;
    mavros_msgs::CommandBool arm_cmd;

    ros::Subscriber relative_alt_sub;
    ros::Subscriber home_sub;
    ros::Publisher home_pub;
    ros::Subscriber state_sub;
    ros::Subscriber local_pose_sub;
    ros::Publisher local_pose_pub;
    ros::Subscriber local_position_sub;
    ros::Subscriber global_position_sub;
    ros::ServiceClient arming_client;
    ros::ServiceClient set_mode_client;
    ros::ServiceClient landing_client;

    double ros_rate = 10.0;
    ros::Time last_mode_request;
    ros::Time last_arm_request;

    void ParseOptions(int argc, char **argv){
        
        int c;

        bool swarm_index_set = false;
        bool swarm_size_set = false;

        while((c = getopt(argc, argv, "s:i:")) != -1){
            switch(c){
                case 'i':
                    swarm_index = atoi(optarg);    
                    swarm_index_set = true;
                    break;
                case 's':
                    swarm_size = atoi(optarg);
                    swarm_size_set = true;
                    break;
                default:
                    fprintf(stderr, "unrecognized option %c\n", optopt);
                    break;
            }
        }

        if(!swarm_index_set){
            std::cout << "swarm index not set, run with -i <number>" << std::endl;
        }    
        if(!swarm_size_set){
            std::cout << "swarm size not set, run with -s <number>" << std::endl;
        }
        if(!swarm_index_set || !swarm_size_set){
            exit(EXIT_FAILURE);   
        }

    }


    void Initialize(int argc, char **argv){


        //TODO how do I tell which roscore to connect to????
        //initialize ROS
        ros::init(argc, argv, "gps_pos");
        ros::NodeHandle nh;
        
        //initialize internal ROS commmunications
        state_sub = nh.subscribe<mavros_msgs::State>
                ("mavros/state", 10, state_cb); /* subscriber for current state.  */
        local_pose_sub = nh.subscribe<geometry_msgs::PoseStamped>
                ("mavros/local_position/pose", 10, local_pose_cb); /* subscriber for current pose. */
        local_pose_pub = nh.advertise<geometry_msgs::PoseStamped>
                ("mavros/setpoint_position/local", 10); /* receives its target location from this publisher. */
        local_position_sub = nh.subscribe<nav_msgs::Odometry>
                ("mavros/global_position/local", 10, local_position_cb); /* subscriber for current pose. */
        global_position_sub = nh.subscribe<sensor_msgs::NavSatFix>
                ("mavros/global_position/global", 10, global_position_cb); /* subscriber for current pose. */
        relative_alt_sub = nh.subscribe<std_msgs::Float64>
                ("mavros/global_position/rel_alt", 10, relative_alt_cb);
    
        

        arming_client = nh.serviceClient<mavros_msgs::CommandBool>
                ("mavros/cmd/arming"); /* arming service. */
        set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
                ("mavros/set_mode"); /* set mode service. */
        landing_client = nh.serviceClient<mavros_msgs::CommandBool>
                ("mavros/cmd/land"); /* landing service. */
        

        home_sub = nh.subscribe<mavros_msgs::HomePosition>
                ("mavros/home_position/home", 10, home_positin_cb);
        home_pub = nh.advertise<mavros_msgs::HomePosition>
                ("mavros/global_position/home", 10); /* receives its target location from this publisher. */
        

        //connect to the flight computer
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


        uint64_t t_micro = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        double timestamp_micro_s = static_cast<double>(t_micro);
        double formation_position_local[3];
        double phase = (double)swarm_index/(double)swarm_size*2.0*3.14159265;
        formation_position_local[0] = 5.0*cos(timestamp_micro_s/1000000.0/50.0*3.14159265-phase);
        formation_position_local[1] = 5.0*sin(timestamp_micro_s/1000000.0/50.0*3.14159265-phase);
        formation_position_local[2] = 3.0;


        //send a 'dummy' pose to prevent the FCU from timing out and dropping out of offboard mode 
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = formation_position_local[0];
        pose.pose.position.y = formation_position_local[1   ];
        pose.pose.position.z = formation_position_local[2];
        local_pose_pub.publish(pose);

        ros::spinOnce();
        ros::Rate rate(ros_rate);
        rate.sleep();    
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
                local_pose_pub.publish(pose);
                ros::spinOnce();
                rate.sleep();
        }
    }

    void SetModeOffboard(){
    
        //TODO put into position mode before going into offboard mode.
        //offboard mode goes back into it's most recent mode as a failsafe.

        std::cout << "[Agent " << swarm_index << "]current state mode is: " <<  current_state.mode << std::endl;
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


    void state_cb(const mavros_msgs::State::ConstPtr& msg){
        current_state = *msg;
        std::cout << "mode is: " << current_state.mode << std::endl;
    }

    void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg){
        current_pose_local = *msg;

        // std::cout << "[agent " << swarm_index <<"] pose is: " << current_pose.pose.position.x << ", " << current_pose.pose.position.y << ", " << current_pose.pose.position.z << std::endl;
        
    }

    void local_position_cb(const nav_msgs::Odometry::ConstPtr& msg){
        current_position_local = *msg;
    }

    void relative_alt_cb(const std_msgs::Float64::ConstPtr& msg){
        relative_alt = msg->data;
    }

    void global_position_cb(const sensor_msgs::NavSatFix::ConstPtr& msg){
        current_position_global = *msg;

        Eigen::Vector3d my_map_point;
        try {
			GeographicLib::Geocentric map(GeographicLib::Constants::WGS84_a(),
						GeographicLib::Constants::WGS84_f());

			// Current fix to ECEF
			map.Forward(current_position_global.latitude, current_position_global.longitude, current_position_global.altitude,
						my_map_point.x(), my_map_point.y(), my_map_point.z());

		}
		catch (const std::exception& e) {
			ROS_INFO_STREAM("GP: Caught exception: " << e.what() << std::endl);
		}

        
        Eigen::Vector3d home_map_point;
        try {
			
			GeographicLib::Geocentric map(GeographicLib::Constants::WGS84_a(),
						GeographicLib::Constants::WGS84_f());


			// Current fix to ECEF                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
			map.Forward(current_home_position.geo.latitude,current_home_position.geo.longitude, current_home_position.geo.altitude,
						home_map_point.x(), home_map_point.y(), home_map_point.z());                                                                                                                                                                                                                    

		}
		catch (const std::exception& e) {
			ROS_INFO_STREAM("GP: Caught exception: " << e.what() << std::endl);
		}



        //take his global lla->ecef position minus my global lla->ecef position then convert to enu 
        Eigen::Vector3d local_home(3);
        local_home[0] = current_home_position.position.x;
        local_home[1] = current_home_position.position.y;
        local_home[2] = current_home_position.position.z;
        // Eigen::Vector3d local_ecef = my_map_point-(home_map_point+local_home); 
        Eigen::Vector3d local_ecef = my_map_point-home_map_point; 
        geometry_msgs::Point local_enu;
        Eigen::Vector3d map_origin(3);
        map_origin[0] = current_home_position.geo.latitude;
        map_origin[1] = current_home_position.geo.longitude;
        map_origin[2] = current_home_position.geo.altitude;
        
        tf::pointEigenToMsg(mavros::ftf::transform_frame_ecef_enu(local_ecef, map_origin), local_enu); 
        // tf::pointEigenToMsg(mavros::ftf::transform_frame_ecef_enu(local_ecef, home_map_point), local_enu); 
        
        // //TODO dont forget altitude offset...
        // /**
		//  * @brief By default, we are using the relative altitude instead of the geocentric
		//  * altitude, which is relative to the WGS-84 ellipsoid
		//  */
		// if (use_relative_alt)
		// 	odom->pose.pose.position.z = relative_alt->data;

        // if (use_relative_alt)
        //     odom->pose.pose.position.z = relative_alt->data;
        if(true){
            local_enu.z = relative_alt;
        }


        //at this point, local_enu should be the same as global_position/local
        std::cout << "global_position/local is: "
                  << current_position_local.pose.pose.position.x << ", "
                  << current_position_local.pose.pose.position.y << ", "
                  << current_position_local.pose.pose.position.z << std::endl;

        std::cout << "calculated local_enu: "
                  << local_enu.x << ", "
                  << local_enu.y << ", "
                  << local_enu.z << std::endl;


        // // Compute the local coordinates in ECEF
		// local_ecef = map_point - ecef_origin;
		// // Compute the local coordinates in ENU
		// tf::pointEigenToMsg(ftf::transform_frame_ecef_enu(local_ecef, map_origin), odom->pose.pose.position);

        


    }

    void home_positin_cb(const mavros_msgs::HomePosition::ConstPtr& msg){
        current_home_position = *msg;
        
    
        // home_pub.publish(current_home_position);
        

    }
}
