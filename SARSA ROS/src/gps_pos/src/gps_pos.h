#ifndef GPS_POS_H
#define GPS_POS_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/NavSatFix.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <list>
#include <mavros_msgs/HomePosition.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>

namespace gps_pos{

    void ParseOptions(int argc, char **argv);
    void Initialize(int argc, char **argv);
    void Update();
    void FollowFormation();
    
    void FCUConnect();
    void SetModeOffboard();
    void SetArm();
    mavros_msgs::State GetState();
    
    void state_cb(const mavros_msgs::State::ConstPtr& msg);
    void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void local_position_cb(const nav_msgs::Odometry::ConstPtr& msg);
    void global_position_cb(const sensor_msgs::NavSatFix::ConstPtr& msg);
    void relative_alt_cb(const std_msgs::Float64::ConstPtr& msg);
    void home_positin_cb(const mavros_msgs::HomePosition::ConstPtr& msg);
    void SpinRos();    

}



#endif