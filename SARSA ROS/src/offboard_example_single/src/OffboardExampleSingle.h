#ifndef OFFBOARD_SINGLE_H
#define OFFBOARD_SINGLE_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <list>

namespace offboard_single{

    void Initialize(int argc, char **argv);
    void Update();
    void FollowSquarePath();
    
    void FCUConnect();
    void SetModeOffboard();
    void SetArm();
    mavros_msgs::State GetState();
    
    void state_cb(const mavros_msgs::State::ConstPtr& msg);
    void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void SpinRos();    

}



#endif