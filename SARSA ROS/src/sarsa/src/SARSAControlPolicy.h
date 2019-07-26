#ifndef SARSA_CONTROLPOLICY_H
#define SARSA_CONTROLPOLICY_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <list>
#include "Module.h"

namespace sarsa_ros{


    
    void Initialize(int argc, char **argv);
    void Update();
    void SelectNextAction();
    void TakeNextAction();

    void FCUConnect();
    void SetModeOffboard();
    void SetArm();
    void UpdateModuleStates();
    mavros_msgs::State GetState();
    
    void state_cb(const mavros_msgs::State::ConstPtr& msg);
    void pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void SpinRos();    

}



#endif