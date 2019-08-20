#ifndef SARSA_TARGETSEEKMODULE_H
#define SARSA_TARGETSEEKMODULE_H

#include <list>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include "Module.h"

namespace sarsa_ros{
    class TargetSeekModule : public Module {
    
        //hold reference to current pose
        //hold reference to target position
    private:    
        geometry_msgs::PoseStamped& current_pose; /* current state of the drone. */
        std::vector<double>& target_position;

    public:
        TargetSeekModule(geometry_msgs::PoseStamped& pose,  std::vector<double>& target);
        ~TargetSeekModule();
        virtual void UpdateState();    
    };
}

#endif