#ifndef SARSA_BOUNDARYMODULE_H
#define SARSA_BOUNDARYMODULE_H

#include <list>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include "Module.h"

namespace sarsa_ros{
    class BoundaryModule : public Module {
    
        //hold reference to current pose
        //hold reference to bounds
    private:    
        geometry_msgs::PoseStamped& current_pose; /* current state of the drone. */
        std::vector<double>& bounds;

    public:
        BoundaryModule(geometry_msgs::PoseStamped& pose,  std::vector<double>& _bounds);
        ~BoundaryModule();
        virtual void UpdateState();    
    };
}

#endif