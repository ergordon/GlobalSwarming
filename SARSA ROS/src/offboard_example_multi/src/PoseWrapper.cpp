#include "PoseWrapper.h"

namespace offboard_multi{

    PoseWrapper::PoseWrapper(){};
    
    PoseWrapper::PoseWrapper(int _agent_index, geometry_msgs::PoseStamped _pose){
        agent_index = _agent_index;
        pose = _pose;
    }

}