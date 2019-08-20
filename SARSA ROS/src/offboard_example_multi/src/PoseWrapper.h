#ifndef OFFBOARD_MULTI_POSEWRAPPER_H
#define OFFBOARD_MULTI_POSEWRAPPER_H

#include <geometry_msgs/PoseStamped.h>

namespace offboard_multi{

    class PoseWrapper{

        public:
            int agent_index;
            geometry_msgs::PoseStamped pose;

            PoseWrapper();
            PoseWrapper(int _agent_index, geometry_msgs::PoseStamped _pose);

    };

}



#endif