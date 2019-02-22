%enumeration class defining the agents possible actions
%TODO consider having separate action sets for different functionality
%modules and for different values of the statespace
%this also currently doesnt handle orientation of the body in any way.
classdef Action
   enumeration
        move_plusX, move_minusX, move_plusY, move_minusY, move_plusZ, move_minusZ, stay
   end
end