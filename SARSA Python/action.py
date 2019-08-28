from enum import Enum

##############################################################################
#   Action enumeration
##############################################################################

# Store the available agent actions in an enumeration for readability and indexability 
class Action(Enum):
    MOVE_PLUS_X = 0
    MOVE_PLUS_X_PLUS_Y = 1
    MOVE_PLUS_Y = 2
    MOVE_MINUS_X_PLUS_Y = 3
    MOVE_MINUS_X = 4
    MOVE_MINUS_X_MINUX_Y = 5
    MOVE_MINUS_Y = 6
    MOVE_PLUS_X_MINUS_Y = 7
    STAY = 8

    def __int__(self):
        return self.value

##############################################################################
#   Action enumeration
##############################################################################
