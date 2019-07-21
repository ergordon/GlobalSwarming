from enum import Enum

##############################################################################
#   Action enumeration
##############################################################################

# Store the available agent actions in an enumeration for readability and indexability 
class Action(Enum):
    MOVE_PLUS_X = 0
    MOVE_MINUS_X = 1
    MOVE_PLUS_Y = 2
    MOVE_MINUS_Y = 3
    STAY = 4

    def __int__(self):
        return self.value

##############################################################################
#   Action enumeration
##############################################################################
