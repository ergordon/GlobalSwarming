from enum import Enum
import numpy as np

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
    MOVE_MINUS_X_MINUS_Y = 5
    MOVE_MINUS_Y = 6
    MOVE_PLUS_X_MINUS_Y = 7
    STAY = 8

    def __int__(self):
        return self.value

##############################################################################
#   Action enumeration
##############################################################################

# class ActionHelper():
    

def populate_action_headings(action_headings):
    for action_data in Action:
        if not action_data == Action.STAY:
            heading = float(action_data.value)/float(len(Action)-1)*2*np.pi
            print(heading)
            action_headings.update({action_data : heading})

def populate_nearest_actions(nearest_actions):
    #+X
    nearest = np.array([Action.MOVE_PLUS_X_PLUS_Y, Action.MOVE_PLUS_X_MINUS_Y])
    nearest_actions.update({Action.MOVE_PLUS_X:nearest})
    #+X+Y
    nearest = np.array([Action.MOVE_PLUS_Y, Action.MOVE_PLUS_X])
    nearest_actions.update({Action.MOVE_PLUS_X_PLUS_Y:nearest})
    #+Y
    nearest = np.array([Action.MOVE_MINUS_X_PLUS_Y, Action.MOVE_PLUS_X_PLUS_Y])
    nearest_actions.update({Action.MOVE_PLUS_Y:nearest})
    #-X+Y
    nearest = np.array([Action.MOVE_MINUS_X, Action.MOVE_PLUS_Y])
    nearest_actions.update({Action.MOVE_MINUS_X_PLUS_Y:nearest})
    #-X
    nearest = np.array([Action.MOVE_MINUS_X_MINUS_Y, Action.MOVE_MINUS_X_PLUS_Y])
    nearest_actions.update({Action.MOVE_MINUS_X:nearest})
    #-X-Y
    nearest = np.array([Action.MOVE_MINUS_Y, Action.MOVE_MINUS_X])
    nearest_actions.update({Action.MOVE_MINUS_X_MINUS_Y:nearest})
    #-Y
    nearest = np.array([Action.MOVE_PLUS_X_MINUS_Y, Action.MOVE_MINUS_X_MINUS_Y])
    nearest_actions.update({Action.MOVE_MINUS_Y:nearest})
    #+X-Y
    nearest = np.array([Action.MOVE_PLUS_X, Action.MOVE_MINUS_Y])
    nearest_actions.update({Action.MOVE_PLUS_X_MINUS_Y:nearest})
    
def populate_opposite_actions(opposite_actions):
    #+X
    opposite = Action.MOVE_MINUS_X
    opposite_actions.update({Action.MOVE_PLUS_X:opposite})
    #+X+Y
    opposite = Action.MOVE_MINUS_X_MINUS_Y
    opposite_actions.update({Action.MOVE_PLUS_X_PLUS_Y:opposite})
    #+Y
    opposite = Action.MOVE_MINUS_Y
    opposite_actions.update({Action.MOVE_PLUS_Y:opposite})
    #-X+Y
    opposite = Action.MOVE_PLUS_X_MINUS_Y
    opposite_actions.update({Action.MOVE_MINUS_X_PLUS_Y:opposite})
    #-X
    opposite = Action.MOVE_PLUS_X
    opposite_actions.update({Action.MOVE_MINUS_X:opposite})
    #-X-Y
    opposite = Action.MOVE_PLUS_X_PLUS_Y
    opposite_actions.update({Action.MOVE_MINUS_X_MINUS_Y:opposite})
    #-Y
    opposite = Action.MOVE_PLUS_Y
    opposite_actions.update({Action.MOVE_MINUS_Y:opposite})
    #+X-Y
    opposite = Action.MOVE_MINUS_X_PLUS_Y
    opposite_actions.update({Action.MOVE_PLUS_X_MINUS_Y:opposite})

class ActionHelper():
    
    action_headings = {}
    nearest_actions = {}
    opposite_actions = {}
    populate_action_headings(action_headings)
    populate_nearest_actions(nearest_actions)
    populate_opposite_actions(opposite_actions)