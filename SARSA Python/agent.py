import action
import numpy as np
import module as module

class Agent:
    
    #TODO move these into the constructor???
    position = None #the agents position
    total_reward = 0 #running reward received by the agent

    def __init__(self,pos):
        self.position = pos
        self.modules = []
        self.modules.append(module.CohesionModule())
        #self.modules.append(module.CollisionModule())

        
    def changePos(self,pos):
        self.position = pos

    def takeAction(self,act):
        if act == action.Action.MOVE_PLUS_X :
            self.position = self.position + [1,0]
        elif  act == action.Action.MOVE_MINUS_X :
            self.position = self.position + [-1,0]
        elif  act == action.Action.MOVE_PLUS_Y :
            self.position = self.position + [0,1]
        elif  act == action.Action.MOVE_MINUS_Y :
            self.position = self.position + [0,-1]
        else: #act == action.Action.STAY
            self.position = self.position + [0,0]

