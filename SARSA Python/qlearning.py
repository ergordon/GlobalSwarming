import numpy as np
import action as action

class Qlearning:

    
    def __init__(self):
        #store Q table (This table is shared by all agents in the same module)
        self.q_table = np.array()
        #store state->Q table association
        self.q_states = np.array()



    #update q
    def update_q(self, state, state_prime, action, action_prime, alpha, gamma, reward):
        print("think about if i really want to do it this way...")

    #fetch the corresponding row of the Q table given a state
    # if no row for this state exists, create one and return that 
    def fetch_row_by_state(self, state):
        
        #TODO research a better (more efficeint) way of doing this
        index = -1
        
        if self.q_states.size != 0: #check for empty matrix
            for i in range(0,self.q_states.shape[0]): #iterate over each row
                if np.array_equal(state, self.q_states[i]): #if a row matches
                    index = i
                    break

            #row not found, add an entry to the q_table and q_states    
            if index == -1:
                empty_row = np.zeros(len(action.Action))
                self.q_states = np.vstack([self.q_states, state])
                self.q_table = np.vstack([self.q_table, empty_row])
                index = self.q_states.shape[0]-1

        else: 
            #no tables are empty, put ours in
            self.q_table = np.zeros(len(action.Action))
            self.q_states = state
            index = 0

        print(self.q_table[index])
        return self.q_table[index]
        
    
