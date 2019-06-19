import numpy as np
import action as action

##############################################################################
#   Q learning class
##############################################################################

# The Q learning class handles the storage and update for the Q table
class Qlearning:
    # Class constructor
    def __init__(self):
        # Store Q table 
        self.q_table = np.array([])
        # Store state -> Q table association
        self.q_states = np.array([])

    # Update the Q table and state array
    def update_q(self, state, state_prime, action, action_prime, alpha, gamma, reward):
        # TODO: Think about if i really want to do it this way...        
        # Get the index of the state in the Q state table, given the passed in state
        s_index = self.fetch_row_index_by_state(state)

        # Get the row of the Q table corresponding to the state
        Q_s = self.q_table[s_index]

        # Get the row of the Q table corresponding to the passed in state_prime
        Q_s_p = self.fetch_row_by_state(state_prime)
        
        # Get the numerical index values for the Action enumeration
        a_index = action.value        
        a_p_index = action_prime.value

        # Update the Q table at the state,action index pair
        self.q_table[s_index, a_index]  = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_p_index] - Q_s[a_index])

    # Getch the index of the row in the Q table that corresponds
    #  to a given state. If now row exist for the state, make one
    def fetch_row_index_by_state(self, state):
        index = -1
        if self.q_states.shape[0] != 0: # Check for empty matrix
            matches = np.equal(self.q_states,[state]).all(1).nonzero()
            if matches[0].size == 0:
                # State not in q states add it along with the row
                empty_row = np.zeros(len(action.Action))
                self.q_states = np.vstack([self.q_states, np.copy(state)])
                self.q_table = np.vstack([self.q_table, empty_row])
                index = self.q_states.shape[0]-1
            else:
                # Working state already in q states for this module, 
                # Sum the working q row with the corresponding entry in the q table for this module
                # Incerement the number of times this row has been updated
                index = matches[0][0] 
        else: 
            # Tables are empty, put ours in
            self.q_table = np.zeros((1,len(action.Action)))
            self.q_states = np.copy(state.reshape((1,state.shape[0])))
            index = 0
        return index

    # Fetch the corresponding row of the Q table given a state
    #  If no row for this state exists, create one and return that 
    def fetch_row_by_state(self, state):
        index = -1
        if self.q_states.shape[0] != 0: # Check for empty matrix
            matches = np.equal(self.q_states,[state]).all(1).nonzero()
            if matches[0].size == 0:
                # State not in q states add it along with the row
                empty_row = np.zeros(len(action.Action))
                self.q_states = np.vstack([self.q_states, np.copy(state)])
                self.q_table = np.vstack([self.q_table, empty_row])
                index = self.q_states.shape[0]-1
            else:
                # Working state already in q states for this module, 
                #  sum the working q row with the corresponding entry in the q table for this module
                #  incerement the number of times this row has been updated
                index = matches[0][0]     
        else: 
            # Tables are empty, put ours in
            self.q_table = np.zeros((1,len(action.Action)))
            self.q_states = np.copy(state.reshape((1,state.shape[0])))
            index = 0
        return self.q_table[index]