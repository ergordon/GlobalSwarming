import numpy as np
import action as action

##############################################################################
#   Q learning class
##############################################################################

# The Q learning class handles the storage and update for the Q table
class Qlearning:
    # Class constructor
    def __init__(self):

        self.q_data = {} # Store the Q table and associated states in a dictionary
        self.q_updates = {} # Store the number of times the Q table has been updated for each state
        
        # # Store Q table 
        # self.q_table = np.array([])
        # # Store state -> Q table association
        # self.q_states = np.array([])

    # Update the Q table and state array
    def update_q(self, state, state_prime, action, action_prime, alpha, gamma, reward):
        # TODO: Think about if i really want to do it this way...        
        # Get the row of the Q table corresponding to the state
        
        if np.array_equal(state,state_prime):
            if  action != action.STAY:
                return
        else:
            if action == action.STAY:
                return
        

        Q_s = self.fetch_row_by_state(state)
        # print('Q_s is ', Q_s)

        # Get the row of the Q table corresponding to the passed in state_prime
        Q_s_p = self.fetch_row_by_state(state_prime)
        # print('Q_s_p is ', Q_s_p)

        # print('action is ', action)
        # print('action prime is ', action_prime)

        # Get the numerical index values for the Action enumeration
        a_index = action.value        
        a_p_index = action_prime.value
        
        # Update the Q table at the state,action index pair
        self.q_data[tuple(state)][a_index]  = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_p_index] - Q_s[a_index])
        self.q_updates[tuple(state)] = self.q_updates[tuple(state)] + 1 
        # print('updated Q row is ', self.q_table[s_index])

    # Fetch the corresponding row of the Q table given a state
    #  If no row for this state exists, create one and return that 
    def fetch_row_by_state(self, state):
        if tuple(state) not in self.q_data:
            empty_row = np.zeros(len(action.Action))
            self.q_data.update({tuple(state):empty_row})
            self.q_updates.update({tuple(state):0})
            return empty_row
        else:
            return self.q_data[tuple(state)]

    def fetch_updates_by_state(self, state):
        if tuple(state) not in self.q_data:
            empty_row = np.zeros(len(action.Action))
            self.q_data.update({tuple(state):empty_row})
            self.q_updates.update({tuple(state):0})
            return 0
        else:
            return self.q_updates[tuple(state)]