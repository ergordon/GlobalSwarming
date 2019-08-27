import numpy as np
from action import Action 

##############################################################################
#   Q learning class
##############################################################################

# The Q learning class handles the storage and update for the Q table
class Qlearning:
    # Class constructor
    def __init__(self):#, obs_state_transitions = False):

        self.q_data = {} # Store the Q table and associated states in a dictionary
        self.q_updates = {} # Store the number of times the Q table has been updated for each state
        self.q_epsilon = {} # store the state based epsilon parameters to be used with VBDE softmax  
        # self.observe_state_transitions = obs_state_transitions #if true, will only update Q when a state transisiton has occured. or hasn't ocurred in the case of Action.stay 

    # Update the Q table and state array
    def update_q(self, state, state_prime, action, action_prime, alpha, gamma, reward):
        
        # if observe_state_transitions:    
        #     if np.array_equal(state, state_prime) and action_prime != Action.STAY:
        #         return
        #     if not np.array_equal(state, state_prime) and action_prime == Action.STAY
        #         return
        

        # TODO: Think about if i really want to do it this way...        
        # Get the row of the Q table corresponding to the state
        
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
        
        sigma = 0.5
        delta = 1/len(Action)

        Q_s_a_old = Q_s[a_index]
        Q_s_a_new = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_p_index] - Q_s[a_index])
        # print('Q_s_a_old', Q_s_a_old)
        # print('Q_s_a_new', Q_s_a_new)
        # print('self.q_epsilon[tuple(state)]', self.q_epsilon[tuple(state)])
        f_num = 1-np.exp(-abs(Q_s_a_new-Q_s_a_old)/sigma)
        f_den = 1+np.exp(-abs(Q_s_a_new-Q_s_a_old)/sigma)
        # print('f_num', f_num)
        # print('f_den', f_den)
        self.q_epsilon[tuple(state)] = delta*f_num/f_den + (1-delta)*self.q_epsilon[tuple(state)] 
        # print('self.q_epsilon[tuple(state)]', self.q_epsilon[tuple(state)])

        # Update the Q table at the state,action index pair
        self.q_data[tuple(state)][a_index]  = Q_s_a_new
        self.q_updates[tuple(state)] = self.q_updates[tuple(state)] + 1 
    
        
        
        # print('updated Q row is ', self.q_table[s_index])

    # Fetch the corresponding row of the Q table given a state
    #  If no row for this state exists, create one and return that 
    def fetch_row_by_state(self, state):
        if tuple(state) not in self.q_data:
            empty_row = np.zeros(len(Action))
            self.q_data.update({tuple(state):empty_row})
            self.q_updates.update({tuple(state):0})
            self.q_epsilon.update({tuple(state):1})
            return empty_row
        else:
            return self.q_data[tuple(state)]

    def fetch_updates_by_state(self, state):
        if tuple(state) not in self.q_data:
            empty_row = np.zeros(len(Action))
            self.q_data.update({tuple(state):empty_row})
            self.q_updates.update({tuple(state):0})
            self.q_epsilon.update({tuple(state):1})
            return 0
        else:
            return self.q_updates[tuple(state)]

    def fetch_epsilon_by_state(self, state):
        if tuple(state) not in self.q_data:
            empty_row = np.zeros(len(Action))
            self.q_data.update({tuple(state):empty_row})
            self.q_updates.update({tuple(state):0})
            self.q_epsilon.update({tuple(state):1})
            return 0
        else:
            return self.q_epsilon[tuple(state)]