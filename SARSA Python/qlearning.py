import numpy as np
import action as action

##############################################################################
#   Q learning class
##############################################################################

#the Q learning class handles the storage and update for the Q table
class Qlearning:

    #class constructor
    def __init__(self):
        #store Q table 
        self.q_table = np.array([])
        #store state->Q table association
        self.q_states = np.array([])


    #update the Q table and state array
    def update_q(self, state, state_prime, action, action_prime, alpha, gamma, reward):
        # TODO think about if i really want to do it this way...        
        
        #get the index of the state in the Q state table, given the passed in state
        s_index = self.fetch_row_index_by_state(state)

        #get the row of the Q table corresponding to the state
        Q_s = self.q_table[s_index]

        #get the row of the Q table corresponding to the passed in state_prime
        Q_s_p = self.fetch_row_by_state(state_prime)
        
        #get the numerical index values for the Action enumeration
        a_index = action.value        
        a_p_index = action_prime.value

        #update the Q table at the state,action index pair
        self.q_table[s_index, a_index]  = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_p_index] - Q_s[a_index])



    #fetch the index of the row in the Q table that corresponds
    #to a given state. If now row exist for the state, make one
    def fetch_row_index_by_state(self, state):
        
        #TODO research a better (more efficeint) way of doing this
        index = -1
       
        if self.q_states.shape[0] != 0: #check for empty matrix

            matches = np.equal(self.q_states,[state]).all(1).nonzero()

            if matches[0].size == 0:
                #state not in q states add it along with the row
                empty_row = np.zeros(len(action.Action))
                self.q_states = np.vstack([self.q_states, np.copy(state)])
                self.q_table = np.vstack([self.q_table, empty_row])
                
                index = self.q_states.shape[0]-1
            else:
                #working state already in q states for this module, 
                #sum the working q row with the corresponding entry in the q table for this module
                #incerement the number of times this row has been updated
                index = matches[0][0] 
                
        else: 
            #tables are empty, put ours in
            self.q_table = np.zeros((1,len(action.Action)))
            self.q_states = np.copy(state.reshape((1,state.shape[0])))
            index = 0

        return index


        # #TODO research a better (more efficeint) way of doing this
        # index = -1

        # if self.q_states.shape[0] != 0: #check for empty matrix
        #     for i in range(0,self.q_states.shape[0]): #iterate over each row
        #         if np.array_equal(state, self.q_states[i]): #if a row matches
        #             index = i
        #             break

        #     #row not found, add an entry to the q_table and q_states    
        #     if index == -1:
        #         empty_row = np.zeros(len(action.Action))
        #         self.q_states = np.vstack([self.q_states, np.copy(state)])
        #         self.q_table = np.vstack([self.q_table, empty_row])
        #         index = self.q_states.shape[0]-1

        # else: 
        #     #tables are empty, put ours in
        #     self.q_table = np.zeros((1,len(action.Action)))
        #     self.q_states = np.copy(state.reshape((1,state.shape[0])))
            
        #     index = 0

        # return index        


    #fetch the corresponding row of the Q table given a state
    # if no row for this state exists, create one and return that 
    def fetch_row_by_state(self, state):
       

        #TODO research a better (more efficeint) way of doing this
        index = -1
       
        if self.q_states.shape[0] != 0: #check for empty matrix

            matches = np.equal(self.q_states,[state]).all(1).nonzero()

            if matches[0].size == 0:
                #state not in q states add it along with the row
                empty_row = np.zeros(len(action.Action))
                self.q_states = np.vstack([self.q_states, np.copy(state)])
                self.q_table = np.vstack([self.q_table, empty_row])
                
                index = self.q_states.shape[0]-1
            else:
                #working state already in q states for this module, 
                #sum the working q row with the corresponding entry in the q table for this module
                #incerement the number of times this row has been updated
                index = matches[0][0] 
                
        else: 
            #tables are empty, put ours in
            self.q_table = np.zeros((1,len(action.Action)))
            self.q_states = np.copy(state.reshape((1,state.shape[0])))
            index = 0

        return self.q_table[index]

        


        # #TODO research a better (more efficeint) way of doing this
        # index = -1
       
        # if self.q_states.shape[0] != 0: #check for empty matrix
        #     for i in range(0,self.q_states.shape[0]): #iterate over each row
        #         if np.array_equal(state, self.q_states[i]): #if a row matches
        #             index = i
        #             break

        #     #row not found, add an entry to the q_table and q_states    
        #     if index == -1:
        #         empty_row = np.zeros(len(action.Action))
        #         self.q_states = np.vstack([self.q_states, np.copy(state)])
        #         self.q_table = np.vstack([self.q_table, empty_row])
                
        #         index = self.q_states.shape[0]-1
                
        # else: 
        #     #tables are empty, put ours in
        #     self.q_table = np.zeros((1,len(action.Action)))
        #     self.q_states = np.copy(state.reshape((1,state.shape[0])))
        #     index = 0

        # return self.q_table[index]
        
    
##############################################################################
#   Q learning Class
##############################################################################