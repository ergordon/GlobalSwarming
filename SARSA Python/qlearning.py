import numpy as np
import action as action

class Qlearning:


    def __init__(self):
        print('q constructor')
        #store Q table 
        self.q_table = np.array([])
        #store state->Q table association
        self.q_states = np.array([])


    #update 
    def update_q(self, state, state_prime, action, action_prime, alpha, gamma, reward):
        # TODO think about if i really want to do it this way...        
        
        s_index = self.fetch_row_index_by_state(state)
        Q_s = self.fetch_row_by_state(state)
        Q_s_p = self.fetch_row_by_state(state_prime)
        

        # print('state is')
        # print(state)
        # print('state prime is')
        # print(state_prime)
        # print('action is')
        # print(action)
        # print('action prime is')
        # print(action_prime) 
        # print('alpha is')
        # print(alpha)
        # print('gamma is')
        # print(gamma)
        # print('reward is')
        # print(reward)

        # print('Q table is')
        # print(self.q_table)
        # print('Q states is')
        # print(self.q_states)

        # print('state index is')
        # print(s_index)
        

        # print('Q_s is ')
        # print(Q_s)
        # print('Q_s_p is ')
        # print(Q_s_p)


        
        # print(action)
        # print(action_prime)

        a_index = action.value
        a_p_index = action_prime.value

        # self.q_table[s_index][a_p_index]  = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_index] - Q_s[a_index])
        # print("debug start")
        # print(a_index)
        # print(Q_s[a_index])
        # print(alpha*(reward + gamma*Q_s_p[a_index] - Q_s[a_index]))
        
        # self.q_table[s_index, a_p_index]  = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_index] - Q_s[a_index])
        self.q_table[s_index, a_index]  = Q_s[a_index] + alpha*(reward + gamma*Q_s_p[a_p_index] - Q_s[a_index])


        # print('updated q_s is')
        # print(self.q_table[s_index,])

    #fetch the index of the row in the Q table that corresponds
    #to a given state. If now row exist for the state,  
    # make one and return the index
    def fetch_row_index_by_state(self, state):
        #TODO research a better (more efficeint) way of doing this
        # index = -1
        
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
            #tables are empty, put ours in
            self.q_table = np.zeros((1,len(action.Action)))
            self.q_states = state.reshape((1,state.shape[0]))
            
            # self.q_table = np.zeros(len(action.Action))
            # self.q_states = state
            index = 0

        # print(index)
        return index        

    
    

    #fetch the corresponding row of the Q table given a state
    # if no row for this state exists, create one and return that 
    def fetch_row_by_state(self, state):
        
        # print("Q table is: ")
        # print(self.q_table)
        # print("State is: ")
        # print(state)
        # print("states are: ")
        # print(self.q_states)


        #TODO research a better (more efficeint) way of doing this
        index = -1
        
        if self.q_states.size != 0: #check for empty matrix
            # print("checkign if state exists in qstates")
            # print("state is")
            # print(state)
            # print("q states is")
            # print(self.q_states)
            for i in range(0,self.q_states.shape[0]): #iterate over each row
                # print("q states row is")
                # print(self.q_states[i])
                if np.array_equal(state, self.q_states[i]): #if a row matches

                    index = i
                    break

            #row not found, add an entry to the q_table and q_states    
            if index == -1:
                print("row not found, adding entry")
                empty_row = np.zeros(len(action.Action))
                self.q_states = np.vstack([self.q_states, state])
                self.q_table = np.vstack([self.q_table, empty_row])
                index = self.q_states.shape[0]-1

        else: 
            #tables are empty, put ours in
            print("tables not initialized, do so now")
            self.q_table = np.zeros((1,len(action.Action)))
            # print("q table is")
            # print(self.q_table)
            self.q_states = state.reshape((1,state.shape[0]))
            # print("states are")
            # print(self.q_states)
            index = 0
            

        # print("the fectched Q row is: ")
        if(self.q_table.ndim == 1):
            # print(self.q_table)
            return self.q_table
        else:
            # print(self.q_table[index])
            return self.q_table[index]

        # print(self.q_table[0])
        # print(self.q_table.shape)
        # print(len(self.q_table.shape))

        # print(self.q_table[index,:])
        # return self.q_table[index,:]
        
        
    
