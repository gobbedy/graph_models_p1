import numpy as np
from scipy.stats import norm

H = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]])
num_iterations = 20

def decode(z, std_deviation):
'''
  z is a 7 item array: (z1, z2, z3, z4, z5, z6, z7). These are described in the project 1 setup
  std_deviation is the known standard deviation of the channel Ni, also described in the setup (setup mentions variance, which is just the square of the std deviation)
  decode returns x, also a 7 value array: (x1, x2, x3, x4, x5, x6, x7)
'''

  # TODO: V, F need to contain two values for each entry!! 

  # variable to function node matrix (initialized to its content before the first iteration)
  V = np.zeros((H.shape + (2,)))

  # function to variable node matrix (initialized to its content after the first iteration -- first iteration will be recomputed anyway)
  F = np.zeros((H.shape + (2,)))
  
  # set V and F to its initial values (all ones, except where H is 0)
  

  # compute the static node messages (not in cycles), ie the Pzi|xi -> xi nodes
  # we store these in a 7 item value, m, where m(i-1) is Pzi|xi -> xi -- m(i-1) and not m(i) since zero based
  # each m(i-1) is a 2 value array where m(i-1)(j) corresponds to x(i)=j
  m = np.zeros((len(sz), 2))
  for idx, zi in enumerate(z):
    # x(i) = 0 is transmitted as -1 so gaussian curve has mean -1
    m[idx, 0] = norm.pdf(zi, -1, std_deviation)

    # x(i) = 0 is transmitted as 1 so gaussian curve has mean 1
    m[idx, 1] = norm.pdf(zi, 1, std_deviation)
  
  # each m(i-1) is now a function of x(i). since it will be used repeatedly, we compute the max of each function x(i)
  # m_max is a 7 value array
  #m_max = np.amax(m, 1)
  #m_argmax = np.argmax(m, 1)

  for iteration in range(0, num_iterations):
    # fill up F
    #for idx in np.ndenumerate(V): # row_idx=idx(0); col_idx=idx(1)
    for row_idx, row in enumerate(H):
      for col_idx, col in enumerate(row):
        # multiply the incoming messages together (ie the messages in the variable matrix from the upstream variables, so excluding the downstream variable)
        # to do so select all column indices of V except the column index of the current var under consideration
        # if there is no connection between the two nodes currently considered, output zero hence use of H matrix
        
        # get only the relevant entries of V, representing the upstream variables coming into the current node function
        # (ie the current row, less all the zero entries of H, less the current column index)
        v_entries_col_indices_bool = row(row!=0)
        v_entries_col_indices_bool[col_idx] = False
        v_entries_col_indices = where(v_entries_col_indices_bool)
        

        # find the max of each msg coming from upstream variables (NOTE: for sum product, this part will change)
        v_entries_max = np.zeros(len(v_entries_col_indices))
        v_entries_argmax = np.zeros(len(v_entries_col_indices))
        for i, v_col_idx in enumerate(v_entries_col_indices):
           v_entries_max[i] = np.amax(V[row_idx, v_col_idx])
           v_entries_argmax[i] = np.argmax(V[row_idx, v_col_idx])
           
        # get the product of these maxes
        v_entries_max_product = np.product(v_entries_max)
        
        # compute the message for the downstream variable evaluated at v_entries_max_argmax_sum (which is just the maxsum computed above, since the lambda function evaluates to 1)
        v_entries_max_argmax_sum = v_entries_argmax.sum(v_entries_argmax)%2
        node_function[v_entries_max_argmax_sum] = v_entries_max_product
        
        # for the max of the downstream var evaluated at (1 - v_entries_max_argmax_sum), the sum of the downstream vars must be (1 - v_entries_max_argmax_sum)
        v_entries = V[row_idx, v_entries_col_indices] # 3x2 matrix
        
        
        
        v_entries_to_multiply = V[row_idx, v_entries_col_indices_bool] # 3x2 matrix
        for i, v_col_idx in enumerate(v_entries_col_indices):
         
        msg = V[row_idx, np.arange(H.shape[1])!=col_idx] # 6x2 matrix
        msg = H[row_idx, col_idx] * np.product(msg)
        
        # 
        F[row_idx, col_idx]
        for v_col_idx in range(0, len(row)):
          if v_col_idx != col_idx and H[v_col_idx, col_idx):
            msg *= V[v_col_idx, col_idx)
          

# 1. pseudo-code for transmitter
# 2. 
