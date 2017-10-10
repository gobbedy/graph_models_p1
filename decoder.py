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
        upstream_entries_col_indices_bool = row(row!=0) # 1x7 boolean array with True only where H is 1 (ie in 4 places)
        upstream_entries_col_indices_bool[col_idx] = False # also remove the current column (down to 3 entries)
        upstream_entries_col_indices = np.where(upstream_entries_col_indices_bool) # find the indices of the "True" values
        upstream_entries = V[row_idx, upstream_entries_col_indices] # extract the entries from V -- 3x2 matrix

        # TODO: sum product
        # TODO: variable node


        # compute the message passed from this function node to the downstream variable -- ie the max product of the upstream variables and the current node's function
        F[row_idx, col_idx] = maxproduct(upstream_entries)          

def maxproduct(upstream_entries):
'''
        For any particular function node, there are 3 upstream variables, xi, each passing in a message Mxi(xi), and one downstream variable, xj
        For the sake of this example we let the xi be x1, x2, x3, and the xj be x4 -- without loss of generality as the reasoning applies to any xi, xj in a cycle
        We compute Mx1(x1)*Mx2(x2)*M(x3)*f(x1,x2,x3,x4) maxed over (x1,x2,x3) -- we call this m(x4) (message to x4)
        
        STEP 1 -- find the max of each individual upstream variable message (optimization to reduce computation)
        We know that max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3)) exists for some x1=a1, x2=a2, x3=a3.
        And regardless of a1, a2, a3, there exists an x4=a4 such that a1 + a2 + a3 + a4 = 0 mod 2, ie such that f(x1,x2,x3,x4)=1
        Its obvious that the a4 that satisfies this requirement is a4=a1+a2+a3 mod 2
        Therefore since f(x1,x2,x3,x4) cannot be higher than 1, we know that m(a4) = max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3))
        
        Step 2
        For x4=(1-a4), we must have that x1 + x2 + x3 = 1 - a4 mod 2 in order to obtain f(x1, x2, x3, x4) = 1.
        There are 4 permutations of (x1, x2, x3) that satisfy this equation.
        To find the max of m(1-a4), we find Mx1(x1)*Mx2(x2)*M(x3) for each permutation then take the max of these 4 results
        
'''
        # STEP 1
        
        # find the max of each msg coming from upstream variables
        upstream_entries_max = np.amax(upstream_entries, 1) # max(Mx1(x1)), max(Mx2(x2)), max(Mx3(x3))
        upstream_entries_argmax = np.argmax(upstream_entries, 1) # a1, a2, a3 in description example
           
        # get the product of these maxes
        upstream_entries_max_product = np.product(upstream_entries_max) # max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3))
        
        # compute the message for the downstream variable evaluated at upstream_entries_max_argmax_sum (which is just the maxsum computed above, since the lambda function evaluates to 1)
        upstream_entries_max_argmax_sum = upstream_entries_argmax.sum(upstream_entries_argmax)%2
        node_function[upstream_entries_max_argmax_sum] = upstream_entries_max_product
        
        # for the max of the downstream var evaluated at (1 - upstream_entries_max_argmax_sum), the sum of the downstream vars must be (1 - upstream_entries_max_argmax_sum)
        
        # indices of dowstream variable permutations whose sum (mod 2) adds up to (1 - upstream_entries_max_argmax_sum)
        desired_indices_arrays = equi_count_generator(1 - upstream_entries_max_argmax_sum)
        
        # convert this array to tuple
        #desired_indices_tuple = tuple(map(tuple,desired_indices_tuple))
        
        # for each valid permutation of the downstream vars, multiply the messages
        products=numpy.zeros(len(desired_indices_arrays))
        for idx, desired_indices_array in enumerate(desired_indices_arrays):
          products[idx] = np.product(upstream_entries[desired_indices_array])
        
        # find the max message
        node_function[1 - upstream_entries_max_argmax_sum] = np.amax(products, 1)
    

def equi_count_generator(desired_parity):
'''
   desired_parity is an integer with value either 0 or 1
   returns an array of all arrays of size 3 with bitcount%2 equal to desired_parity -- [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]
   if desired_parity is 1, returns an array of all arrays of size 3 with bitcount%2 equal to 1 -- [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]
   equi_count_arrays is the returned array described above
   
   for generalizing with values above 8, can use ideas from http://p-nand-q.com/python/algorithms/math/bit-parity.html
'''
  # all integers between 0 and 7 with bit parity equal to desired_parity
  equi_count_int = np.where(((0x6996 >> x) & 1) == desired_parity)[0]
  
  # convert the desired indices to array of arrays (eg 1 --> [1, 0, 0], 5, [1, 0, 1], etc)
  equi_count_arrays=((equi_count_int[:, None] & (1 << np.arange(3))) > 0)*1
