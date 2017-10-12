#!/usr/bin/python3.4
import numpy as np
from scipy.stats import norm
from scipy import misc
import math

debug = 0
hamming_decode=0

# parity matrix
H = np.array([[1,0,1,0,1,0,1],[0,1,1,0,0,1,1],[0,0,0,1,1,1,1]])

# decoding matrix
R = np.array([[0,0,1,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])

num_iterations = 20

def decode(z, std_deviation, use_maxproduct=1):
    '''
      z is a 7 item array: (z1, z2, z3, z4, z5, z6, z7). These are described in the project 1 setup
      std_deviation is the known standard deviation of the channel Ni, also described in the setup (setup mentions variance, which is just the square of the std deviation)
      use_maxproduct is the algorithm used for decoding. 1 for max product, 0 for sum product
      decode returns x, also a 7 value array: (x1, x2, x3, x4, x5, x6, x7)
    '''

    # variable to function node matrix (initialized to its content before the first iteration)
    V = np.zeros((H.shape + (2,)))

    # function to variable node matrix (initialized to its content after the first iteration -- first iteration will be recomputed anyway)
    F = np.zeros((H.shape + (2,)))

    # V and F have been set to their initial log values (all zeros) -- note that where H is zero, the entries will be ignored anyway.


    # compute the static node messages (not in cycles), ie the Pzi|xi -> xi nodes
    # we store these in a 7 entry array, m, where m(i-1) is Pzi|xi -> xi -- m(i-1) and not m(i) since zero based
    # each m(i-1) is a 2 entry array where m(i-1)(j) corresponds to x(i)=j
    m = np.zeros((len(z), 2))
    for idx, zi in enumerate(z):

        # x(i) = 0 is transmitted as -1 so gaussian curve has mean -1
        m[idx, 0] = math.log(norm.pdf(zi, 1, std_deviation))

        # x(i) = 0 is transmitted as 1 so gaussian curve has mean 1
        m[idx, 1] = math.log(norm.pdf(zi, -1, std_deviation))
    
    for iteration in range(0, num_iterations):

        # fill up F
        for row_idx, row in enumerate(H):

            for col_idx, entry in enumerate(row):

                # if H entry is zero, these two nodes are not connected      
                if entry == 0:
                    continue

                # multiply the incoming messages together (ie the messages in the variable matrix from the upstream variables, so excluding the downstream variable)
                # to do so select all column indices of V except the column index of the current var under consideration
                # if there is no connection between the two nodes currently considered, output zero hence use of H matrix
                
                # note that multiplication in log domain is addition, so we sum wherever multiplication is required

                # get only the relevant entries of V, representing the upstream variables coming into the current node function
                # (ie the current row, less all the zero entries of H, less the current column index)
                upstream_entries_col_indices_bool = row!=0 # 1x7 boolean array with True only where H is 1 (ie in 4 places)
                upstream_entries_col_indices_bool[col_idx] = False # also remove the current column (down to 3 entries)
                upstream_entries_col_indices = np.where(upstream_entries_col_indices_bool)[0] # find the indices of the "True" values
                upstream_entries = V[row_idx, upstream_entries_col_indices] # extract the entries from V -- 3x2 matrix
                
                #if np.isnan(upstream_entries).any():
                #    print("bad entries")
                #    exit()
                #print("a")
                #print(upstream_entries)
                #print("b")

                # TODO: debug sum product -- try mao's algorithm
                # TODO: verify code via simulation
                # TODO: sum product (note that only the maxproduct line below changes)

                # compute the message passed from this function node to the downstream variable -- ie the max product of the upstream variables and the current node's function
                if use_maxproduct:
                    F[row_idx, col_idx] = maxproduct(upstream_entries)
                else:
                    F[row_idx, col_idx] = sumproduct(upstream_entries)

        # fill up V
        #print("F after iteration: " + str(iteration))
        #print(F[:,:,:])
        #print("end F")
        #exit()
        for col_idx, col in enumerate(H.T):

            for row_idx, entry in enumerate(col):

                # if H entry is zero, these two nodes are not connected      
                if entry == 0:
                    continue

                # multiply the incoming messages together (ie the messages in the variable matrix from the upstream functions, so excluding the downstream function)
                # to do so select all column indices of F except the row index of the current function under consideration
                # if there is no connection between the two nodes currently considered, output zero hence use of H matrix

                # get only the relevant entries of F, representing the upstream function coming into the current variable node
                # (ie the current row, less all the zero entries of H, less the current column index)
                upstream_entries_row_indices_bool = col!=0 # 1xn boolean array with True only where H is 1 (ie in up to 3 places, so n<=3)
                upstream_entries_row_indices_bool[row_idx] = False # also remove the current row (down to 2 or less entries)
                upstream_entries_row_indices = np.where(upstream_entries_row_indices_bool)[0] # find the indices of the "True" values
                upstream_entries = F[upstream_entries_row_indices, col_idx] # extract the entries from F -- cx2 matrix, where c is number of upstream function nodes

                # compute the message passed from this variable node to the downstream function -- ie the max product of the upstream variables and the current node's function                
                if len(np.sum(upstream_entries, 0)) == 0:
                    V[row_idx, col_idx] = m[col_idx]
                    #print("x")
                else:
                    #if np.isnan(np.sum(upstream_entries, 0)).any():
                    #    print("V nan error")
                    #    exit()
                    V[row_idx, col_idx] = np.sum(upstream_entries, 0) + m[col_idx]

                # Note: what happens when upstream_entries is empty? which happens when variable node not in a cycle:
                # np.sum() returns 0 for an empty array, yet the result is m(i), hence the if statement above

        #print("V after iteration: " + str(iteration))
        #print(V[:,:,:])
        #print("end V")

        
        if debug == 1:
            #np.set_printoptions(precision=4)
            print("AFTER ITERATION " + str(iteration + 1))
            print("all m")
            print(m)
            print("all H")
            print(H)
            print("F for all x=0")
            print(F[:,:,0])
            print("F for all x=1")
            print(F[:,:,1])
            print("V for all x=0")
            print(V[:,:,0])
            print("V for all x=1")
            print(V[:,:,1])

            if iteration == 1:
                return 1


    # find max likelihood for xi
    x = np.zeros(z.shape)
    for col_idx, col in enumerate(H.T):

          # multiply the messages together (ie the messages in the variable matrix from all the connected functions)

          # get only the relevant entries of F, representing the upstream variables coming into the current node function
          # (ie the current row, less all the zero entries of H, less the current column index)
          msg_entries_row_indices_bool = col!=0 # 1xn boolean array with True only where H is 1 (ie in up to 3 places, so n<=3)
          msg_entries_row_indices = np.where(upstream_entries_row_indices_bool)[0] # find the indices of the "True" values
          msg_entries = F[msg_entries_row_indices, col_idx] # extract the entries from F -- cx2 matrix, where c is number of connected function nodes 

          # compute the summary message at this variable -- ie the max product of the upstream variables and function nodes
          summary_msg = np.sum(msg_entries, 0) + m[col_idx]
          
          x[col_idx] = np.argmax(summary_msg)

    if hamming_decode == 1:
        return hamming_decode(x)
    else:
        return x

def sumproduct(upstream_entries):
    '''
            Input: upstream_entries
               3 x 2 matrix contain the incoming message from every upstream variable node (each row is a message function Mxi(xi) where xi are the upstream nodes)
            
            Ouput: node_msg
               outgoing message to downstream variable (2 entry array representing a function Mxj(xj) where xj is a downstream node)
            
            
            For any particular function node, there are 3 upstream variables, xi, each passing in a message Mxi(xi), and one downstream variable, xj
            For the sake of this example we let the xi be x1, x2, x3, and the xj be x4 -- without loss of generality as the reasoning applies to any xi, xj in a cycle
            We compute Mx1(x1)*Mx2(x2)*M(x3)*f(x1,x2,x3,x4) summed over (x1,x2,x3) -- we call this m(x4) (message to x4)

            STEP 1
            For x4=0, if x1 + x2 + x3 = 1, f(x1,x2,x3,x4) = 0, so all such permutations do not contribute to the sum
            if x1 + x2 + x3 = 0, f(x1,x2,x3,x4) = 1, so we find Mx1(x1)*Mx2(x2)*M(x3) summed over those permutations of x1, x2, x3

            Step 2
            For x4=1 we use the same procedure as STEP 1 except that we use the permutations where x1 + x2 + x3 = 1
            
            # note1: since we are in the log domain, we sum wherever multiplication is required
            # note2: since we are in the log domain, we use logsum where ever a sum is required

    '''
    
    node_msg = np.zeros(2)
    
    for downstream_var_value in range(0,2):
        # indices of dowstream variable permutations whose sum (mod 2) adds up to (1 - upstream_entries_max_argmax_sum)
        desired_indices_arrays = equi_count_generator(downstream_var_value) # 4 permutations of (x1, x2, x3)

        # for each valid permutation of the downstream vars, multiply the messages (Mx1(x1)*Mx2(x2)*M(x3) for each permutation)
        products=np.zeros(len(desired_indices_arrays))
        for idx, desired_indices_array in enumerate(desired_indices_arrays):
            #total_sum=0
            #print(upstream_entries)
            #print(desired_indices_array)
            #print(upstream_entries[range(0,len(upstream_entries)), desired_indices_array])
            #for ele in upstream_entries[range(0,len(upstream_entries)), desired_indices_array]:
            #    if math.isnan(ele):
            #        print("not a number, error")
            #        exit()
            #    total_sum = logsum(total_sum, ele)            
            products[idx] = np.sum(upstream_entries[range(0,len(upstream_entries)), desired_indices_array])

            # np_logsum = np.frompyfunc(logsum)
            # products[idx] = np_logsum.reduce(upstream_entries[:, desired_indices_array])
        
        # find the sum of the permutations
        #print(misc.logsumexp(products))
        node_msg[downstream_var_value] = misc.logsumexp(products)

    return node_msg


def maxproduct(upstream_entries):
    '''
            Input: upstream_entries
               3 x 2 matrix contain the incoming message from every upstream variable node (each row is a message function Mxi(xi) where xi are the upstream nodes)
            
            Ouput: node_msg
               outgoing message to downstream variable (2 entry array representing a function Mxj(xj) where xj is a downstream node)
            
            
            For any particular function node, there are 3 upstream variables, xi, each passing in a message Mxi(xi), and one downstream variable, xj
            For the sake of this example we let the xi be x1, x2, x3, and the xj be x4 -- without loss of generality as the reasoning applies to any xi, xj in a cycle
            We compute Mx1(x1)*Mx2(x2)*M(x3)*f(x1,x2,x3,x4) maxed over (x1,x2,x3) -- we call this m(x4) (message to x4)

            STEP 1 -- find the max of each individual upstream variable message (optimization to reduce computation)
            We know that max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3)) exists for some x1=a1, x2=a2, x3=a3.
            And regardless of a1, a2, a3, there exists an x4=a4 such that a1 + a2 + a3 + a4 = 0 mod 2, ie such that f(x1,x2,x3,x4)=1
            Its obvious that the a4 that satisfies the equation a4=a1+a2+a3 mod 2
            Therefore since f(x1,x2,x3,x4) cannot be higher than 1, we know that m(a4) = max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3))

            Step 2
            For x4=(1-a4), we must have that x1 + x2 + x3 = 1 - a4 mod 2 in order to obtain f(x1, x2, x3, x4) = 1.
            There are 4 permutations of (x1, x2, x3) that satisfy this equation.
            To find the max of m(1-a4), we find Mx1(x1)*Mx2(x2)*M(x3) for each permutation then take the max of these 4 results
            
            # note1: since we are in the log domain, we sum wherever multiplication is required
            # note2: since log is a monotonic increasing function, max(loga, logb) = log(max(a,b)), so we still max in log domain

    '''
    
    node_msg = np.zeros(2)        
    
    # STEP 1
    
    # find the max of each msg coming from upstream variables
    upstream_entries_max = np.amax(upstream_entries, 1) # max(Mx1(x1)), max(Mx2(x2)), max(Mx3(x3))
    upstream_entries_argmax = np.argmax(upstream_entries, 1) # a1, a2, a3 in description example
       
    # get the product of these maxes
    upstream_entries_max_product = np.sum(upstream_entries_max) # max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3))
    
    # compute the message for the downstream variable evaluated at upstream_entries_max_argmax_sum
    upstream_entries_max_argmax_sum = np.sum(upstream_entries_argmax)%2  # a4=a1+a2+a3 mod 2
    node_msg[upstream_entries_max_argmax_sum] = upstream_entries_max_product # m(a4) = max(Mx1(x1)) * max(Mx2(x2)) * max(Mx3(x3))
    
    
    # STEP 2
    
    # indices of dowstream variable permutations whose sum (mod 2) adds up to (1 - upstream_entries_max_argmax_sum)
    desired_indices_arrays = equi_count_generator(1 - upstream_entries_max_argmax_sum) # 4 permutations of (x1, x2, x3)
    
    # for each valid permutation of the downstream vars, multiply the messages (Mx1(x1)*Mx2(x2)*M(x3) for each permutation)
    products=np.zeros(len(desired_indices_arrays))
    for idx, desired_indices_array in enumerate(desired_indices_arrays):
        products[idx] = np.sum(upstream_entries[:, desired_indices_array])
    
    # find the max message (m(1-a4) is the max of these 4 results)
    node_msg[1 - upstream_entries_max_argmax_sum] = np.amax(products)

    return node_msg

def equi_count_generator(desired_parity):
    '''
       desired_parity is an integer with value either 0 or 1
       returns an array of all arrays of size 3 with bitcount%2 equal to desired_parity
       if desired_parity is 0, returns an array of all arrays of size 3 with bitcount%2 equal to 0 -- [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]
       if desired_parity is 1, returns an array of all arrays of size 3 with bitcount%2 equal to 1 -- [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]
       equi_count_arrays is the returned array described above

       for generalizing with values above 8, can use ideas from http://p-nand-q.com/python/algorithms/math/bit-parity.html
    '''
    # all integers between 0 and 7 with bit parity equal to desired_parity
    equi_count_int = np.where(((0x6996 >> np.arange(8)) & 1) == desired_parity)[0]

    # convert the desired indices to array of arrays (eg 1 --> [1, 0, 0], 5 --> [1, 0, 1], etc)
    equi_count_arrays=((equi_count_int[:, None] & (1 << np.arange(3))) > 0)*1
    
    return equi_count_arrays


def hamming_decode(x):
    '''
	    m is a 4 entry array (the recovered message): (m1, m2, m3, m4)
	    x is a 7 entry array of mostly likely encoded message: (x1, x2, x3, x4, x5, x6, x7)
    '''

    # multiplyx decoding matrix R with x and take modulo 2
    m = np.remainder(R.dot(x), 2)
    return m

def logsum(x,y):
    diff=x-y
    if diff>23:
        return x
    elif diff <-23:
        return y
    else:
        return (y + math.log(math.exp(diff)+1))
    


z = np.array([-1.5, -1.5, -1.5,  1.5,  1.5,  1.5,  1.5]).reshape((7,1))
x = decode(z, 1, 0)
print(x)
