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

  # variable to function node matrix (initialized to its content before the first iteration)
  V = numpy.copy(H)

  # function to variable node matrix (initialized to its content after the first iteration -- first iteration will be recomputed anyway)
  F = numpy.copy(H)

  # compute the static node messages (not in cycles), ie the Pzi|xi -> xi nodes
  # we store these in a 7 item value, m, where m(i-1) is Pzi|xi -> xi -- m(i-1) and not m(i) since zero based
  # each m(i-1) is a 2 value array where m(i-1)(j) corresponds to x(i)=j
  m = numpy.zeros((len(sz), 2))
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
    for row_idx, row in enumerate(F):
      for col_idx, col in enumerate(row):
        #F[row_idx, col_idx]
        msg = 1
        #following two lines do the same as the loop below, just harder to understand
        #msg = V[row_idx, np.arange(H.shape[1])!=col_idx]
        #msg = H[row_idx, col_idx] * np.product(msg[msg!=0])
        # multiply the incoming messages together (ie the messages in the variable matrix from the upstream variables, so excluding the downstream variable)
        for v_col_idx in range(0, len(row)):
          if v_col_idx != col_idx and H[v_col_idx, col_idx):
            msg *= V[v_col_idx, col_idx)
          

