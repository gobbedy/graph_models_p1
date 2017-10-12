import numpy as np

class Transmitter:

    def __init__(self, G, use_hamming_encode=0):
        self.G  = G # code generator matrix
        self.use_hamming_encode = use_hamming_encode # if 1, input to transmitter assumed to be original message (m1, m2, m3, 4). if 0, input to transmitter assumed to be hamming code (x1, x2, x3, x4, x5, x6, x7)


    def hamming(self, m):
            '''
                    m is a 4 item array (the inital message): (m1, m2, m3, m4)
                    x is a 7 item array returned as the hamming encoded message: (x1, x2, x3, x4, x5, x6, x7)
            '''

            # multiply m with standard matrix G for hamming code encryption and take modulo 2
            x = np.remainder(G.dot(m), 2)
            return x


    def inversion(self, x):
            '''
                    x is a 7 item array encoded with HM(7,4): (x1, x2, x3, x4, x5, x6, x7)
                    y is a 7 item array with extended range from [-1, 1]: (y1, y2, y3, y4, y5, y6, y7)
            '''

            # creating deep copy of x
            y = np.copy(x)

            # if the element is equal to 1, it becomes -1
            # if the element is equal to 0, it becomes 1
            y[y==1] = -1
            y[y==0] = 1
            return y


    def noise(self, y, std_deviation):
            '''
                    y is a 7 item array that has been extended in range to [-1, 1]: (y1, y2, y3, y4, y5, y6, y7)
                    std_deviation is the standard deviation of the channel Ni, also described in the setup (setup mentions variance, which is just the square of the std deviation)
                    z is a 7 item array with added noise from gaussian distr: (z1, z2, z3, z4, z5, z6, z7)
            '''

            # add noise from gaussian normal curve with defined standard deviation
            z = np.zeros(len(y))
            for i in range(0, len(y)):
                z[i] = y[i] + np.random.normal(0, std_deviation)
            return z


    def transmit(self, m, std_deviation):
            '''
                    m is a 4 item array (the inital message): (m1, m2, m3, m4)
                    z is a 7 item array with added noise from gaussian distr: (z1, z2, z3, z4, z5, z6, z7)
            '''

            if self.use_hamming_encode:
                x = self.hamming(m)
            else:
                x = m
            y = self.inversion(x)
            z = self.noise(y, std_deviation)
            return z
