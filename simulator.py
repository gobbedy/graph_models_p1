import numpy as np
import decoder
import transmitter
import math
from matplotlib import pyplot as plt

class Simulator:

    def __init__(self, num_iterations, H, R, G):
        self.num_iterations = num_iterations # number codewords to pass through transmitter and decoder when computing bit error rate
        self.H = H # parity matrix
        self.R = R # decoding matrix
        self.G = G # code generator matrix
        
        self.Transmitter = transmitter.Transmitter(G)
        self.Decoder = decoder.Decoder(20, H, R)

    def iteration(self, x, algorithm, std_deviation):
        
        z = self.Transmitter.transmit(x, std_deviation)
        r = self.Decoder.decode(z, algorithm, std_deviation)
        return np.sum(x!=r)
        
    
    def get_bit_error_rates(self, algorithm):
    
        # set x to zero for simplicity
        x = np.array([0, 0, 0, 0, 0, 0, 0]).T
        
        # list of variances over which to compute bit error rate (1, 1/2, 1/4, 1/8
        variances = 1/np.power(2, np.arange(1,5)-1)
        variance_bit_error_rate_table = np.zeros((2,len(variances))) # 2x4 matrix: first row is variances, second row is bit error rate
        variance_bit_error_rate_table[0] = variances

        # for each variance, find bit error rate
        for idx, variance in enumerate(variances):
            
            # get std deviation from variance
            std_deviation = math.sqrt(variance)
            
            # run the transmitter and decoder "num_iterations" times and add up number of bit errors
            bit_errors = 0
            for jdx in range(0, self.num_iterations):
                bit_errors += self.iteration(x, algorithm, std_deviation)
            
            # compute bit error rate and store in table
            bit_error_rate = bit_errors / self.num_iterations / len(x)
            variance_bit_error_rate_table[1, idx] = bit_error_rate

        # return variance / bit error rate table
        return variance_bit_error_rate_table


    def simulate(self):
    
        # get bit error rates for max product
        variance_bit_error_rate_table_log = np.log10(self.get_bit_error_rates(1))
        plt.figure()
        plt.xlabel('log10(variance)')
        plt.ylabel('log10(probability of bit error)')
        plt.title('Max product performance graph')
        plt.plot(variance_bit_error_rate_table_log[0], variance_bit_error_rate_table_log[1], 'bx')
        
        
        # get bit error rates for sum product
        variance_bit_error_rate_table_log = np.log10(self.get_bit_error_rates(1))
        plt.figure()
        plt.xlabel('log10(variance)')
        plt.ylabel('log10(probability of bit error)')
        plt.title('Sum product performance graph')
        plt.plot(variance_bit_error_rate_table_log[0], variance_bit_error_rate_table_log[1], 'rx')
        
        plt.show(block=False)

        var = input("Enter something to close plots and exit: ")
        plt.close('all')
