import numpy as np
import decoder
import transmitter
import math
import pylab

class Simulator:

    def __init__(self, num_iterations, H, R, G):
        self.num_iterations = num_iterations # number codewords to pass through transmitter and decoder when computing bit error rate
        self.H = H # parity matrix
        self.R = R # decoding matrix
        self.G = G # code generator matrix
        
        self.Transmitter = transmitter.Transmitter(G)
        self.Decoder = decoder.Decoder(20, H, R)

    def iteration(self, x, algorithm, std_deviation):
        
        # send the codeword thru the transmitter (includes channel aka gaussian noise)
        z = self.Transmitter.transmit(x, std_deviation)
        
        # send transmitted codeword thru decoder -- r is decoder's best guess for x, use maxproduct or sumproduct as specified by 'algorithm'
        r = self.Decoder.decode(z, algorithm, std_deviation)
        return np.sum(x!=r)
        
    
    def get_bit_error_rates(self, algorithm):

        # set x to zero for simplicity
        x = np.zeros((7,1))
        
        # list of variances over which to compute bit error rate (0.1, 0.2, 0.3 ... 1.0)
        variances = np.arange(1,11)/10
        variance_bit_error_rate_table = np.zeros((2,len(variances))) # 2x4 matrix: first row is variances, second row is bit error rate
        
        # add first row to table (variances)
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
            
            # add second row to table (bit error rates for each variance) one by one
            variance_bit_error_rate_table[1, idx] = bit_error_rate

        # return variance / bit error rate table
        return variance_bit_error_rate_table


    def simulate(self):
    
        # TODO: show more numbers on the axes, looks empty now
        # maybe try one of these
        # https://stackoverflow.com/questions/16830520/how-can-i-label-the-minor-tics-in-a-loglog-plot-in-matplotlib
        # https://stackoverflow.com/questions/6567724/matplotlib-so-log-axis-only-has-minor-tick-mark-labels-at-specified-points-also

        # prepare plot with x/y labels and title
        pylab.xlabel('Variance')
        pylab.ylabel('Probability of Bit Error')
        pylab.title('(7,4) Hamming + Gaussian Decoder Performance (Max Product vs Sum Product)')
    
        # get bit error rates for max product, and plot
        variance_bit_error_rate_table_log = self.get_bit_error_rates(1)
        print("Max product variance vs probability error:")
        print(variance_bit_error_rate_table_log)
        pylab.loglog(variance_bit_error_rate_table_log[0], variance_bit_error_rate_table_log[1], 'bx', label='Max Product')
        
        
        # get bit error rates for sum product, and plot (on same figure)
        variance_bit_error_rate_table_log = self.get_bit_error_rates(0)
        print("Sum product variance vs probability error:")
        print(variance_bit_error_rate_table_log)
        pylab.loglog(variance_bit_error_rate_table_log[0], variance_bit_error_rate_table_log[1], 'rx', label='Sum Product')

        # add legend on graph
        pylab.legend(loc='lower right')
        
        # now that's ready, render the plot. block=False allows code to continue instead of waiting for user to manually close graph        
        pylab.show(block=False)

        # wait for user input otherwise script ends and plots automatically closed
        var = input("Enter something to close plots and exit: ")
        
        # close graphs before ending (not sure if needed, here for rigour)
        pylab.close('all')
