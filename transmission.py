import numpy as np


G = np.array([1,1,0,1],[1,0,1,1],[1,0,0,0],[0,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1])


def hamming(m):
	'''
		m is a 4 item array (the inital message): (m1, m2, m3, m4)
	'''

	# multiply m with standard matrix G for hamming code encryption and take modulo 2
	x = np.remainder(G.dot(m), 2)
	return x


def inversion(x):
	'''
		x is a 7 item array encoded with HM(7,4): (x1, x2, x3, x4, x5, x6, x7)
		if the element is equal to 1, it becomes -1
		if the element is equal to 0, it becomes 1
	'''

	x[x==1] = -1
	x[x==0] = 1
	return x


def noise(y, std_deviation):
	'''
		y is a 7 item array that has been extended in range to [-1, 1]: (y1, y2, y3, y4, y5, y6, y7)
	'''

	# add noise from gaussian normal curve with defined standard deviation
	for i in range(0, len(y)):
		y[i] += np.random.normal(0, std_deviation)

	return y
