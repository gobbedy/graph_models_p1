import numpy as np


G = np.array([1,1,0,1],[1,0,1,1],[1,0,0,0],[0,1,1,1],[0,1,0,0],[0,0,1,0],[0,0,0,1])


def hamming(m):
	'''
		m is a 4 item array (the inital message): (m1, m2, m3, m4)
		x is a 7 item array returned as the hamming encoded message: (x1, x2, x3, x4, x5, x6, x7)
	'''

	# multiply m with standard matrix G for hamming code encryption and take modulo 2
	x = np.remainder(G.dot(m), 2)
	return x


def inversion(x):
	'''
		x is a 7 item array encoded with HM(7,4): (x1, x2, x3, x4, x5, x6, x7)
		y is a 7 item array with extended range from [-1, 1]: (y1, y2, y3, y4, y5, y6, y7)
	'''

	# creating deep copy of x
	y = np.zeros(5)
	y[:] = x

	# if the element is equal to 1, it becomes -1
	# if the element is equal to 0, it becomes 1
	y[y==1] = -1
	y[y==0] = 1
	return y


def noise(y, std_deviation):
	'''
		y is a 7 item array that has been extended in range to [-1, 1]: (y1, y2, y3, y4, y5, y6, y7)
		z is a 7 item array with added noise from gaussian distr: (z1, z2, z3, z4, z5, z6, z7)
	'''

	# add noise from gaussian normal curve with defined standard deviation
	z = np.zeros(len(y))
	for i in range(0, len(y)):
		z[i] = y[i] + np.random.normal(0, std_deviation)
	return z


def transmitter(m, std_deviation):
	'''
		m is a 4 item array (the inital message): (m1, m2, m3, m4)
		z is a 7 item array with added noise from gaussian distr: (z1, z2, z3, z4, z5, z6, z7)
	'''

	x = hamming(m)
	y = inversion(x)
	z = noise(y, std_deviation)
	return z
