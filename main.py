import numpy as np
import pandas as pd

class NN:

	def __init__(self , n_inputs):
		self.label_names = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ']
		# learning info
		self.n_iterations = 1
		# layer info
		self.l_sizes = [n_inputs, 15 , 10 , 10]
		self.n_layer = len(self.l_sizes)
		# generating biases and weights on every hidden layer
		self.biases = [np.random.randn(i, 1) for i in self.l_sizes[1:]]
		self.weights = [np.random.randn(j, i) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]
		# learning rate
		self.l_rate = 0.01

	# Activation function
	def sigmoid(self, s):
		return 1.0 / (np.exp(-1 * s) + 1.0)

	# Derivative of activation function
	def sigmoid_der(self, s):
		return sigmoid(s) * (1.0 - sigmoid(s))

	# Forward propagation
	def forward(self, data):
		curr = data
		for i in range(len(self.biases)):
			bias = self.biases[i]
			weight = self.weights[i]
			mult = np.dot(weight , curr)
			curr = self.sigmoid(mult + bias)

		return curr

	# Backward propagation
	def backward(self, X, curr_y, y):
		biases_err = [np.zeros(i, 1) for i in self.l_sizes[1:]]
		weights_err = [np.zeros(j, i) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]
		activation = [None] * self.n_layer


	def training(self, data):
		for i in range(self.n_iterations):
			
				

	def classify(self , data):
		best = float("-inf")
		best_char = -1
		for i in range(len(self.label_names)):
			ans = self.forward(data)
			if (ans > best):
				best = ans
				best_char = i
		
		return best_char

	
if __name__ == "__main__":

	xor_ex = [[[0, 0], 0], [[0, 1], 1], [[1, 0], 1], [[1, 1], 0]]

	aq = NN(3)
	d = np.array([2,3,1])
	d = d.reshape((3 , 1))
	print(d.shape)
	print (aq.training(d , [1]))