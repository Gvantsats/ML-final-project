import numpy as np
import pandas as pd

class NN:

	def __init__(self , n_inputs):
		self.label_names = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ']
		# learning info
		self.n_iterations = 1000
		# layer info
		self.l_sizes = [n_inputs, 4, 4, 1]
		# generating biases and weights on every hidden layer
		self.biases = [np.random.randn(i, 1) for i in self.l_sizes[1:]]
		self.weights = [np.random.randn(i, j) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]

	# Activation function
	def sigmoid(s):
		return 1.0 / (np.exp(-1 * s) + 1.0)

	# Derivative of activation function
	def sigmoid_der(s):
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
		pass

	def training(self):
		for i in range(self.n_iterations):
			curr_y = forward(X)
			backward(X, curr_y, y)

	def classify(self , data):
		best = float("-inf")
		best_char = -1
		for i in range(len(self.label_names)):
			ans = self.forward(data)
			if (ans > best):
				best = ans
				best_char = i
		
		return best_char

	