import numpy as np
import pandas as pd

class NN:

	def __init__(self , n_inputs):
		self.label_names = ['უ', 'ყ', 'მ', 'შ', 'ძ', 'წ', 'ს', 'ხ', 'ლ', 'ჩ']
		# learning info
		self.n_iterations = 1
		# layer info
		self.l_sizes = [n_inputs, 2, 1]
		self.n_layer = len(self.l_sizes)
		# generating biases and weights on every hidden layer
		self.biases = [np.random.randn(i, 1) for i in self.l_sizes[1:]]
		self.weights = [np.random.randn(j, i) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]
		# learning rate
		self.l_rate = 0.01

	# Activation function
	def sigmoid(self, s):
		return 1.0 / (np.exp(-s) + 1.0)

	# Derivative of activation function
	def sigmoid_der(self, s):
		return self.sigmoid(s) * (1.0 - self.sigmoid(s))

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
	def backward(self, X, y):
		biases_err = [np.zeros((i, 1)) for i in self.l_sizes[1:]]
		weights_err = [np.zeros((j, i)) for i, j in zip(self.l_sizes[:-1], self.l_sizes[1:])]
		
		
		# forward propagation while saving a and z values
		a = [X]
		z = []
		for i in range(len(self.biases)):
			bias = self.biases[i]
			weight = self.weights[i]
			curr = a[-1]
			mult = np.dot(weight , curr)
			z.append(mult + bias)
			curr = self.sigmoid(mult + bias)
			a.append(curr)

		# backpropagation
		loss = (a[-1] - y) * self.sigmoid_der(z[-1])
		weights_err[-1] = np.dot(loss, a[-2].transpose())
		biases_err[-1] = loss
		
		for i in range(2 , self.n_layer):
			loss = np.dot(self.weights[-i + 1].transpose(), loss) * self.sigmoid_der(z[-i])
			weights_err[-i] = np.dot(loss, a[-i - 1].transpose())
			biases_err[-i] = loss

		print (self.weights)
		print(weights_err)
		#update weights and biases
		for i in range(len(self.biases)):
			self.weights[i] -= self.l_rate * weights_err[i]
			self.biases[i] -= self.l_rate * biases_err[i]

	def training(self, data):
		for i in range(self.n_iterations):
			for j in range(len(data)):
				X = data[j][0]
				y = data[j][1]
				self.backward(X , y)
			
				

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

	xor_ex = [(np.array([0, 0]), np.array([0])), (np.array([0, 1]), np.array([1])), (np.array([1, 0]), np.array([1])), (np.array([1, 1]), ([0]))]

	aq = NN(2)
	aq.training(xor_ex)
	print(d.forward([0 , 0]))