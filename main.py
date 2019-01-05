import numpy as np
import pandas as pd

""" Handling data """

""" Neural network implementation """
NUM_ITERATIONS = 1000

# Activation function
def sigmoid(s):
    return 1 / (np.exp(-1 * s) + 1)

# Derivative of activation function
def sigmoid_der(s):
    return np.exp(-1 * s / ((1 + np.exp(-1 * s)) ** 2))

# Forward propagation
def forward(X):
	pass

# Backward propagation
def backward(X, curr_y, y):
	pass

def training():
	for i in range(NUM_ITERATIONS):
   	curr_y = forward(X)
   	backward(X, curr_y, y)

""" Getting predictions """