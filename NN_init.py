import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
import numpy as np
from numpy import exp, array, random, dot
import random
import pickle

# create NeuralNetwork class
class NeuralNetwork:
    def load(self, weights):
        self.weights = weights
        
    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # data will flow through the neural network.
    def feed_forward(self):
        self.hidden = np.random.rand(len(inputs),LV,N)
        for i in range(len(self.inputs)):
            self.hidden[i] = self.sigmoid(np.dot(self.inputs[i], self.weights))

    # going backwards through the network to update weights
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        for i in range(len(self.inputs)):
            self.weights += np.dot(self.inputs[i].T, delta[i])

    # train the neural net for 25,000 iterations
    def train(self, inputs, outputs, epochs=10000):
        self.inputs  = inputs
        self.outputs = outputs
        self.weights = np.random.rand(N,N)
        self.error_history = []
        self.epoch_list = []
        
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# only uses load, predict and sigmoid
def initial_policies_from(delta):
    with open('NN_weights.pickle','rb') as f:
        weights = pickle.load(f)

    NN = NeuralNetwork()
    NN.load(weights)

    # shape that the NN was trained on
    N = 20
    LV = 15
    GV = 10

    # if len(delta.shape)==3:
    n, gv, lv = delta.shape
    input_padded = np.zeros([LV, N])
    for j in range(n):
        for i in range(lv):
            input_padded[i][j] = sum([q[i] for q in delta[j]])

    policies = np.zeros([lv,n+1])
    output_padded = NN.predict(input_padded)
    for i in range(lv):
        for j in range(n):
            policies[i][j] = output_padded[i][j]

    return policies