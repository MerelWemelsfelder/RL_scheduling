import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
import numpy as np
from numpy import exp, array, random, dot
import random
import pickle

# create NeuralNetwork class
class NeuralNetwork:
    def __init__(self, NN_weights):
        self.weights = NN_weights

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # going backwards through the network to update weights
    def backpropagation(self, score, inputs, predictions):
        delta = score * self.sigmoid(predictions, deriv=True)
        for i in range(len(inputs)):
            self.weights += np.dot(inputs[i].T, delta[i])

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# Generate the input vector for value prediction by the Neural Network
def generate_NN_input(i, j, o, z, heur_job, heur_res, heur_order, N, LV, GV):
    # if the chosen action is the idle action "do_nothing"
    if j == N:
        idle_action = 1
        calctime_job = 0
        calctime_resource = 0
        blocking = 0

    # if the chosen action is a waiting job
    else:
        idle_action = 0
        # how many standard deviations is the processing time of job j on
        # resource i from the average processing time of all jobs on resource i
        calctime_job = (j-np.mean(list(heur_job[i].values())))/np.std(list(heur_job[i].values()))
        # how many standard deviations is the processing time of job j on
        # resource i from the average processing time of all jobs on resource i
        calctime_resource = (j-np.mean(list(heur_res[j].values())))/np.std(list(heur_res[j].values())) 

        # blocking time due to the order of scheduled jobs on resource i
        if o == None:
            blocking = 0
        else:
            blocking = heur_order[i][j][o.j]
    
    return [N, LV, GV, z, calctime_job, calctime_resource, blocking, idle_action]