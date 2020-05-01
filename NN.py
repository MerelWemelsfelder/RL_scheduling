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
def generate_NN_input(v, i, j, due_dates, o, z, heur_job, heur_res, heur_order, N, M, LV, GV):
    # if the chosen action is the idle action "do_nothing"
    if j == N:
        idle_action = 1
        proctime_job = 0
        proctime_resource = 0
        T_expected = 0
        relative_time_to_duedate = 0
        blocking = 0

    # if the chosen action is a waiting job
    else:
        idle_action = 0

        processing_time = heur_job[v][i][j]
        # how many standard deviations is the processing time of job j on
        # resource i from the average processing time of all jobs on resource i
        proctime_job = (processing_time-np.mean(list(heur_job[v][i].values())))/np.std(list(heur_job[v][i].values()))
        # how many standard deviations is the processing time of job j on
        # resource i from the average processing time of all jobs on resource i
        proctime_resource = (processing_time-np.mean(list(heur_res[j][v].values())))/np.std(list(heur_res[j][v].values()))

        T_expected = (z + processing_time) - due_dates[-1]
        relative_time_to_duedate = (due_dates[-1] - z) / due_dates[-1]

        # blocking time due to the order of scheduled jobs on resource i
        if o == None:
            blocking = 0
        else:
            blocking = heur_order[v][i][j][o.j]

    return [N, M, LV[v], GV[v], T_expected, relative_time_to_duedate, proctime_job, proctime_resource, blocking, idle_action]