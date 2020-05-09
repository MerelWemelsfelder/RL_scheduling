import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
import numpy as np
from numpy import exp, array, random, dot
import random
import pickle
import math

# create NeuralNetwork class
# class NeuralNetwork:
#     def __init__(self, NN_weights):
#         self.weights = NN_weights

#     #activation function ==> S(x) = 1/1+e^(-x)
#     def sigmoid(self, x, deriv=False):
#         if deriv == True:
#             return x * (1 - x)
#         return 1 / (1 + np.exp(-x))

#     # going backwards through the network to update weights
#     def backpropagation(self, score, inputs, predictions):
#         delta = score * self.sigmoid(predictions, deriv=True)
#         for i in range(len(inputs)):
#             self.weights += np.dot(inputs[i].T, delta[i])

#     # function to predict output on new and unseen input data                               
#     def predict(self, new_input):
#         prediction = self.sigmoid(np.dot(new_input, self.weights))
#         return prediction

class Layer(object):
    
    def __init__(self):
        self.training_phase = True
        self.output = 0.0
        
    def forward(self, x_input):
        self.output = x_input
        return self.output
    
    def backward(self, x_input, grad_output):
        return grad_output
    
    def get_params(self):
        return []
    
    def get_params_gradients(self):
        return []

class Dense(Layer):
    
    def __init__(self, W, grad_W, b, grad_b):
        # super(Dense, self).__init__()
        #Randomly initializing the weights from normal distribution
        self.W = W                              # np.random.normal(scale=0.01, size=(n_input, n_output))
        self.grad_W = grad_W
        #initializing the bias with zero
        self.b = b                              # np.zeros(n_output)
        self.grad_b = grad_b

    def forward(self, x_input):
        self.output = np.dot(x_input, self.W) + self.b
        return self.output
    
    def backward(self, x_input, grad_output):
        # get gradients of weights
        self.grad_W = np.dot(np.array(x_input).reshape(self.W.shape[0], 1), np.array(grad_output).reshape(1,len(grad_output)))
        self.grad_b = np.dot(np.ones(len(grad_output)), grad_output)
        # propagate the gradient backwards
        return np.dot(grad_output, np.transpose(self.W))
    
    def get_params(self):
        return [self.W, self.b]

    def get_params_gradients(self):
        return [self.grad_W, self.grad_b]

class ReLU(Layer):

    def forward(self, x_input):
        self.output = x_input.copy()
        self.output[self.output<0] = 0
        return self.output
    
    def backward(self, x_input, grad_output):
        HX = x_input.copy()
        HX[HX<0] = 0
        HX[HX>0] = 1
        
        grad_input = HX * grad_output
        return grad_input

class Sigmoid(Layer):

    def sigmoid_forward(self, x_input):
        output = 1 / (1 + np.exp(-x_input))
        return output

    def sigmoid_grad_input(self, x_input, grad_output):
        f = 1 / (1 + np.exp(-x_input))
        fh = f*(1-f)
        grad_input = fh * grad_output
        return grad_input
        
    def forward(self, x_input):
        self.output = self.sigmoid_forward(x_input)
        return self.output
    
    def backward(self, x_input, grad_output):
        return self.sigmoid_grad_input(x_input, grad_output)

class NeuralNetwork(object):

    def __init__(self, *layers):
        self.layers = layers
        self.training_phase = True

    def set_training_phase(self, is_training=True):
        self.training_phase = is_training
        for layer in self.layers:
            layer.training_phase = is_training
        
    def forward(self, x_input):
        self.output = x_input
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output
    
    def backward(self, x_input, grad_output):
        inputs = [x_input] + [l.output for l in self.layers[:-1]]
        for input_, layer_ in zip(inputs[::-1], self.layers[::-1]):
            grad_output = layer_.backward(input_, grad_output)
            
    def get_params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.get_params())
        return params
    
    def get_params_gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.get_params_gradients())
        return grads

class NLL(object):

    def nll_forward(self, target_pred, target_true):
        target_pred *= 0.99999
        target_true *= 0.99999

        part1 = np.dot(np.transpose(target_true),np.log(target_pred))
        part2 = np.dot(np.transpose(1 - target_true), np.log(1 - target_pred))

        whole = part1 + part2
        N = len(target_pred)
        output = -((1*whole)/N)

        return output[0][0]

    def nll_grad_input(self, target_pred, target_true):
        numerator = target_pred - target_true
        denominator = target_pred * (1 - target_pred)

        grad_input = (1/len(target_pred)) * (numerator/denominator)
        
        return grad_input
    
    def forward(self, target_pred, target_true):
        self.output = self.nll_forward(target_pred, target_true)
        return self.output
  
    def backward(self, target_pred, target_true):
        return self.nll_grad_input(target_pred, target_true)

def l2_regularizer(weight_decay, weights):
    print(weights)
    w = np.sum(weights*weights)
    output = (w*weight_decay)/2   
    return output

class SGD(object):

    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def update_params(self):
        weights = self.model.get_params()
        grads = self.model.get_params_gradients()
        for w, dw in zip(weights, grads):
            update = self.lr * (dw + self.weight_decay * w)
            # it writes the result to the previous variable instead of copying
            np.subtract(w, update, out=w)

# model = RL.NN, X_train = RL.NN_inputs
def update_NN(model, X_train, y_pred, weight_decay, lr, loss, r, r_best):
    
    # y_pred = model.forward(X_train)
    score = (r_best-r)/min(r_best, r)
    y_true = y_pred + (score * y_pred)

    loss_value = loss.forward(y_pred, y_true)  #+ l2_regularizer(weight_decay, model.get_params())
    loss_grad = loss.backward(y_pred, y_true)

    for i in range(len(X_train)):
        model.backward(X_train[i], loss_grad[i])

    sgd = SGD(model, lr=lr, weight_decay=weight_decay)
    sgd.update_params()
    return sgd.model



# Generate the input vector for value prediction by the Neural Network
def generate_NN_input(N, M, LV, GV, ws, resource, jobs, v, i, j, o, z, heur_job, heur_res, heur_order):

    # if the chosen action is the idle action "do_nothing"
    if j == N:
        due_dates = [0 for v in range(M)]
        idle_action = 1
        proctime_job = 0
        proctime_resource = 0
        T_expected = 0
        relative_time_to_duedate = 0
        blocking = 0
    # if the chosen action is a waiting job
    else:
        # how long does the job take to process on this resource
        processing_time = heur_job[v][i][j]
        # due dates of job j on all work stations
        due_dates = jobs[j].D
        idle_action = 0

        # how many standard deviations is the processing time of job j on
        # resource i from the average processing time of all coming jobs on resource i
        proctimes_jobs = []
        for job in ws.jobs_to_come:
            proctimes_jobs.append(heur_job[v][i][job.j])
        proctime_job = processing_time-np.mean(proctimes_jobs)
        if np.std(proctimes_jobs) != 0:
            proctime_job /= np.std(proctimes_jobs)

        # how many standard deviations is the processing time of job j on
        # resource i from the average processing time of job j on all resources,
        # taking into account the time that other resources will be unavailable
        # as a result of processing other jobs, and the blocking on the other
        # resources as a result of still processing another job
        first_units = [resource.units[0] for resource in ws.resources]
        times = []
        for i in range(len(ws.resources)):
            unit = first_units[i]
            if unit.processing == None:
                if ws.resources[i].units[1].processing == None:
                    times.append(heur_job[v][i][j])
                else:
                    times.append(heur_order[v][i][j][ws.resources[i].units[1].processing.j] + heur_job[v][i][j])
            else:
                # time still processing other job + blocking after starting job j + processing time job j
                times.append((unit.c_idle - z) + heur_order[v][i][j][unit.processing.j] + heur_job[v][i][j])
        proctime_resource = processing_time - np.mean(times)
        if np.std(times) != 0:
            proctime_resource /= np.std(times)

        # expected tardiness of job if processing starts now
        T_expected = (z + processing_time) - due_dates[v]
        # time until due date, relative to timespan from z=0 to due date
        relative_time_to_duedate = (due_dates[-1] - z) / due_dates[-1]


        # blocking time due to the order of scheduled jobs on resource i
        if o == None:
            blocking = 0
        else:
            blocking = heur_order[v][i][j][o.j]

    # (v+1)/M, proctimes_jobs, idle_action
    inputs = [T_expected, relative_time_to_duedate, proctime_resource, blocking]

    return inputs