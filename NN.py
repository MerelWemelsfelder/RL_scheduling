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
        target_pred += 0.00001
        target_true += 0.00001
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
    score = (r_best-r)
    if min(r_best, r) > 0:
        score /= min(r_best, r)
    y_true = y_pred + (score * y_pred)

    loss_value = loss.forward(y_pred, y_true)  #+ l2_regularizer(weight_decay, model.get_params())
    loss_grad = loss.backward(y_pred, y_true)

    for i in range(len(X_train)):
        model.backward(X_train[i], loss_grad[i])

    sgd = SGD(model, lr=lr, weight_decay=weight_decay)
    sgd.update_params()
    return sgd.model



# Generate the input vector for value prediction by the Neural Network
def generate_NN_input(N, M, LV, GV, CONFIG, ws, resource, jobs, v, i, j, z, heur_job, heur_res, heur_order, deltas):

    # if the chosen action is the idle action "do_nothing"
    # if j == N:
    #     time_res_minmax = 0
    #     time_res_stdev = 0
    #     time_res_stdev_occupied = 0
    #     time_res_stdev_blocking = 0
    #     time_res_stdev_occ_block = 0
    #     time_job_minmax = 0
    #     time_job_minmax_coming = 0
    #     time_job_stdev_all = 0
    #     time_job_stdev_coming = 0
    #     time_job_stdev_blocking_all = 0
    #     time_job_stdev_blocking_coming = 0
    #     blocking_res_stdev = 0
    #     blocking_job_all_stdev = 0
    #     blocking_job_coming_stdev = 0
    #     blocking = 0
    #     future_blockings_mean = 0
    #     future_blockings_median = 0
    #     u0_occupied = 0
    #     T_expected = 0
    #     time_to_duedate = 0
    #     relative_duedate = 0
    #     idle_action = 1

    # else:
    ### PROCESSING TIME ###
    processing_time = heur_res[j][v][i]
    u1_proc = resource.units[1].processing
    
    # PROC TIME OF I ON J, RELATIVE TO J ON ALL RESOURCES
    
    # on a scale from min to max processing time of j on all resources
    times_res = list(heur_res[j][v].values())
    time_res_minmax = (processing_time - min(times_res)) / (max(times_res) - min(times_res))

    # measuring how many standard deviations it is from the mean
    time_res_stdev = (processing_time - np.mean(times_res)) / np.std(times_res)

    # measuring how many standard deviations it is from the mean
    # taking the time into account that other resources will still be unavailable
    # taking the blocking on other resources into account
    # blocking, relative to job j on other resources
    times_res_occupied = []
    times_res_blocking = []
    times_res_occ_block = []
    blocking_res = []
    for res in ws.resources:
        u0 = res.units[0]
        u1 = res.units[1]
        if u0.processing != None:
            blocking_0 = heur_order[v][res.i][j][u0.processing.j]
            times_res_occupied.append(heur_res[j][v][res.i] + (u0.c_idle - z))
            times_res_blocking.append(heur_res[j][v][res.i] + blocking_0)
            times_res_occ_block.append(heur_res[j][v][res.i] + (u0.c_idle - z) + blocking_0)
            blocking_res.append((u0.c_idle - z) + blocking_0)
        elif u1.processing != None:
            blocking_1 = heur_order[v][res.i][j][u1.processing.j]
            times_res_occupied.append(heur_res[j][v][res.i])
            times_res_blocking.append(heur_res[j][v][res.i] + blocking_1)
            times_res_occ_block.append(heur_res[j][v][res.i] + blocking_1)
            blocking_res.append(blocking_1)
        else:
            times_res_occupied.append(heur_res[j][v][res.i])
            times_res_blocking.append(heur_res[j][v][res.i])
            times_res_occ_block.append(heur_res[j][v][res.i])
            blocking_res.append(0)
    time_res_stdev_occupied = 0
    if np.std(times_res_occupied) > 0:
        time_res_stdev_occupied = (processing_time - np.mean(times_res_occupied)) / np.std(times_res_occupied)


    blocking_res_stdev = 0
    time_res_stdev_blocking = 0
    time_res_stdev_occ_block = 0
    if u1_proc != None:
        if np.std(times_res_blocking) > 0:
            time_res_stdev_blocking = ((processing_time + heur_order[v][i][j][u1_proc.j]) - np.mean(times_res_blocking)) / np.std(times_res_blocking)
        if np.std(times_res_occ_block) > 0:            
            time_res_stdev_occ_block = ((processing_time + heur_order[v][i][j][u1_proc.j]) - np.mean(times_res_occ_block)) / np.std(times_res_occ_block)
        if np.std(blocking_res) > 0:
            blocking_res_stdev = (heur_order[v][i][j][u1_proc.j] - np.mean(blocking_res)) / np.std(blocking_res)
    else:
        if np.std(times_res_blocking) > 0:
            time_res_stdev_blocking = (processing_time - np.mean(times_res_blocking)) / np.std(times_res_blocking)
        if np.std(times_res_occ_block) > 0:
            time_res_stdev_occ_block = (processing_time - np.mean(times_res_occ_block)) / np.std(times_res_occ_block)
        if np.std(blocking_res) > 0:
            blocking_res_stdev = (0 - np.mean(blocking_res)) / np.std(blocking_res)

    # PROC TIME OF I ON J, RELATIVE TO ALL JOBS ON I
    j_coming = [job.j for job in ws.jobs_to_come]
    times_job_all = list(heur_job[v][i].values())
    times_job_coming = [heur_job[v][i][j] for j in j_coming]

    # on a scale from min to max processing time of all jobs on i
    time_job_minmax = (processing_time - min(times_job_all)) / (max(times_job_all) - min(times_job_all))
    time_job_minmax_coming = 0
    if max(times_job_coming) != min(times_job_coming):
        time_job_minmax_coming = (processing_time - min(times_job_coming)) / (max(times_job_coming) - min(times_job_coming))

    # measuring how many standard deviations it is from the mean
    time_job_stdev_all = (processing_time - np.mean(times_job_all)) / np.std(times_job_all)
    time_job_stdev_coming = 0
    if np.std(times_job_coming) > 0:
        time_job_stdev_coming = (processing_time - np.mean(times_job_coming)) / np.std(times_job_coming)

    # measuring how many standard deviations it is from the mean
    # taking the blocking on other resources into account
    # blocking, relative to other jobs on same resource
    times_job_blocking_all = []
    times_job_blocking_coming = []
    blocking_job_all = []
    blocking_job_coming = []

    time_job_stdev_blocking_all = 0
    time_job_stdev_blocking_coming = 0
    blocking_job_all_stdev = 0
    blocking_job_coming_stdev = 0
    if u1_proc != None:
        other = jobs.copy()
        other.remove(u1_proc)
        for job in other:
            blocking_j = heur_order[v][i][job.j][u1_proc.j]
            times_job_blocking_all.append(heur_job[v][i][job.j] + blocking_j)
            blocking_job_all.append(blocking_j)
            if job.j in j_coming:
                times_job_blocking_coming.append(heur_job[v][i][job.j] + heur_order[v][i][job.j][u1_proc.j])
                blocking_job_coming.append(blocking_j)

        if np.std(times_job_blocking_all) > 0:
            time_job_stdev_blocking_all = ((processing_time + heur_order[v][i][j][u1_proc.j]) - np.mean(times_job_blocking_all)) / np.std(times_job_blocking_all)
        if np.std(times_job_blocking_coming) > 0:
            time_job_stdev_blocking_coming = ((processing_time + heur_order[v][i][j][u1_proc.j]) - np.mean(times_job_blocking_coming)) / np.std(times_job_blocking_coming) 
        if np.std(blocking_job_all) > 0:
            blocking_job_all_stdev = (heur_order[v][i][j][u1_proc.j] - np.mean(blocking_job_all)) / np.std(blocking_job_all)
        if np.std(blocking_job_coming) > 0:
            blocking_job_coming_stdev = (heur_order[v][i][j][u1_proc.j] - np.mean(blocking_job_coming)) / np.std(blocking_job_coming)
    else:
        times_job_blocking = times_job_all.copy()
        times_job_blocking_coming = times_job_coming.copy()

        if np.std(times_job_blocking_all) > 0:
            time_job_stdev_blocking_all = (processing_time - np.mean(times_job_blocking_all)) / np.std(times_job_blocking_all)
        if np.std(times_job_blocking_coming) > 0:
            (processing_time - np.mean(times_job_blocking_coming)) / np.std(times_job_blocking_coming)
        

    # absolute blocking time due to the order of scheduled jobs on resource i
    blocking = 0
    if u1_proc != None:
        blocking = heur_order[v][i][j][u1_proc.j]

    # the time that unit 0 of resource i will be occupied as a result from scheduling j on i
    u0_occupied = deltas[v][j][0][i]

    # mean and median blocking in the future caused by start processing job j
    other = [job.j for job in jobs].copy()
    other.remove(j)
    future_blockings = [heur_order[v][i][o][j] for o in other]
    future_blockings_mean = np.mean(future_blockings)
    future_blockings_median = np.median(future_blockings)

    # how many jobs are already being processed on resource i
    already_processing = 0
    for q in range(GV[v]):
        if resource.units[q].processing != None:
            already_processing += 1

    # TARDINESS

    # due dates of job j on all work stations
    duedate = jobs[j].D[-1]
    duedates_coming = [jobs[j].D[-1] for j in j_coming]
        
    # expected tardiness of job if processing starts now
    T_expected = (z + processing_time) - duedate
    # time until due date, relative to timespan from z=0 to due date
    time_to_duedate = (duedate - z) / duedate
    # number of standard deviations of due date from due dates of coming jobs
    relative_duedate = 0
    if np.std(duedates_coming) > 0:
        relative_duedate = (duedate - np.mean(duedates_coming)) / np.std(duedates_coming)

    # lengths = 23 15 9 4 5 1 9 12 6
    all_vars = [time_res_minmax, time_res_stdev, time_res_stdev_occupied, time_res_stdev_blocking, time_res_stdev_occ_block, time_job_minmax, time_job_minmax_coming, time_job_stdev_all, time_job_stdev_coming, time_job_stdev_blocking_all, time_job_stdev_blocking_coming, blocking_res_stdev, blocking_job_all_stdev, blocking_job_coming_stdev, blocking, u0_occupied, future_blockings_mean, future_blockings_median, already_processing, T_expected, time_to_duedate, relative_duedate]
    XV = [time_res_minmax, time_res_stdev, time_res_stdev_occ_block, time_job_minmax, time_job_stdev_coming, time_job_stdev_blocking_coming, blocking_res_stdev, blocking_job_coming_stdev, blocking, u0_occupied, future_blockings_median, already_processing, T_expected, time_to_duedate, relative_duedate]
    minmax_large = [time_res_stdev, time_job_minmax, time_job_stdev_coming, blocking_res_stdev, blocking, u0_occupied, future_blockings_median, T_expected, time_to_duedate]
    minmax_small = [blocking, future_blockings_mean, future_blockings_median, T_expected]
    generalizability = [time_res_minmax, time_job_stdev_coming, time_job_stdev_blocking_coming, blocking_job_coming_stdev, relative_duedate]
    high = [T_expected]
    absolute = [time_res_minmax, time_job_minmax, time_job_minmax_coming, blocking, u0_occupied, future_blockings_median, already_processing, T_expected, time_to_duedate]
    relative = [time_res_stdev, time_res_stdev_occupied, time_res_stdev_blocking, time_res_stdev_occ_block, time_job_stdev_all, time_job_stdev_coming, time_job_stdev_blocking_all, time_job_stdev_blocking_coming, blocking_res_stdev, blocking_job_all_stdev, blocking_job_coming_stdev, relative_duedate]
    generalizability_T = [time_res_minmax, time_job_stdev_coming, time_job_stdev_blocking_coming, blocking_job_coming_stdev, relative_duedate, T_expected]

    configs = [all_vars, XV, minmax_large, minmax_small, generalizability, high, absolute, relative, generalizability_T]

    # for i in inputs:
    #     if math.isnan(i):
    #         print(inputs.index(i))

    return configs[CONFIG]