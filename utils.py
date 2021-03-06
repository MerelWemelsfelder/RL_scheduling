import numpy as np
import random
import pickle
import _pickle as cpickle
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from numpy import save
from numpy import load
from time import gmtime, strftime

from NN import *

# Make a pretty plot of the found schedule and save it as a png
def plot_schedule(OUTPUT_DIR, config_name, schedule, N, M, LV, GV, EPSILON, TIMEOUT):
    lv = max(LV)
    gv = max(GV)

    fig, gnt = plt.subplots() 
    gnt.set_ylim(0, lv*gv*10)
    gnt.set_xlim(0, max(schedule.c))
    gnt.set_xlabel('time') 
    gnt.set_ylabel('resources ri, units uq')
      
    gnt.set_yticks(list(range(5,lv*gv*10,10)))
    y_labels = []
    for i in range(lv):
        for q in range(gv):
            y_labels.append("r"+str(i)+", u"+str(q))
    gnt.set_yticklabels(y_labels) 

    legend_colors = list(range(N))
    legend_names = ["job "+str(j) for j in range(N)]

    for v in range(M):
        for i in range(LV[v]):
            for j in schedule.schedule[v][i]:
                color = (random.random(), random.random(), random.random())
                legend_colors[j] = Line2D([0], [0], color=color, lw=4)
                for q in range(GV[v]):
                    start = schedule.t_q[v][j][q]
                    duration = schedule.c_q[v][j][q] - schedule.t_q[v][j][q]
                    y_position = (10*i*GV[v])+(10*q)
                    gnt.broken_barh([(start, duration)], (y_position, 9), facecolors = color)

    gnt.legend(legend_colors, legend_names)
    plt.savefig(OUTPUT_DIR+"schedules/"+config_name+"/E"+str(EPSILON)+"/"+str(TIMEOUT)+"_"+str(N)+"-"+str(LV[0])+".png")
    plt.close(fig)

# Store statistics of some test iteration to log file
def write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, config_name, GAMMA, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch, MILP_objval, MILP_calctime, MILP_TIMEOUT, MCTS_calctime, MCTS_objval):
    file = open(OUTPUT_DIR+"batch/"+config_name+"/log.csv",'a')
    file.write("\n"+METHOD+"\t"+PHASE+"\t"+str(N)+"\t"+str(M)+"\t"+str(LV[0])+"\t"+str(GV[0])+"\t"+config_name+"\t"+str(EPOCHS)+"\t"+str(GAMMA)+"\t"+str(round(EPSILON,2))+"\t"+str(EPSILON_DECREASE)+"\t"+str(layer_dims)+"\t"+str(weight_decay)+"\t"+str(OBJ_FUN["Cmax"])+"\t"+str(OBJ_FUN["Tsum"])+"\t"+str(makespan)+"\t"+str(Tsum)+"\t"+str(Tmax)+"\t"+str(Tn)+"\t"+str(calc_time)+"\t"+str(epoch)+"\t"+str(MILP_objval)+"\t"+str(MILP_calctime)+"\t"+str(MILP_TIMEOUT)+"\t"+str(MCTS_calctime)+"\t"+str(MCTS_objval))
    file.close()

# Store statistics of some test iteration to log file
def write_training_batch(OUTPUT_DIR, X_train, y_true, INPUT_CONFIGS, CONFIG):

    folder = INPUT_CONFIGS[CONFIG] + "/"
    file_X = open(OUTPUT_DIR+"batch/"+folder+"X.txt",'a')
    file_y = open(OUTPUT_DIR+"batch/"+folder+"y.txt",'a')

    for i in range(len(X_train)):
        file_X.write(";".join([str(x) for x in X_train[i]])+"\n")
        file_y.write(str(y_true)+"\n")

    file_X.close()
    file_y.close()

    with open(OUTPUT_DIR+"batch/"+folder+"X.txt") as f:
        X = f.readlines()
        X = [x.strip().split(";") for x in X]
        for f in range(0,len(X)):
            X[f] = [float(x) for x in X[f]]
    with open(OUTPUT_DIR+"batch/"+folder+"y.txt") as f:
        y = f.readlines()
        y = [float(y.strip("\n")) for y in y]

    if (len(X)!=len(y)):
        print(y)
        print(X_train)
        print(y_true)
        print(len(X), len(y))    

# Store the trained weights of the Neural Network, used as a policy value function
def write_NN_weights(OUTPUT_DIR, layer_dims, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, INPUT_CONFIGS, CONFIG):
    folder = INPUT_CONFIGS[CONFIG] + "/"
    with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-weights.pickle",'wb') as f:
        pickle.dump(NN_weights, f)
    with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-biases.pickle",'wb') as f:
        pickle.dump(NN_biases, f)
    with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-weights_grad.pickle",'wb') as f:
        pickle.dump(NN_weights_gradients, f)
    with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-biases_grad.pickle",'wb') as f:
        pickle.dump(NN_biases_gradients, f)

def find_layer_dims(CONFIG):
    if CONFIG  == 0:
        return [32, 25, 16, 7, 1]
    if CONFIG  == 1:
        return [19, 13, 6, 1]
    if CONFIG  == 2:
        return [12, 8, 4, 1]
    if CONFIG  == 3:
        return [6, 4, 2, 1]
        # return [6, 10, 8, 3, 1]
    if CONFIG  == 4:
        return [11, 7, 3, 1]
    if CONFIG  == 5:
        return [15, 11, 6, 1]
    if CONFIG  == 6:
        return [18, 13, 6, 1]
    if CONFIG  == 7:
        return [10, 7, 3, 1]
    if CONFIG  == 8:
        return [6, 4, 2, 1]

# Heuristic: for each resource in each work station, the time that each job costs to process
def heuristic_best_job(deltas, N, M, LV, GV):
    heur_job = dict()

    for v in range(M):
        dict_v = dict()
        for i in range(LV[v]):
            dict_i = dict()
            for j in range(N):
                j_total = 0
                for q in range(GV[v]):
                    j_total += deltas[v][j][q][i]
                dict_i[j] = j_total
            dict_v[i] = dict_i
        heur_job[v] = dict_v

    return heur_job

# Heuristic: for each job, the time it costs for each resource if processed on it
def heuristic_best_resource(heur_j, N, M, LV):
    heur_r = dict()
    for j in range(N):
        dict_j = dict()
        for v in range(M):
            dict_v = dict()
            for i in range(LV[v]):
                dict_v[i] = heur_j[v][i][j]
            dict_j[v] = dict_v
        heur_r[j] = dict_j
    return heur_r

# Heuristic: For each resource, the blocking that occurs as a result
#            of executing job j after having executed job o
def heuristic_blocking(deltas, N, M, LV, GV):
    all_jobs = list(range(N))
    heur_blocking = dict()             # key = resource i
    for v in range(M):
        dict_v = dict()
        for i in range(LV[v]):
            dict_i = dict()             # key = job j
            for j in range(N):
                dict_j = dict()         # key = job o
                other = all_jobs.copy()
                other.remove(j)

                for o in other:
                    counter = 0
                    spare = 0
                    for q in range(GV[v]-1):
                        dj = deltas[v][j][q+1][i]
                        do = deltas[v][o][q][i]
                        blocking = dj-do
                        if blocking < 0:
                            spare += blocking
                        if blocking > 0:
                            if spare >= blocking:
                                spare -= blocking
                            else:
                                blocking -= spare
                                counter += blocking
                    dict_j[o] = counter

                dict_i[j] = dict_j
            dict_v[i] = dict_i
        heur_blocking[v] = dict_v

    return heur_blocking

def heuristic_reverse_blocking(deltas, N, M, LV, GV):
    all_jobs = list(range(N))
    heur_rev_blocking = dict()          # key = resource i
    for v in range(M):
        dict_v = dict()
        for i in range(LV[v]):
            dict_i = dict()             # key = job j
            for j in range(N):
                dict_j = dict()         # key = job o
                other = all_jobs.copy()
                other.remove(j)

                for o in other:
                    j_count = 0
                    o_count = 0
                    counter = 0
                    delay = 0
                    for q in range(GV[v]-1):
                        dj = deltas[v][j][q+1][i]
                        do = deltas[v][o][q][i]
                        rev_block = (o_count + delay + do) - (j_count + dj)
                        j_count += dj
                        o_count += do

                        if rev_block >= 0:
                            counter += rev_block
                            delay = 0
                        if rev_block < 0:
                            delay = rev_block
                    dict_j[o] = counter

                dict_i[j] = dict_j
            dict_v[i] = dict_i
        heur_rev_blocking[v] = dict_v

    return heur_rev_blocking

# Execute the NN policy value function with stored weights
# to initialize policy values to be used by JEPS
def load_NN_into_JEPS(NN_weights, NN_biases, policies, N, M, LV, GV, due_dates, heur_job, heur_res, heur_blocking, heur_rev_blocking):
    policy_function = NN(Dense(10, 4, NN_weights[0], NN_biases[0]), ReLU(),Dense(4, 1, NN_weights[1], NN_biases[1]),Sigmoid())

    for v in range(M):
        for i in range(LV[v]):
            for j in range(N):
                due_dates_j = [d[j] for d in due_dates]

                inputs = generate_NN_input(v, i, j, due_dates_j, None, 0, heur_job, heur_res, heur_blocking, heur_rev_blocking, N, M, LV, GV)
                policies[v][i][j] = policy_function.forward(inputs)
            # inputs = generate_NN_input(v, i, N, [0 for v in range(M)], None, 0, heur_job, heur_res, heur_blocking, heur_rev_blocking, N, M, LV, GV)
            # policies[v][i][N] = policy_function.forward(inputs)

    return policies