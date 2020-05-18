import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

from NN import *

# Print found schedule in the terminal
def print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime):
    print("MILP solution")

    print("processing time: "+str(MILP_calctime))
    print("objective value: "+str(MILP_objval))
    print("schedule "+str(MILP_schedule))

    print("\nRL solution")

    print("processing time: "+str(calc_time))
    print("makespan: "+str(schedule.Cmax))
    print("Tmax: "+str(schedule.Tmax))
    print("Tmean: "+str(schedule.Tmean))
    print("Tn: "+str(schedule.Tn))

    print("\nschedule: ")
    for s in schedule.schedule:
        print(s)
    print("starting times: "+str(schedule.t))
    print("completion times: "+str(schedule.c))

# Make a pretty plot of the found schedule and save it as a png
def plot_schedule(OUTPUT_DIR, schedule, N, M, LV, GV):
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
    plt.savefig(OUTPUT_DIR+"schedules/"+str(M)+"-"+str(N)+"-"+str(LV)+"-"+str(GV)+".png")
    plt.close(fig)

# Store statistics of some test iteration to log file
def write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, CONFIG, GAMMA, EPSILON, layer_dims, weight_decay, lr, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch, MILP_objval, MILP_calctime):
    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("\n"+METHOD+"\t"+PHASE+"\t"+str(N)+"\t"+str(M)+"\t"+str(LV[0])+"\t"+str(GV[0])+"\t"+CONFIG+"\t"+str(EPOCHS)+"\t"+str(GAMMA)+"\t"+str(round(EPSILON,2))+"\t"+str(layer_dims)+"\t"+str(weight_decay)+"\t"+str(lr)+"\t"+str(OBJ_FUN["Cmax"])+"\t"+str(OBJ_FUN["Tsum"])+"\t"+str(makespan)+"\t"+str(Tsum)+"\t"+str(Tmax)+"\t"+str(Tn)+"\t"+str(calc_time)+"\t"+str(epoch)+"\t"+str(MILP_objval)+"\t"+str(MILP_calctime))
    file.close()

# Store the trained weights of the Neural Network, used as a policy value function
def write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, OBJ_FUN, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients):
    with open(OUTPUT_DIR+"NN_weights/"+str(layer_dims)+"-"+str(N)+"_"+str(LV)+"-"+str(OBJ_FUN["Cmax"])+"_"+str(OBJ_FUN["Tsum"])+"-"+"weights.pickle",'wb') as f:
        pickle.dump(NN_weights, f)
    with open(OUTPUT_DIR+"NN_weights/"+str(layer_dims)+"-"+str(N)+"_"+str(LV)+"-"+str(OBJ_FUN["Cmax"])+"_"+str(OBJ_FUN["Tsum"])+"-"+"biases.pickle",'wb') as f:
        pickle.dump(NN_biases, f)
    with open(OUTPUT_DIR+"NN_weights/"+str(layer_dims)+"-"+str(N)+"_"+str(LV)+"-"+str(OBJ_FUN["Cmax"])+"_"+str(OBJ_FUN["Tsum"])+"-"+"weights_grad.pickle",'wb') as f:
        pickle.dump(NN_weights_gradients, f)
    with open(OUTPUT_DIR+"NN_weights/"+str(layer_dims)+"-"+str(N)+"_"+str(LV)+"-"+str(OBJ_FUN["Cmax"])+"_"+str(OBJ_FUN["Tsum"])+"-"+"biases_grad.pickle",'wb') as f:
        pickle.dump(NN_biases_gradients, f)

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
def heuristic_order(deltas, N, M, LV, GV):
    all_jobs = list(range(N))
    heur_order = dict()             # key = resource i
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
        heur_order[v] = dict_v

    return heur_order

# Execute the NN policy value function with stored weights
# to initialize policy values to be used by JEPS
def load_NN_into_JEPS(NN_weights, NN_biases, policies, N, M, LV, GV, due_dates, heur_job, heur_res, heur_order):
    policy_function = NN(Dense(10, 4, NN_weights[0], NN_biases[0]), ReLU(),Dense(4, 1, NN_weights[1], NN_biases[1]),Sigmoid())

    for v in range(M):
        for i in range(LV[v]):
            for j in range(N):
                due_dates_j = [d[j] for d in due_dates]

                inputs = generate_NN_input(v, i, j, due_dates_j, None, 0, heur_job, heur_res, heur_order, N, M, LV, GV)
                policies[v][i][j] = policy_function.forward(inputs)
            # inputs = generate_NN_input(v, i, N, [0 for v in range(M)], None, 0, heur_job, heur_res, heur_order, N, M, LV, GV)
            # policies[v][i][N] = policy_function.forward(inputs)

    return policies