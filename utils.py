import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

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
def plot_schedule(OUTPUT_DIR, schedule, N, LV, GV):

    fig, gnt = plt.subplots() 
    gnt.set_ylim(0, LV*GV*10)
    gnt.set_xlim(0, max(schedule.c))
    gnt.set_xlabel('time') 
    gnt.set_ylabel('resources ri, units uq')
      
    gnt.set_yticks(list(range(5,LV*GV*10,10)))
    y_labels = []
    for i in range(LV):
        for q in range(GV):
            y_labels.append("r"+str(i)+", u"+str(q))
    gnt.set_yticklabels(y_labels) 

    legend_colors = list(range(N))
    legend_names = ["job "+str(j) for j in range(N)]

    for i in range(LV):
        for j in schedule.schedule[i]:
            color = (random.random(), random.random(), random.random())
            legend_colors[j] = Line2D([0], [0], color=color, lw=4)
            for q in range(GV):
                start = schedule.t_q[j][q]
                duration = schedule.c_q[j][q] - schedule.t_q[j][q]
                y_position = (10*i*GV)+(10*q)
                gnt.broken_barh([(start, duration)], (y_position, 9), facecolors = color)

    gnt.legend(legend_colors, legend_names)
    plt.savefig(OUTPUT_DIR+"schedules/"+str(N)+"-"+str(LV)+"-"+str(GV)+".png")
    plt.close(fig)

# Store statistics of some test iteration to log file
def write_log(OUTPUT_DIR, N, LV, GV, GAMMA, EPSILON, METHOD, EPOCHS, makespan, calc_time, epoch, MILP_objval, MILP_calctime):
    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("\n"+METHOD+","+str(N)+","+str(LV)+","+str(GV)+","+str(EPOCHS)+","+str(GAMMA)+","+str(EPSILON)+","+str(makespan)+","+str(calc_time)+","+str(epoch)+","+str(MILP_objval)+","+str(MILP_calctime))
    file.close()

# Store the trained weights of the Neural Network, used as a policy value function
def write_NN_weights(OUTPUT_DIR, N, LV, GV, EPSILON, NN_weights):
    with open(OUTPUT_DIR+"NN_weights/"+str(N)+"-"+str(LV)+"-"+str(GV)+'.pickle','wb') as f:
        pickle.dump(NN_weights, f)

# Heuristic: for each resource, the time that each job costs to process
def heuristic_best_job(tau, N, LV, GV):
    heur_job = dict()

    for i in range(LV):
        heur_j = dict()
        for j in range(N):
            j_total = 0
            for q in range(GV):
                j_total += tau[j][q][i]
            heur_j[j] = j_total
        heur_job[i] = heur_j

    return heur_job

# Heuristic: for each job, the time it costs for each resource if processed on it
def heuristic_best_resource(heur_j):
    heur_r = dict()
    for j in heur_j[0].keys():
        heur_r[j] = dict()
        for r in heur_j.keys():
            heur_r[j][r] = heur_j[r][j]
    return heur_r

# Heuristic: For each resource, the blocking that occurs as a result
#            of executing job j after having executed job o
def heuristic_order(delta, N, LV, GV):
    all_jobs = list(range(N))
    heur_order = dict()             # key = resource i
    for i in range(LV):
        r_dict = dict()             # key = job j
        for j in range(N):
            j_dict = dict()         # key = job o
            other = all_jobs.copy()
            other.remove(j)
            for o in other:
                counter = 0
                spare = 0
                for q in range(GV-1):
                    dj = delta[j][q+1][i]
                    do = delta[o][q][i]
                    blocking = dj-do
                    if blocking < 0:
                        spare += blocking
                    if blocking > 0:
                        if spare >= blocking:
                            spare -= blocking
                        else:
                            blocking -= spare
                            counter += blocking
                j_dict[o] = counter
            r_dict[j] = j_dict
        heur_order[i] = r_dict
    return heur_order

# Execute the NN policy value function with stored weights
# to initialize policy values to be used by JEPS
def load_NN_into_JEPS(N, LV, GV, heur_job, heur_res, heur_order):
    with open('NN_weights.pickle','rb') as f:
        NN_weights = pickle.load(f)
    policy_function = NeuralNetwork(NN_weights)

    policies = np.zeros([LV, N+1])
    for i in range(LV):
        for j in range(N+1):
            inputs = generate_NN_input(i, j, None, 0, heur_job, heur_res, heur_order, N, LV, GV)
            policies[i][j] = policy_function.predict(inputs)

    return policies