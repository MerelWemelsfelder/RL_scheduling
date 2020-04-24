import numpy as np
import random, time
from itertools import chain, combinations
import scipy.stats
import pickle
import matplotlib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import os
from matplotlib.lines import Line2D

from MDP import *
from JEPS import *
from Q_learning import *
from MILP import *
from NN_init import *

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

def write_log(OUTPUT_DIR, METHOD, STACT, N, LV, GV, EPOCHS, ALPHA, GAMMA, EPSILON, makespan, calc_time, epoch, MILP_objval, MILP_calctime):
    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("\n"+METHOD+","+STACT+","+str(N)+","+str(LV)+","+str(GV)+","+str(EPOCHS)+","+str(ALPHA)+","+str(GAMMA)+","+str(EPSILON)+","+str(makespan)+","+str(calc_time)+","+str(epoch)+","+str(MILP_objval)+","+str(MILP_calctime))
    file.close()

def write_training_files(OUTPUT_DIR, training_inputs, training_outputs):
    with open(OUTPUT_DIR+'training_inputs.pickle','wb') as f:
        pickle.dump(training_inputs, f)
    with open(OUTPUT_DIR+'training_outputs.pickle','wb') as f:
        pickle.dump(training_outputs, f)

def write_NN_weights(OUTPUT_DIR, N, LV, GV, EPSILON, NN_weights):
    with open(OUTPUT_DIR+"NN_weights/"+str(N)+"-"+str(LV)+"-"+str(GV)+"-"+str(EPSILON)+'.pickle','wb') as f:
        pickle.dump(NN_weights, f)

def plot_schedule(OUTPUT_DIR, schedule, N, LV, GV):

    fig, gnt = plt.subplots() 
      
    # Setting axis limits
    gnt.set_ylim(0, LV*GV*10)
    gnt.set_xlim(0, max(schedule.c)) 
      
    # Setting labels for x-axis and y-axis 
    gnt.set_xlabel('time') 
    gnt.set_ylabel('resources ri, units uq')
      
    # Setting ticks
    gnt.set_yticks(list(range(5,LV*GV*10,10)))
    y_labels = []
    for i in range(LV):
        for q in range(GV):
            y_labels.append("r"+str(i)+", u"+str(q))
    gnt.set_yticklabels(y_labels) 

    legend_colors = list(range(N))
    legend_names = ["job "+str(j) for j in range(N)]

    # Declaring a bar in schedule
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

def find_schedule(OUTPUT_DIR, M, LV, GV, N, delta, due_dates, release_dates, ALPHA, GAMMA, EPSILON, R_WEIGHTS, NN_WEIGHTS_LEN, EPOCHS, METHOD, STACT):
    
    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(delta, LV, GV, N)
    heur_res = heuristic_best_resource(heur_job)
    heur_order = heuristic_order(delta, LV, GV, N)

    # if STACT == "st_act":                           # st_act for state-action pairs, act for only actions
    #     policy_init = np.zeros([LV, 2**N, N+1])     # states, actions
    # elif STACT == "act":                            # st_act for state-action pairs, act for only actions
    #     policy_init = np.zeros([LV, N+1])           # actions
    # elif STACT == "NN":
    #     policy_init = initial_policies_from(delta)

    RL = MDP(LV, GV, N, due_dates, release_dates, NN_WEIGHTS_LEN)            # initialize MDP
    
    timer_start = time.time()

    DONE = False
    z = 0
    RL.reset(due_dates, release_dates, LV, GV, N)
    while not DONE:
        RL, DONE = RL.step(z, N, LV, GV, ALPHA, GAMMA, EPSILON, delta, STACT, heur_job, heur_res, heur_order)
        z += 1
    schedule = RL.schedule.objectives()
    r_best = schedule.calc_reward(R_WEIGHTS)
    best_schedule = schedule
    epoch_best_found = 0

    for epoch in range(1,EPOCHS):

        DONE = False
        z = 0
        RL.reset(due_dates, release_dates, LV, GV, N)
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, N, LV, GV, ALPHA, GAMMA, EPSILON, delta, STACT, heur_job, heur_res, heur_order)
            z += 1

        schedule = RL.schedule.objectives()
        r = schedule.calc_reward(R_WEIGHTS)
        RL.policy_function.backpropagation((r_best-r)/r_best, np.array(RL.NN_inputs), np.array(RL.NN_predictions))

        if r < r_best:
            r_best = r
            best_schedule = schedule
            epoch_best_found = epoch

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    return r_best, best_schedule, epoch_best_found, calc_time, RL, RL.policy_function.weights

def test(M, LV, GV, N, ALPHA, GAMMA, EPSILON, R_WEIGHTS, NN_WEIGHTS_LEN, METHOD, STACT, EPOCHS, OUTPUT_DIR):
    
    ins = MILP_instance(M, LV, GV, N)
    # MILP_objval = 0
    # MILP_calctime = 0
    timer_start = time.time()
    MILP_schedule, MILP_objval = MILP_solve(M, LV, GV, N)
    timer_finish = time.time()
    MILP_calctime = timer_finish - timer_start

    delta = np.round(ins.lAreaInstances[0].tau)
    due_dates = ins.lAreaInstances[0].d
    release_dates = np.zeros([N])

    # max_d = []
    # for j in range(N):
    #     d = []
    #     for i in range(LV):
    #         d.append(sum([x[i] for x in delta[j]]))
    #     max_d.append(max(d))
    # upper_bound = sum(max_d) + (N-1)

    makespan, schedule, epoch, calc_time, RL, NN_weights = find_schedule(OUTPUT_DIR, M, LV, GV, N, delta, due_dates, release_dates, ALPHA, GAMMA, EPSILON, R_WEIGHTS, NN_WEIGHTS_LEN, EPOCHS, METHOD, STACT) # upper_bound

    plot_schedule(OUTPUT_DIR, schedule, N, LV, GV)
    # print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)

    write_NN_weights(OUTPUT_DIR, N, LV, GV, EPSILON, NN_weights)
    write_log(OUTPUT_DIR, METHOD, STACT, N, LV, GV, EPOCHS, ALPHA, GAMMA, EPSILON, makespan, calc_time, epoch, MILP_objval, MILP_calctime)

def main():
    M = 1       # number of work stations
    LV = 3      # number of resources
    GV = 2      # number of units per resource
    N = 6       # number of jobs

    ALPHA = 0.4     # discount factor (0≤α≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)
    GAMMA = 0.8     # learning rate (0<γ≤1): the extent to which Q-values are updated every timestep / epoch
    EPSILON = 0.4   # probability of choosing a random action (= exploring)

    R_WEIGHTS = {
        "Cmax": 1,
        "Tsum": 0,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    NN_WEIGHTS_LEN = 8

    METHOD = "NN" # JEPS, Q_learning, NN
    STACT = "act"    # act, st_act, NN

    EPOCHS = 3000
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD,STACT,N,LV,GV,EPOCHS,ALPHA,GAMMA,EPSILON,MAKESPAN,TIME,EPOCH_BEST,MILP_OBJVAL,MILP_CALCTIME")
    file.close() 

    for N in range(1,26):
        for LV in range(1,16):
            for GV in range(1,6):
                for EPSILON in [x/10 for x in range(1,11)]:
                    print(str(N)+","+str(LV)+","+str(GV)+","+str(EPSILON))
                    test(M, LV, GV, N, ALPHA, GAMMA, EPSILON, R_WEIGHTS, NN_WEIGHTS_LEN, METHOD, STACT, EPOCHS, OUTPUT_DIR)

    
if __name__ == '__main__':
    main()