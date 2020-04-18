import numpy as np
import random, time
from itertools import chain, combinations
import scipy.stats
import pickle
import matplotlib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import os

from MDP import *
from JEPS import *
from Q_learning import *
from MILP import *

def find_schedule(M, LV, GV, N, delta, due_dates, release_dates, ALPHA, GAMMA, EPSILON, EPOCHS, METHOD, STACT):
    
    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(delta, LV, GV, N)
    heur_res = heuristic_best_resource(heur_job)
    heur_order = heuristic_order(delta, LV, GV, N)

    if STACT == "st_act":                       # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([2**N, N+1])     # states, actions
    if STACT == "act":                          # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([N+1])           # actions

    RL = MDP(LV, GV, N, policy_init, due_dates, release_dates)            # initialize MDP
    r_best = 99999
    best_schedule = []
    best_policy = np.zeros([LV, N+1])
    epoch_best_found = 0
    timer_start = time.time()
    for epoch in range(EPOCHS):
        # if epoch%100==0:
        #     print(epoch)

        DONE = False
        z = 0
        RL.reset(due_dates, release_dates, LV, GV, N)
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, GV, N, METHOD, delta, ALPHA, GAMMA, EPSILON, STACT, heur_job, heur_res, heur_order)
            z += 1

        schedule = RL.schedule.objectives()
        r = schedule.Cmax
        if r < r_best:
            r_best = r
            best_schedule = schedule
            epoch_best_found = epoch

            for i in range(len(RL.resources)):
                best_policy[i] = RL.resources[i].policy

            if METHOD == "JEPS":
                resources = RL.resources
                states = RL.states
                actions = RL.actions

                for i in range(len(resources)):
                    resource = update_policy_JEPS(resources[i], states, actions, r_best, z, GAMMA, STACT)
                    RL.resources[i] = resource

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    return r_best, best_schedule, best_policy, epoch_best_found, calc_time, RL

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

def plot_schedule(OUTPUT_DIR, schedule, N, LV, GV):
    gantt = []
    for i in range(LV):
        for j in schedule.schedule[i]:
            for q in range(GV):
                start = schedule.t_q[j][q]
                end = schedule.c_q[j][q]
                gantt.append(dict(Task="u_"+str(i)+str(q), Start=start, Finish=end, Resource="job "+str(j)))

    colors = dict()
    for j in range(N):
        colors["job "+str(j)] = (random.random(), random.random(), random.random())

    print(colors) 
    fig = ff.create_gantt(gantt, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    fig.write_image(OUTPUT_DIR+'schedule.png')

def main():
    M = 1       # number of work stations
    LV = 3      # number of resources
    GV = 2      # number of units per resource
    N = 5       # number of jobs

    ALPHA = 0.2     # learning rate (0<α≤1): the extent to which Q-values are updated every timestep
    GAMMA = 0.7     # discount factor (0≤γ≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)   
    EPSILON = 0.3   # probability of choosing a random action (= exploring)

    METHOD = "JEPS"
    STACT = "act"

    EPOCHS = 8000       # set number of epochs to train RL model
    OUTPUT_DIR = '../output/'

    # file = open(OUTPUT_DIR+"log.csv",'a')
    # file.write("METHOD,STACT,N,LV,GV,EPOCHS,ALPHA,GAMMA,EPSILON,MAKESPAN,TIME,EPOCH_BEST,MILP_OBJVAL,MILP_CALCTIME")
    # file.close() 

    # training_inputs = []
    # training_outputs = []

    ins = MILP_instance(M, LV, GV, N)
    timer_start = time.time()
    MILP_schedule, MILP_objval = MILP_solve(M, LV, GV, N)
    timer_finish = time.time()
    MILP_calctime = timer_finish - timer_start
    
    
    delta = np.round(ins.lAreaInstances[0].tau)
    due_dates = ins.lAreaInstances[0].d
    release_dates = np.zeros([N])

    makespan, schedule, policy, epoch, calc_time, RL = find_schedule(M, LV, GV, N, delta, due_dates, release_dates, ALPHA, GAMMA, EPSILON, EPOCHS, METHOD, STACT)

    plot_schedule(OUTPUT_DIR, schedule, N, LV, GV)
    print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)

    write_log(OUTPUT_DIR, METHOD, STACT, N, LV, GV, EPOCHS, ALPHA, GAMMA, EPSILON, makespan, calc_time, epoch, MILP_objval, MILP_calctime)

    if makespan <= MILP_objval:
        training_inputs.append(delta)
        training_outputs.append(policy)

        # write_training_files(OUTPUT_DIR, training_inputs, training_outputs)

if __name__ == '__main__':
    main()