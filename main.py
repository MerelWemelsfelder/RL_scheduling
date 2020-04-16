import numpy as np
import random, time
from itertools import chain, combinations
import scipy.stats

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

            if METHOD == "JEPS":
                resources = RL.resources
                states = RL.states
                actions = RL.actions

                for i in range(len(resources)):
                    resource = update_policy_JEPS(resources[i], states, actions, r_best, z, GAMMA, STACT)
                    RL.resources[i] = resource

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    return r_best, best_schedule, epoch_best_found, calc_time, RL

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

def main():
    M = 1       # number of work stations
    LV = 3      # number of resources
    GV = 2      # number of units per resource
    N = 7      # number of jobs

    ALPHA = 0.2     # learning rate (0<α≤1): the extent to which Q-values are updated every timestep
    GAMMA = 0.7     # discount factor (0≤γ≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)   
    EPSILON = 0.4   # probability of choosing a random action (= exploring)

    METHOD = "JEPS"
    STACT = "act"

    EPOCHS = 10000       # set number of epochs to train RL model
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD,STACT,N,LV,GV,EPOCHS,ALPHA,GAMMA,EPSILON,MAKESPAN,TIME,EPOCH_BEST,MILP_OBJVAL,MILP_CALCTIME")
    file.close() 

    # for N in range(1,25):
    #     for LV in range(1,11):
    #         for GV in range(1,11):
    #             for EPOCHS in range(1,10001,1000):

    ins = MILP_instance(M, LV, GV, N)
    timer_start = time.time()
    MILP_schedule, MILP_objval = MILP_solve(M, LV, GV, N)
    timer_finish = time.time()
    MILP_calctime = timer_finish - timer_start
    
    
    delta = np.round(ins.lAreaInstances[0].tau)
    due_dates = ins.lAreaInstances[0].d
    release_dates = np.zeros([N])

    # print("N: "+str(N)+", LV: "+str(LV)+", GV: "+str(GV)+", EPOCHS: "+str(EPOCHS))

    makespan, schedule, epoch, calc_time, RL = find_schedule(M, LV, GV, N, delta, due_dates, release_dates, ALPHA, GAMMA, EPSILON, EPOCHS, METHOD, STACT)

    print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)
    write_log(OUTPUT_DIR, METHOD, STACT, N, LV, GV, EPOCHS, ALPHA, GAMMA, EPSILON, makespan, calc_time, epoch, MILP_objval, MILP_calctime)

if __name__ == '__main__':
    main()