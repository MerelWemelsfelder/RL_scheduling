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
# from Q_learning import *
from MILP import *
from NN import *
from utils import *

def find_schedule(M, N, LV, GV, GAMMA, EPSILON, delta, due_dates, release_dates, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR):
    
    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(delta, N, LV, GV)
    heur_res = heuristic_best_resource(heur_job)
    heur_order = heuristic_order(delta, N, LV, GV)

    if (PHASE == "load") and (METHOD == "JEPS"):
        policies = load_NN_into_JEPS(N, LV, GV, heur_job, heur_res, heur_order)

    # First epoch, used as initialization of all parameters and results
    RL = MDP(N, LV, GV, release_dates, due_dates, NN_weights, policies)
    timer_start = time.time()
    DONE = False
    z = 0
    RL.reset(N, LV, GV, release_dates, due_dates)
    while not DONE:
        RL, DONE = RL.step(z, N, LV, GV, GAMMA, EPSILON, delta, heur_job, heur_res, heur_order, PHASE)
        z += 1
    schedule = RL.schedule.objectives()
    r_best = schedule.calc_reward(R_WEIGHTS)
    best_schedule = schedule
    epoch_best_found = 0

    # All other epochs
    for epoch in range(1,EPOCHS):

        DONE = False
        z = 0
        RL.reset(N, LV, GV, release_dates, due_dates)
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, N, LV, GV, GAMMA, EPSILON, delta, heur_job, heur_res, heur_order)
            z += 1

        # Load the resulting schedule and its objective value
        schedule = RL.schedule.objectives()
        r = schedule.calc_reward(R_WEIGHTS)

        # Update the weighs of the policy value function
        if PHASE == "train":
            RL.policy_function.backpropagation((r_best-r)/r_best, np.array(RL.NN_inputs), np.array(RL.NN_predictions))

        # If this schedule has the best objective value found so far,
        # update best schedule and makespan, and update policy values for JEPS
        if r < r_best:
            r_best = r
            best_schedule = schedule
            epoch_best_found = epoch

            if (PHASE == "load") and (METHOD == "JEPS"):
                for i in range(len(RL.resources)):
                    RL.resources[i] = update_policy_JEPS(RL.resources[i], RL.states, RL.actions, z, GAMMA)

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    return r_best, best_schedule, epoch_best_found, calc_time, RL, RL.policy_function.weights

# Test function, which executes the both the MILP and the NN/JEPS algorithm, and stores all relevant information
def test(M, N, LV, GV, GAMMA, EPSILON, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR):
    
    ins = MILP_instance(M, LV, GV, N)
    # MILP_objval = 0
    # MILP_calctime = 0
    timer_start = time.time()
    MILP_schedule, MILP_objval = MILP_solve(M, LV, GV, N)
    timer_finish = time.time()
    MILP_calctime = timer_finish - timer_start

    # Load durations of jobs on units, and all job's due dates and release dates
    delta = np.round(ins.lAreaInstances[0].tau)
    due_dates = ins.lAreaInstances[0].d
    release_dates = np.zeros([N])

    # Determine the upper bound for the schedule's makespan
    # max_d = []
    # for j in range(N):
    #     d = []
    #     for i in range(LV):
    #         d.append(sum([x[i] for x in delta[j]]))
    #     max_d.append(max(d))
    # upper_bound = sum(max_d) + (N-1)

    makespan, schedule, epoch, calc_time, RL, NN_weights = find_schedule(M, N, LV, GV, GAMMA, EPSILON, delta, due_dates, release_dates, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR)

    plot_schedule(OUTPUT_DIR, schedule, N, LV, GV)
    # print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)
    write_NN_weights(OUTPUT_DIR, N, LV, GV, EPSILON, NN_weights)
    write_log(OUTPUT_DIR, N, LV, GV, GAMMA, EPSILON, METHOD, EPOCHS, makespan, calc_time, epoch, MILP_objval, MILP_calctime)

    return NN_weights

def main():
    M = 1       # number of work stations
    LV = 3      # number of resources
    GV = 2      # number of units per resource
    N = 6       # number of jobs

    # ALPHA = 0.4     # discount factor (0≤α≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)
    GAMMA = 0.8     # learning rate (0<γ≤1): the extent to which Q-values are updated every timestep / epoch
    EPSILON = 0.5   # probability of choosing a random action (= exploring)

    R_WEIGHTS = {
        "Cmax": 1,
        "Tsum": 0,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    NN_weights = np.random.rand(8)

    PHASE = "train"     # train / load
    METHOD = "JEPS"     # JEPS / Q_learning / NN

    EPOCHS = 3000
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD,N,LV,GV,EPOCHS,GAMMA,EPSILON,MAKESPAN,TIME,EPOCH_BEST,MILP_OBJVAL,MILP_CALCTIME")
    file.close() 

    for N in range(2,26):
        for LV in range(2,16):
            for GV in range(1,6):
                print(str(N)+","+str(LV)+","+str(GV)+","+str(EPSILON))
                NN_weights = test(M, N, LV, GV, GAMMA, EPSILON, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR)
    
if __name__ == '__main__':
    main()