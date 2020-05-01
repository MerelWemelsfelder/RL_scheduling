import numpy as np
import random, time

from MDP import *
from JEPS import *
from MILP import *
from NN import *
from utils import *

def find_schedule(N, M, LV, GV, GAMMA, EPSILON, deltas, due_dates, release_dates, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR):
    
    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(deltas, N, M, LV, GV)
    heur_res = heuristic_best_resource(heur_job)
    heur_order = heuristic_order(deltas, N, M, LV, GV)

    policies = []
    for v in range(M):
        policies.append(np.zeros([LV[v], N+1]))

    # Load stored weights for the policy value function
    if PHASE == "load":
        with open('NN_weights.pickle','rb') as f:
            NN_weights = pickle.load(f)

        # Transform the NN weights into policies to be used by JEPS
        if METHOD == "JEPS":
            policies = load_NN_into_JEPS(NN_weights, policies, N, M, LV, GV, due_dates, heur_job, heur_res, heur_order)

    # First epoch, used as initialization of all parameters and results
    RL = MDP(N, M, LV, GV, release_dates, due_dates, NN_weights, policies)
    timer_start = time.time()
    DONE = False
    z = 0
    RL.reset(N, M, LV, GV, release_dates, due_dates)
    while not DONE:
        RL, DONE = RL.step(z, N, M, LV, GV, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_order, PHASE, METHOD)
        z += 1
    schedule = RL.schedule.objectives()
    r_best = schedule.calc_reward(R_WEIGHTS)
    best_schedule = schedule
    epoch_best_found = 0

    # All other epochs
    for epoch in range(1,EPOCHS):

        DONE = False
        z = 0
        RL.reset(N, M, LV, GV, release_dates, due_dates)
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, N, M, LV, GV, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_order, PHASE, METHOD)
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
                    RL.resources[i] = update_policy_JEPS(RL.resources[i], RL.actions, z, GAMMA)

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    return best_schedule, epoch_best_found, calc_time, RL, RL.policy_function.weights

# Test function, which executes the both the MILP and the NN/JEPS algorithm, and stores all relevant information
def test(N, M, LV, GV, GAMMA, EPSILON, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR):
    
    ins = MILP_instance(M, LV, GV, N)
    MILP_objval = 0
    MILP_calctime = 0
    timer_start = time.time()
    MILP_schedule, MILP_objval = MILP_solve(M, LV, GV, N)
    timer_finish = time.time()
    MILP_calctime = timer_finish - timer_start

    # Load durations of jobs on units, and all job's due dates and release dates
    deltas = []
    due_dates = []
    for v in range(M):
        deltas.append(np.round(ins.lAreaInstances[v].tau))
        due_dates.append(ins.lAreaInstances[v].d)
    release_dates = np.zeros([N])

    schedule, epoch, calc_time, RL, NN_weights = find_schedule(N, M, LV, GV, GAMMA, EPSILON, deltas, due_dates, release_dates, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR)

    makespan = schedule.Cmax
    Tsum = schedule.Tsum
    Tmax = schedule.Tmax
    Tn = schedule.Tn

    # plot_schedule(OUTPUT_DIR, schedule, N, M, LV, GV)
    # print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)
    # write_NN_weights(OUTPUT_DIR, N, LV, GV, EPSILON, NN_weights)
    write_log(OUTPUT_DIR, N, M, LV, GV, GAMMA, EPSILON, METHOD, EPOCHS, makespan, Tsum, Tmax, Tn, calc_time, epoch, MILP_objval, MILP_calctime)

    return NN_weights

def main():
    N = 15          # number of jobs
    M = 2           # number of work stations
    LV = [5,5]      # number of resources in each work station
    GV = [3,4]      # number of units in each resource of each work station
    
    # ALPHA = 0.4   # discount factor (0≤α≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)
    GAMMA = 0.8     # learning rate (0<γ≤1): the extent to which Q-values are updated every timestep / epoch
    EPSILON = 0.2   # probability of choosing a random action (= exploring)

    R_WEIGHTS = {
        "Cmax": 1,
        "Tsum": 1,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    NN_weights = np.random.rand(10)

    PHASE = "load"     # train / load
    METHOD = "NN"     # JEPS / Q_learning / NN

    EPOCHS = 5000
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD,N,M,LV,GV,EPOCHS,GAMMA,EPSILON,MAKESPAN,TSUM,TMAX,TN,TIME,EPOCH_BEST,MILP_OBJVAL,MILP_CALCTIME")
    file.close()

    for N in range(15,21):
        for M in range(1,5):
            for METHOD in ["NN","JEPS"]:
                for lv in range(2,6):
                    for gv in range(1,5):
                        LV = [lv for v in range(M)]
                        GV = [gv for v in range(M)]
                
                        NN_weights = test(N, M, LV, GV, GAMMA, EPSILON, R_WEIGHTS, NN_weights, PHASE, METHOD, EPOCHS, OUTPUT_DIR)
    
if __name__ == '__main__':
    main()