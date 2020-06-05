import numpy as np
import random, time
import itertools
import math
import multiprocessing

from MDP import *
from JEPS import *
from MILP import *
from NN import *
from utils import *

def find_schedule(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):

    timer_start = time.time()

    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(deltas, N, M, LV, GV)
    heur_res = heuristic_best_resource(heur_job, N, M, LV)
    heur_blocking = heuristic_blocking(deltas, N, M, LV, GV)
    heur_rev_blocking = heuristic_reverse_blocking(deltas, N, M, LV, GV)

    policies = []
    for v in range(M):
        policies.append(np.zeros([LV[v], N]))

    # Load stored weights for the policy value function
    if PHASE == "load":
        with open('weights/'+str(layer_dims)+'-MILP-weights.pickle','rb') as f:
            NN_weights = pickle.load(f)
        with open('weights/'+str(layer_dims)+'-MILP-biases.pickle','rb') as f:
            NN_biases = pickle.load(f)
        with open('weights/'+str(layer_dims)+'-MILP-weights_grad.pickle','rb') as f:
            NN_weights_gradients = pickle.load(f)
        with open('weights/'+str(layer_dims)+'-MILP-biases_grad.pickle','rb') as f:
            NN_biases_gradients = pickle.load(f)

        # Transform the NN weights into policies to be used by JEPS
        if METHOD == "JEPS":
            policies = load_NN_into_JEPS(NN_weights, NN_biases, policies, N, M, LV, GV, due_dates, heur_job, heur_res, heur_blocking, heur_rev_blocking)

    # First epoch, used as initialization of all parameters and results
    RL = MDP(N, M, LV, GV, release_dates, due_dates, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, policies)
    DONE = False
    z = 0
    RL.reset(N, M, LV, GV, release_dates, due_dates)
    while not DONE:
        RL, DONE = RL.step(z, N, M, LV, GV, CONFIG, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_blocking, heur_rev_blocking, PHASE, METHOD)
        z += 1
    schedule = RL.schedule.objectives()
    r_best = schedule.calc_reward(OBJ_FUN)
    best_params = RL.NN.get_params()
    best_params_grad = RL.NN.get_params_gradients()

    comparing_score = schedule.Cmax + (100*schedule.Tsum)

    return_dict["time"] = TIMEOUT
    return_dict["best_schedule"] = schedule
    return_dict["epoch_best_found"] = 0
    return_dict["epochs"] = 0
    return_dict["mdp"] = RL

    # All other epochs
    # for epoch in range(1,EPOCHS):
    epoch = 0
    while True:
    # while r_best > MILP_objval:
    # while (EPSILON > 0.1) and (comparing_score > round(MILP_objval,0)):

        if epoch%1000==0:
            makespan = return_dict["best_schedule"].Cmax
            Tsum = return_dict["best_schedule"].Tsum
            Tmax = return_dict["best_schedule"].Tmax
            Tn = return_dict["best_schedule"].Tn
            write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, epoch, OBJ_FUN, makespan, Tsum, Tmax, Tn, 0, return_dict["epoch_best_found"], MILP_objval, 0, 0)

            EPSILON *= 1 - EPSILON_DECREASE

        DONE = False
        z = 0
        RL.reset(N, M, LV, GV, release_dates, due_dates)
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, N, M, LV, GV, CONFIG, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_blocking, heur_rev_blocking, PHASE, METHOD)
            z += 1

        # Load the resulting schedule and its objective value
        schedule = RL.schedule.objectives()
        r = schedule.calc_reward(OBJ_FUN)

        # Update the weighs of the policy value function
        # if PHASE == "train":
        RL.NN = update_NN(model=RL.NN, X_train=np.array(RL.NN_inputs), y_pred=np.array(RL.NN_predictions), weight_decay=weight_decay, GAMMA=GAMMA, loss=RL.loss, r=r, r_best=r_best, MILP_objval=MILP_objval)

        # If this schedule has the best objective value found so far,
        # update best schedule and makespan, and update policy values for JEPS
        return_dict["epochs"] = epoch
        return_dict["mdp"] = RL
        if r < r_best:
            r_best = r
            return_dict["best_schedule"] = schedule
            return_dict["epoch_best_found"] = epoch

            comparing_score = schedule.Cmax + (100*schedule.Tsum)

            if (PHASE == "load") and (METHOD == "JEPS"):
                for v in range(M):
                    resources = RL.workstations[v].resources
                    for i in range(LV[v]):
                        resources[i] = update_policy_JEPS(resources[i], RL.actions, z, GAMMA)
                    RL.workstations[v].resources = resources

        epoch += 1

    timer_finish = time.time()
    calc_time = timer_finish - timer_start

    return_dict["time"] = calc_time
    return_dict["mdp"] = RL

    return return_dict

# Test function, which executes the both the MILP and the NN/JEPS algorithm, and stores all relevant information
def test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):

    ins = MILP_instance(M, LV, GV, N)
    MILP_objval = 0
    MILP_calctime = 0
    # timer_start = time.time()
    # MILP_schedule, MILP_objval = MILP_solve(M, LV, GV, N)
    # timer_finish = time.time()
    # MILP_calctime = timer_finish - timer_start

    # Load durations of jobs on units, and all job's due dates and release dates
    deltas = []
    due_dates = []
    for v in range(M):
        deltas.append(np.round(ins.lAreaInstances[v].tau))
        due_dates.append(ins.lAreaInstances[v].d)
    release_dates = np.zeros([N])

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=find_schedule, args=(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        print("Runtime error: abort mission.\n")
        p.terminate()
        p.join()

    # return_dict = find_schedule(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT)
    
        EPOCHS = return_dict["epochs"]
        schedule = return_dict["best_schedule"]
        epoch_best_found = return_dict["epoch_best_found"]
        calc_time = return_dict["time"]
        RL = return_dict["mdp"]

        makespan = schedule.Cmax
        Tsum = schedule.Tsum
        Tmax = schedule.Tmax
        Tn = schedule.Tn

        params = RL.NN.get_params()
        NN_weights = [params[i] for i in range(0,len(params),2)]
        NN_biases = [params[i] for i in range(1,len(params),2)]
        grads = RL.NN.get_params_gradients()
        NN_weights_gradients = [grads[i] for i in range(0,len(grads),2)]
        NN_biases_gradients = [grads[i] for i in range(1,len(grads),2)]

        if PHASE == "train":
            write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, OBJ_FUN, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, GAMMA_DECREASE)

        plot_schedule(OUTPUT_DIR, schedule, N, M, LV, GV)
        write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch_best_found, MILP_objval, MILP_calctime, MILP_TIMEOUT)
        # write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, 0, OBJ_FUN, 0, 0, 0, 0, 1, 0, MILP_objval, MILP_calctime, MILP_TIMEOUT)

def main():
    N = 15        # number of jobs
    M = 1         # number of work stations
    LV = [6]      # number of resources in each work station
    GV = [6]      # number of units in each resource of each work station
    
    GAMMA = 0.3     # learning rate (0<γ≤1): the extent to which policy weights are updated every epoch
    EPSILON = 1.0   # probability of choosing a random action (= exploring)

    GAMMA_DECREASE = 0.1
    EPSILON_DECREASE = 0.1  # 0.025 for training, 0.1 for loading
    weight_decay = 0.00001

    OBJ_FUN = {
        "Cmax": 1,
        "Tsum": 10,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    layer_dims = [15, 11, 6, 1]
    INPUT_CONFIGS = ["all_vars", "XV", "minmax_large", "minmax_small", "generalizability", "high", "absolute", "relative", "generalizability_T"]
    CONFIG = 7

    NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    NN_weights_gradients = np.zeros_like(NN_weights)
    NN_biases_gradients = np.zeros_like(NN_biases)

    PHASE = "load"     # train / load
    METHOD = "NN"     # JEPS / Q_learning / NN

    EPOCHS = 30000
    TIMEOUT = 15*60
    MILP_TIMEOUT = 0
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD\tPHASE\tN\tM\tLV\tGV\tCONFIG\tEPOCHS\tGAMMA\tGAMMA_DECREASE\tEPSILON\tEPSILON_DECREASE\tLAYER_DIMS\tWEIGHT_DECAY\tCMAX_WEIGHT\tTSUM_WEIGHT\tMAKESPAN\tTSUM\tTMAX\tTN\tTIME\tEPOCH_BEST\tMILP_OBJVAL\tMILP_CALCTIME\tMILP_TIMEOUT")
    file.close()

    # for N in [8, 13, 18]:
    #     for LV in [[4], [9]]:
    #         test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)
    #         GAMMA *= (1-GAMMA_DECREASE)

    for i in range(3):
        for EPSILON in [0, 0.2]:
            # for PHASE in ["load", "train"]:
            for TIMEOUT in [3*60, 30*60]: # 30, 60
    
                N = 4
                LV = [2 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

                N = 8
                LV = [4 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

                N = 18
                LV = [8 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

                N = 32
                LV = [13 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

                N = 47
                LV = [17 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

                N = 64
                LV = [21 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

                N = 81
                LV = [26 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

if __name__ == '__main__':
    main()