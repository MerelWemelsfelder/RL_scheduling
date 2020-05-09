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

def find_schedule(return_dict, N, M, LV, GV, GAMMA, EPSILON, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT):
    
    timer_start = time.time()

    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(deltas, N, M, LV, GV)
    heur_res = heuristic_best_resource(heur_job, N, M, LV)
    heur_order = heuristic_order(deltas, N, M, LV, GV)

    policies = []
    for v in range(M):
        policies.append(np.zeros([LV[v], N+1]))

    # Load stored weights for the policy value function
    if PHASE == "load":
        with open('NN_weights.pickle','rb') as f:
            NN_weights = pickle.load(f)
        with open('NN_biases.pickle','rb') as f:
            NN_biases = pickle.load(f)
        with open('NN_weights_gradients.pickle','rb') as f:
            NN_weights_gradients = pickle.load(f)
        with open('NN_biases_gradients.pickle','rb') as f:
            NN_biases_gradients = pickle.load(f)

        # Transform the NN weights into policies to be used by JEPS
        if METHOD == "JEPS":
            policies = load_NN_into_JEPS(NN_weights, NN_biases, policies, N, M, LV, GV, due_dates, heur_job, heur_res, heur_order)

    # First epoch, used as initialization of all parameters and results
    RL = MDP(N, M, LV, GV, release_dates, due_dates, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, policies)
    DONE = False
    z = 0
    RL.reset(N, M, LV, GV, release_dates, due_dates)
    while not DONE:
        RL, DONE = RL.step(z, N, M, LV, GV, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_order, PHASE, METHOD)
        z += 1
    schedule = RL.schedule.objectives()
    r_best = schedule.calc_reward(OBJ_FUN)
    best_params = RL.NN.get_params()
    best_params_grad = RL.NN.get_params_gradients()

    return_dict["time"] = TIMEOUT

    # All other epochs
    # for epoch in range(1,EPOCHS):
    epoch = 0
    while True:
        DONE = False
        z = 0
        RL.reset(N, M, LV, GV, release_dates, due_dates)
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, N, M, LV, GV, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_order, PHASE, METHOD)
            z += 1

        # Load the resulting schedule and its objective value
        schedule = RL.schedule.objectives()
        r = schedule.calc_reward(OBJ_FUN)

        # Update the weighs of the policy value function
        if PHASE == "train":
            RL.NN = update_NN(model=RL.NN, X_train=np.array(RL.NN_inputs), y_pred=np.array(RL.NN_predictions), weight_decay=weight_decay, lr=lr, loss=RL.loss, r=r, r_best=r_best)
            # RL.NN.backward(np.array(RL.NN_inputs), (r_best-r)/min(r_best,r))

        # If this schedule has the best objective value found so far,
        # update best schedule and makespan, and update policy values for JEPS
        return_dict["epochs"] = epoch
        return_dict["mdp"] = RL
        if r < r_best:
            r_best = r
            best_params = RL.NN.get_params()
            best_params_grad = RL.NN.get_params_gradients()

            return_dict["best_schedule"] = schedule
            return_dict["epoch_best_found"] = epoch

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

# Test function, which executes the both the MILP and the NN/JEPS algorithm, and stores all relevant information
def test(N, M, LV, GV, GAMMA, EPSILON, OBJ_FUN, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT):
    
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

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    # return_dict["best_schedule"] = None
    # return_dict["epoch_best_found"] = 0
    # return_dict["time"] = TIMEOUT
    # return_dict["mdp"] = None
    p = multiprocessing.Process(target=find_schedule, args=(return_dict, N, M, LV, GV, GAMMA, EPSILON, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        print("Runtime error: abort mission.\n")
        p.terminate()
        p.join()
    
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

        plot_schedule(OUTPUT_DIR, schedule, N, M, LV, GV)
        # print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)
        write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, OBJ_FUN, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients)
        write_log(OUTPUT_DIR, N, M, LV, GV, GAMMA, EPSILON, layer_dims, weight_decay, lr, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch_best_found, MILP_objval, MILP_calctime)

def main():
    N = 15         # number of jobs
    M = 1           # number of work stations
    LV = [6]      # number of resources in each work station
    GV = [4]      # number of units in each resource of each work station
    
    # ALPHA = 0.4   # discount factor (0≤α≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)
    GAMMA = 0.8     # learning rate (0<γ≤1): the extent to which Q-values are updated every timestep / epoch
    EPSILON = 0.9   # probability of choosing a random action (= exploring)

    OBJ_FUN = {
        "Cmax": 1,
        "Tsum": 10,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    layer_dims = [4,9,11,6,4,1]
    NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    NN_weights_gradients = np.zeros_like(NN_weights)
    NN_biases_gradients = np.zeros_like(NN_biases)

    weight_decay = 0.0001
    lr = 0.2

    PHASE = "train"     # train / load
    METHOD = "NN"     # JEPS / Q_learning / NN

    EPOCHS = 10000
    TIMEOUT = 3*60
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD\tN\tM\tLV\tGV\tEPOCHS\tGAMMA\tEPSILON\tLAYER_DIMS\tWEIGHT_DECAY\tLR\tCMAX_WEIGHT\tTSUM_WEIGHT\tMAKESPAN\tTSUM\tTMAX\tTN\tTIME\tEPOCH_BEST\tMILP_OBJVAL\tMILP_CALCTIME")
    file.close()

    # layer_dims = [12,3,5,21,18]
    #     all_combinations = list(itertools.combinations_with_replacement(layer_dims, 5))
    #     for c in range(len(all_combinations)):
    #         all_combinations[c] = [7] + list(all_combinations[c]) + [1]
    #     for layer_dims in all_combinations:
    #         NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    #         NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    #         NN_weights_gradients = np.zeros_like(NN_weights)
    #         NN_biases_gradients = np.zeros_like(NN_biases)

    for N in range(14,26):
        for LV in range(4,11):
            for GV in range(4,11):
                for weights in [[1,0],[0,1],[1,10],[1,100]]:
                    OBJ_FUN["Cmax"] = weights[0]
                    OBJ_FUN["Tsum"] = weights[1]
                    test(N, M, [LV], [GV], GAMMA, EPSILON, OBJ_FUN, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT)
    
if __name__ == '__main__':
    main()