import numpy as np
import random, time
import itertools
import math
import multiprocessing
from numpy import load
import shap

from MDP import *
from JEPS import *
from MILP import *
from NN import *
from utils import *
from MCTS import *

def find_schedule(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, MCTS_objval):

    timer_start = time.time()

    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(deltas, N, M, LV, GV)
    heur_res = heuristic_best_resource(heur_job, N, M, LV)
    heur_blocking = heuristic_blocking(deltas, N, M, LV, GV)
    heur_rev_blocking = heuristic_reverse_blocking(deltas, N, M, LV, GV)

    # policies = []
    # for v in range(M):
    #     policies.append(np.zeros([LV[v], N]))

    # Load stored weights for the policy value function
    if PHASE == "load":
        with open('weights/'+str(layer_dims)+'-JSSP-weights.pickle','rb') as f:
            NN_weights = pickle.load(f)
        with open('weights/'+str(layer_dims)+'-JSSP-biases.pickle','rb') as f:
            NN_biases = pickle.load(f)
        with open('weights/'+str(layer_dims)+'-JSSP-weights_grad.pickle','rb') as f:
            NN_weights_gradients = pickle.load(f)
        with open('weights/'+str(layer_dims)+'-JSSP-biases_grad.pickle','rb') as f:
            NN_biases_gradients = pickle.load(f)

    # First epoch, used as initialization of all parameters and results
    RL = MDP(N, M, LV, GV, release_dates, due_dates, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients)
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
    epoch = 1
    while True:

        if epoch%1000==0:
            EPSILON *= 1 - EPSILON_DECREASE

            makespan = return_dict["best_schedule"].Cmax
            Tsum = return_dict["best_schedule"].Tsum
            Tmax = return_dict["best_schedule"].Tmax
            Tn = return_dict["best_schedule"].Tn
            write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, epoch, OBJ_FUN, makespan, Tsum, Tmax, Tn, 0, return_dict["epoch_best_found"], MILP_objval, 0, 0, 0, MCTS_objval)

        DONE = False
        z = 0
        RL.reset(N, M, LV, GV, release_dates, due_dates)
        
        # Take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, N, M, LV, GV, CONFIG, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_blocking, heur_rev_blocking, PHASE, METHOD)
            z += 1

        # Retrieve generated schedule and its objective value
        schedule = RL.schedule.objectives()
        r = schedule.calc_reward(OBJ_FUN)
        
        # Update the weighs of the policy value function
        # LOADING
        # score = (r_best-r)
        # if min(r_best, r) > 0:
        #     score /= min(r_best, r)

        # TRAINING
        y_pred = np.array(RL.NN_predictions)
        score = (MILP_objval-r)
        if min(MILP_objval, r) > 0:
            score /= min(MILP_objval, r)
        y_true = y_pred + (score * y_pred)

        # if r > MILP_objval:
        #     y_true = (r - MILP_objval) / r
        # else:
        #     y_true = 0

        write_training_batch(OUTPUT_DIR=OUTPUT_DIR, X_train=RL.NN_inputs, y_true=y_true)
        RL.NN = update_NN(model=RL.NN, X_train=np.array(RL.NN_inputs), y_pred=y_pred, y_true=y_true, weight_decay=weight_decay, GAMMA=GAMMA, loss=RL.loss)
        
        # If this schedule has the best objective value found so far,
        # update best schedule and makespan, and update policy values
        return_dict["epochs"] = epoch
        return_dict["mdp"] = RL
        if r < r_best:
            r_best = r
            return_dict["best_schedule"] = schedule
            return_dict["epoch_best_found"] = epoch

        epoch += 1

    timer_finish = time.time()
    calc_time = timer_finish - timer_start

    return_dict["time"] = calc_time
    return_dict["mdp"] = RL

    return return_dict

# Test function, which executes the both the MILP and the NN/JEPS algorithm, and stores all relevant information
def test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):

    ins = MILP_instance(M, LV, GV, N)
    MILP_objval = 0
    MILP_calctime = 0
    MCTS_objval = 0
    MCTS_calctime = 0

    # UNCOMMENT TO RUN MILP
    timer_start = time.time()
    MILP_schedule, MILP_objval = MILP_solve(ins, M, LV, GV, N)
    timer_finish = time.time()
    MILP_calctime = timer_finish - timer_start

    deltas = []
    due_dates = []
    for v in range(M):
        deltas.append(np.round(ins.lAreaInstances[v].tau))
        due_dates.append(ins.lAreaInstances[v].d)
    release_dates = np.zeros([N])

    # MONTE CARLO TREE SEARCH
    # MCTS = Monte_Carlo(N, M, LV, GV, due_dates)
    # timer_start = time.time()
    # root = MCTS.search(MCTS_BUDGET, N, M, LV, GV, deltas, EPSILON, OBJ_FUN)
    # timer_finish = time.time()
    # MCTS_calctime = timer_finish - timer_start
    # MCTS_objval = root.objval

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=find_schedule, args=(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, MCTS_objval))
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
            write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, OBJ_FUN, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, GAMMA, GAMMA_DECREASE)

        # plot_schedule(OUTPUT_DIR, schedule, N, M, LV, GV)
        write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch_best_found, MILP_objval, MILP_calctime, MILP_TIMEOUT, MCTS_calctime, MCTS_objval)
        # write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, 0, OBJ_FUN, 0, 0, 0, 0, 1, 0, MILP_objval, MILP_calctime, MILP_TIMEOUT)

    #     return NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients 
    # return NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients

def main():
    N = 5        # number of jobs
    M = 1         # number of work stations
    LV = [3]      # number of resources in each work station
    GV = [6]      # number of units in each resource of each work station
    
    GAMMA = 0.3             # 0.7 for training, 0.3 for loading
    EPSILON = 1.0           # 1.0 (with decrease) for training, 0.0/0.2/0.5 for loading

    GAMMA_DECREASE = 0.3
    EPSILON_DECREASE = 0.025    # 0.025 for training, 0.1 for loading
    weight_decay = 0.00001

    OBJ_FUN = {
        "Cmax": 1,
        "Tsum": 10,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    # [10, 7, 3, 1]
    # [17, 12, 5, 1]
    # [7, 5, 2, 1]
    layer_dims = [17, 12, 5, 1]    # [32, 25, 16, 7, 1]
    INPUT_CONFIGS = ["all_vars", "XV", "minmax_large", "minmax_small", "generalizability", "high", "absolute", "relative", "generalizability_T", "diff", "better", "diff_better"]
    CONFIG = 10      #7

    NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    NN_weights_gradients = np.zeros_like(NN_weights)
    NN_biases_gradients = np.zeros_like(NN_biases)

    PHASE = "train"         # train / load
    METHOD = "NN"           # JEPS / Q_learning / NN

    EPOCHS = 30000
    TIMEOUT = 60            # allowed computation time in sec
    MILP_TIMEOUT = 30*60    # allowed computation time for MILP in sec
    MCTS_BUDGET = 10000

    OUTPUT_DIR = '/home/merel/Documents/studie/IS/thesis/RL_scheduling/output/'

    # file = open(OUTPUT_DIR+"log.csv",'a')
    # file.write("METHOD\tPHASE\tN\tM\tLV\tGV\tCONFIG\tEPOCHS\tGAMMA\tGAMMA_DECREASE\tEPSILON\tEPSILON_DECREASE\tLAYER_DIMS\tWEIGHT_DECAY\tCMAX_WEIGHT\tTSUM_WEIGHT\tMAKESPAN\tTSUM\tTMAX\tTN\tTIME\tEPOCH_BEST\tMILP_OBJVAL\tMILP_CALCTIME\tMILP_TIMEOUT\tMCTS_CALCTIME\tMCTS_OBJVAL")
    # file.close()

    # test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    # NN = NeuralNetwork(
    #     Dense(NN_weights[0], NN_weights_gradients[0], NN_biases[0], NN_biases_gradients[0]), 
    #     Sigmoid(),
    #     Dense(NN_weights[1], NN_weights_gradients[1], NN_biases[1], NN_biases_gradients[1]), 
    #     Sigmoid(),
    #     Dense(NN_weights[2], NN_weights_gradients[2], NN_biases[2], NN_biases_gradients[2]),
    #     Sigmoid(),
    #     Dense(NN_weights[3], NN_weights_gradients[3], NN_biases[3], NN_biases_gradients[3]), 
    #     Sigmoid())
    # loss = NLL()

    # NN = batch_train_NN(NN, loss, OUTPUT_DIR, weight_decay, GAMMA)
    # params = NN.get_params()
    # NN_weights = [params[i] for i in range(0,len(params),2)]
    # NN_biases = [params[i] for i in range(1,len(params),2)]
    # grads = NN.get_params_gradients()
    # NN_weights_gradients = [grads[i] for i in range(0,len(grads),2)]
    # NN_biases_gradients = [grads[i] for i in range(1,len(grads),2)]
    # write_NN_weights(OUTPUT_DIR, M, N, LV, GV, None, layer_dims, OBJ_FUN, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, GAMMA, None)

    N = 5
    for LV in [2, 3, 5]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 7
    for LV in [3, 4, 6]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 9
    for LV in [3, 5, 7]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 11
    for LV in [4, 6, 8]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 13
    for LV in [3, 5, 7, 9]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 15
    for LV in [3, 5, 8, 10]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 17
    for LV in [4, 6, 9, 11]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)

    N = 19
    for LV in [4, 6, 10, 13]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, GAMMA_DECREASE, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)


if __name__ == '__main__':
    main()