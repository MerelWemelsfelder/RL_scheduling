import numpy as np
import random, time
import itertools
import math
import multiprocessing
from numpy import load
import shap

from instances import *
from MDP import *
from JEPS import *
from MILP import *
from NN import *
from utils import *
from MCTS import *

def find_schedule(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, MCTS_objval):

    timer_start = time.time()

    # Generate heuristics for Q_learning rewards
    heur_job = heuristic_best_job(deltas, N, M, LV, GV)
    heur_res = heuristic_best_resource(heur_job, N, M, LV)
    heur_blocking = heuristic_blocking(deltas, N, M, LV, GV)
    heur_rev_blocking = heuristic_reverse_blocking(deltas, N, M, LV, GV)

    # Load stored weights for the policy value function
    if PHASE == "load":
        folder = INPUT_CONFIGS[CONFIG] + "/"
        with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-weights.pickle",'rb') as f:
            NN_weights = pickle.load(f)
        with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-biases.pickle",'rb') as f:
            NN_biases = pickle.load(f)
        with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-weights_grad.pickle",'rb') as f:
            NN_weights_gradients = pickle.load(f)
        with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+"-biases_grad.pickle",'rb') as f:
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

            # if PHASE == "train":
            #     makespan = return_dict["best_schedule"].Cmax
            #     Tsum = return_dict["best_schedule"].Tsum
            #     Tmax = return_dict["best_schedule"].Tmax
            #     Tn = return_dict["best_schedule"].Tn
            #     write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, epoch, OBJ_FUN, makespan, Tsum, Tmax, Tn, 0, return_dict["epoch_best_found"], MILP_objval, 0, 0, 0, MCTS_objval)

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
        y_pred = np.array(RL.NN_predictions)
        
        # Update the weighs of the policy value function
        if PHASE == "train":
            diff_norm = (r - MILP_objval) / max(r, MILP_objval)
            if r >= MILP_objval:
                y_true = 0.5 * (1 - diff_norm) + diff_norm
            else:
                y_true = 0.5 * (1 - diff_norm)
        elif PHASE == "load":
            diff_norm = (r - r_best) / max(r, r_best)
            if r >= r_best:
                y_true = 0.5 * (1 - diff_norm) + diff_norm
            else:
                y_true = 0.5 * (1 - diff_norm)

        # if PHASE == "train":
        #     write_training_batch(OUTPUT_DIR=OUTPUT_DIR, X_train=RL.NN_inputs, y_true=y_true, INPUT_CONFIGS=INPUT_CONFIGS, CONFIG=CONFIG)
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
def test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_TIMEOUT, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED):

    ins = MILP_instance(M, LV, GV, N, RSEED)
    MILP_objval = 0
    MILP_calctime = 0
    MCTS_objval = 0
    MCTS_calctime = 0

    # UNCOMMENT TO RUN MILP
    # timer_start = time.time()
    # MILP_schedule, MILP_objval = MILP_solve(ins, M, LV, GV, N)
    # timer_finish = time.time()
    # MILP_calctime = timer_finish - timer_start
    # write_log(OUTPUT_DIR, "none", N, M, LV, GV, "benchmark", 0, EPSILON, 0, [], 0, METHOD, 0, OBJ_FUN, 0, 0, 0, 0, 0, 0, MILP_objval, MILP_calctime, MILP_TIMEOUT, MCTS_TIMEOUT, 0)

    deltas = []
    due_dates = []
    for v in range(M):
        deltas.append(np.round(ins.lAreaInstances[v].tau))
        due_dates.append(ins.lAreaInstances[v].d)
    release_dates = np.zeros([N])

    # # MONTE CARLO TREE SEARCH
    # EPSILON = 0.5
    # MCTS = Monte_Carlo(N, M, LV, GV, due_dates)
    # return_dict = multiprocessing.Manager().dict()
    # p = multiprocessing.Process(target=MCTS.search, args=(return_dict, N, M, LV, GV, deltas, EPSILON, OBJ_FUN))
    # p.start()
    # p.join(MCTS_TIMEOUT)
    # if p.is_alive():
    #     print("Runtime error: abort mission.\n")
    #     p.terminate()
    #     p.join()

    #     plot_schedule(OUTPUT_DIR, "MCTS", return_dict["schedule"], N, M, LV, GV, EPSILON, MCTS_TIMEOUT)
    #     write_log(OUTPUT_DIR, "none", N, M, LV, GV, "benchmark", 0, EPSILON, 0, [], 0, METHOD, 0, OBJ_FUN, 0, 0, 0, 0, 0, 0, MILP_objval, MILP_calctime, MILP_TIMEOUT, MCTS_TIMEOUT, return_dict["root"])
    
    # AGENT LEARNING
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=find_schedule, args=(return_dict, MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, MCTS_objval))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        print("Runtime error: abort mission.\n")
        p.terminate()
        p.join()

    # return_dict = find_schedule(dict(), MILP_objval, N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, deltas, due_dates, release_dates, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, MCTS_objval)
	    
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

        # if PHASE == "train":
        #     write_NN_weights(OUTPUT_DIR, layer_dims, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, INPUT_CONFIGS, CONFIG)

        # plot_schedule(OUTPUT_DIR, INPUT_CONFIGS[CONFIG], schedule, N, M, LV, GV, EPSILON, TIMEOUT)
        write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, INPUT_CONFIGS[CONFIG], GAMMA, EPSILON, EPSILON_DECREASE, layer_dims, weight_decay, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch_best_found, MILP_objval, MILP_calctime, MILP_TIMEOUT, MCTS_calctime, MCTS_objval)

def main():
    N = 5         # number of jobs
    M = 1         # number of work stations
    LV = [3]      # number of resources in each work station
    GV = [6]      # number of units in each resource of each work station
    
    GAMMA = 0.7             # 0.7 for training, 0.3 for loading
    EPSILON = 0.5           # 1.0 (with decrease) for training, 0.0/0.2/0.5 for loading
    EPSILON_DECREASE = 0.1  # 0.025 for training, 0.1 for loading
    weight_decay = 0.00001

    OBJ_FUN = {
        "Cmax": 1,
        "Tsum": 100,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    INPUT_CONFIGS = ["all_vars", "sim_select", "minmax", "generalizability", "absolute", "relative", "diff", "better", "diff_better"]
    CONFIG = 3
    layer_dims = find_layer_dims(CONFIG)	# change manually when testing network architectures

    NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    NN_weights_gradients = np.zeros_like(NN_weights)
    NN_biases_gradients = np.zeros_like(NN_biases)

    PHASE = "load"         # train / load
    METHOD = "NN"           # JEPS / Q_learning / NN

    EPOCHS = 50000
    TIMEOUT = 0          # allowed computation time in sec
    MILP_TIMEOUT = 0       # allowed computation time for MILP in sec
    MCTS_TIMEOUT = 0		# allowed computation time for MCTS in sec
    RSEED = 10

    OUTPUT_DIR = '/home/merel/Documents/studie/IS/thesis/RL_scheduling/output/'

    # file = open(OUTPUT_DIR+"batch/"+INPUT_CONFIGS[CONFIG]+"/log.csv",'a')
    # file.write("METHOD\tPHASE\tN\tM\tLV\tGV\tCONFIG\tEPOCHS\tGAMMA\tEPSILON\tEPSILON_DECREASE\tLAYER_DIMS\tWEIGHT_DECAY\tCMAX_WEIGHT\tTSUM_WEIGHT\tMAKESPAN\tTSUM\tTMAX\tTN\tTIME\tEPOCH_BEST\tMILP_OBJVAL\tMILP_CALCTIME\tMILP_TIMEOUT\tMCTS_CALCTIME\tMCTS_OBJVAL")
    # file.close()

    # generate_batch_data(INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, METHOD, EPOCHS, MCTS_TIMEOUT, OUTPUT_DIR)
    # batch_training(NN_weights, NN_weights_gradients, NN_biases, NN_biases_gradients, OUTPUT_DIR, layer_dims, weight_decay, GAMMA, INPUT_CONFIGS, CONFIG)   
    feature_set_selection(INPUT_CONFIGS, CONFIG, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_TIMEOUT, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)
    # network_architectures(INPUT_CONFIGS, CONFIG, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_TIMEOUT, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)
    # performance_tests_JSSP(INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_TIMEOUT, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)
    # performance_tests_FFSSP(INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_TIMEOUT, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT)
    

if __name__ == '__main__':
    main()