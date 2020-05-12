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
        with open('../output/NN_weights/[4, 9, 11, 6, 4, 1]-15_[6]-1_100-weights.pickle','rb') as f:
            NN_weights = pickle.load(f)
        with open('../output/NN_weights/[4, 9, 11, 6, 4, 1]-15_[6]-1_100-biases.pickle','rb') as f:
            NN_biases = pickle.load(f)
        with open('../output/NN_weights/[4, 9, 11, 6, 4, 1]-15_[6]-1_100-weights_grad.pickle','rb') as f:
            NN_weights_gradients = pickle.load(f)
        with open('../output/NN_weights/[4, 9, 11, 6, 4, 1]-15_[6]-1_100-biases_grad.pickle','rb') as f:
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
    return_dict["best_schedule"] = schedule
    return_dict["epoch_best_found"] = 0

    # All other epochs
    # for epoch in range(1,EPOCHS):
    epoch = 0
    while True:

        if epoch%100==0:
            makespan = schedule.Cmax
            Tsum = schedule.Tsum
            Tmax = schedule.Tmax
            Tn = schedule.Tn
            write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, GAMMA, EPSILON, layer_dims, weight_decay, lr, METHOD, epoch, OBJ_FUN, makespan, Tsum, Tmax, Tn, 0, return_dict["epoch_best_found"], 0, 0)

        # Change instance size for cross-instance training
        if epoch%1000==0:
            params = RL.NN.get_params()
            NN_weights = [params[i] for i in range(0,len(params),2)]
            NN_biases = [params[i] for i in range(1,len(params),2)]
            grads = RL.NN.get_params_gradients()
            NN_weights_gradients = [grads[i] for i in range(0,len(grads),2)]
            NN_biases_gradients = [grads[i] for i in range(1,len(grads),2)]

            N = random.randrange(10,21)
            LV = [random.randrange(5,10)]
            GV = [random.randrange(4,10)]

            ins = MILP_instance(M, LV, GV, N)
            deltas = [np.round(ins.lAreaInstances[0].tau)]
            due_dates = [ins.lAreaInstances[0].d]
            release_dates = np.zeros([N])
            policies = [np.zeros([LV[0], N+1])]

            RL = MDP(N, M, LV, GV, release_dates, due_dates, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, policies)

            heur_job = heuristic_best_job(deltas, N, M, LV, GV)
            heur_res = heuristic_best_resource(heur_job, N, M, LV)
            heur_order = heuristic_order(deltas, N, M, LV, GV)

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

            print([best_params[i] for i in range(0,len(best_params),2)])

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

        if PHASE == "train":
            write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, OBJ_FUN, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients)

        # plot_schedule(OUTPUT_DIR, schedule, N, M, LV, GV)
        # print_schedule(schedule, calc_time, MILP_schedule, MILP_objval, MILP_calctime)
        write_log(OUTPUT_DIR, PHASE, N, M, LV, GV, GAMMA, EPSILON, layer_dims, weight_decay, lr, METHOD, EPOCHS, OBJ_FUN, makespan, Tsum, Tmax, Tn, calc_time, epoch_best_found, MILP_objval, MILP_calctime)

def main():
    N = 15         # number of jobs
    M = 1           # number of work stations
    LV = [6]      # number of resources in each work station
    GV = [6]      # number of units in each resource of each work station
    
    # ALPHA = 0.4   # discount factor (0≤α≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)
    GAMMA = 0.8     # learning rate (0<γ≤1): the extent to which Q-values are updated every timestep / epoch
    EPSILON = 0.2   # probability of choosing a random action (= exploring)

    OBJ_FUN = {
        "Cmax": 1,
        "Tsum": 100,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    layer_dims = [15,6,1]
    NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    NN_weights_gradients = np.zeros_like(NN_weights)
    NN_biases_gradients = np.zeros_like(NN_biases)

    weight_decay = 0.01
    lr = 0.1

    PHASE = "train"     # train / load
    METHOD = "NN"     # JEPS / Q_learning / NN

    EPOCHS = 100000
    TIMEOUT = 30*60
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD\tPHASE\tN\tM\tLV\tGV\tEPOCHS\tGAMMA\tEPSILON\tLAYER_DIMS\tWEIGHT_DECAY\tLR\tCMAX_WEIGHT\tTSUM_WEIGHT\tMAKESPAN\tTSUM\tTMAX\tTN\tTIME\tEPOCH_BEST\tMILP_OBJVAL\tMILP_CALCTIME")
    file.close()

    test(N, M, LV, GV, GAMMA, EPSILON, OBJ_FUN, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT)

    # layer_dims = [12,3,5,21,18]
    #     all_combinations = list(itertools.combinations_with_replacement(layer_dims, 5))
    #     for c in range(len(all_combinations)):
    #         all_combinations[c] = [7] + list(all_combinations[c]) + [1]
    #     for layer_dims in all_combinations:
    #         NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    #         NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    #         NN_weights_gradients = np.zeros_like(NN_weights)
    #         NN_biases_gradients = np.zeros_like(NN_biases)

    # layer_dims_set = set(
    #     [4,9,11,6,4,1],
    #     )

    # for N in [10, 20]:
    #     for LV in range(3,10,2):
    #         for GV in range(3,10,2):
    #             for lr in [x/10 for x in range(1,10)]:

    #                 test(N, M, [LV], [GV], GAMMA, EPSILON, OBJ_FUN, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR, TIMEOUT)
    
if __name__ == '__main__':
    main()