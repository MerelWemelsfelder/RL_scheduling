import numpy as np
import random, time
import itertools
import math

from MDP import *
from JEPS import *
from MILP import *
from NN import *
from utils import *

def find_schedule(N, M, LV, GV, GAMMA, EPSILON, deltas, due_dates, release_dates, R_WEIGHTS, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR):
    
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
    best_params = RL.NN.get_params()
    best_params_grad = RL.NN.get_params_gradients()

    # All other epochs
    for epoch in range(1,EPOCHS):

        # if epoch%100==0:
        #     print(epoch)
        #     makespan = schedule.Cmax
        #     Tsum = schedule.Tsum
        #     Tmax = schedule.Tmax
        #     Tn = schedule.Tn
        #     write_log(OUTPUT_DIR, N, M, LV, GV, GAMMA, EPSILON, layer_dims, NN_weights, NN_biases, METHOD, EPOCHS, makespan, Tsum, Tmax, Tn, "-", epoch_best_found, 26.99, 50.75)
        #     plot_schedule(OUTPUT_DIR, best_schedule, N, M, LV, GV)

        #     NN_weights = [best_params[i] for i in range(0,len(best_params),2)]
        #     NN_biases = [best_params[i] for i in range(1,len(best_params),2)]
        #     NN_weights_gradients = [best_params_grad[i] for i in range(0,len(best_params_grad),2)]
        #     NN_biases_gradients = [best_params_grad[i] for i in range(1,len(best_params_grad),2)]
        #     write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients)

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
            RL.NN = update_NN(model=RL.NN, X_train=np.array(RL.NN_inputs), y_pred=np.array(RL.NN_predictions), weight_decay=weight_decay, lr=lr, loss=RL.loss, r=r, r_best=r_best)
            # RL.NN.backward(np.array(RL.NN_inputs), (r_best-r)/min(r_best,r))

        # If this schedule has the best objective value found so far,
        # update best schedule and makespan, and update policy values for JEPS
        if r < r_best:
            r_best = r
            best_schedule = schedule
            epoch_best_found = epoch
            best_params = RL.NN.get_params()
            best_params_grad = RL.NN.get_params_gradients()

            if (PHASE == "load") and (METHOD == "JEPS"):
                for v in range(M):
                    resources = RL.workstations[v].resources
                    for i in range(LV[v]):
                        resources[i] = update_policy_JEPS(resources[i], RL.actions, z, GAMMA)
                    RL.workstations[v].resources = resources

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    
    return best_schedule, epoch_best_found, calc_time, RL

# Test function, which executes the both the MILP and the NN/JEPS algorithm, and stores all relevant information
def test(N, M, LV, GV, GAMMA, EPSILON, R_WEIGHTS, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR):
    
    ins = MILP_instance(M, LV, GV, N)
    # MILP_objval = 26.99
    # MILP_calctime = 50.75
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

    schedule, epoch, calc_time, RL = find_schedule(N, M, LV, GV, GAMMA, EPSILON, deltas, due_dates, release_dates, R_WEIGHTS, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR)

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
    write_NN_weights(OUTPUT_DIR, M, N, LV, GV, EPSILON, layer_dims, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients)
    write_log(OUTPUT_DIR, N, M, LV, GV, GAMMA, EPSILON, layer_dims, weight_decay, lr, NN_weights, NN_biases, METHOD, EPOCHS, makespan, Tsum, Tmax, Tn, calc_time, epoch, MILP_objval, MILP_calctime)

def main():
    N = 15         # number of jobs
    M = 1           # number of work stations
    LV = [6]      # number of resources in each work station
    GV = [4]      # number of units in each resource of each work station
    
    # ALPHA = 0.4   # discount factor (0≤α≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)
    GAMMA = 0.8     # learning rate (0<γ≤1): the extent to which Q-values are updated every timestep / epoch
    EPSILON = 0.9   # probability of choosing a random action (= exploring)

    R_WEIGHTS = {
        "Cmax": 1,
        "Tsum": 0,
        "Tmax": 0,
        "Tmean": 0,
        "Tn": 0
    }

    layer_dims = [7,11,4,1]
    NN_weights = [np.random.rand(layer_dims[i], layer_dims[i+1]) for i in range(len(layer_dims)-1)]
    NN_biases = [np.zeros(layer_dims[i]) for i in range(1,len(layer_dims))]
    NN_weights_gradients = np.zeros_like(NN_weights)
    NN_biases_gradients = np.zeros_like(NN_biases)

    weight_decay = 0.001
    lr = 0.1

    PHASE = "train"     # train / load
    METHOD = "NN"     # JEPS / Q_learning / NN

    EPOCHS = 5000
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD\tN\tM\tLV\tGV\tEPOCHS\tGAMMA\tEPSILON\tLAYER_DIMS\tWEIGHT_DECAY\tLR\tMAKESPAN\tTSUM\tTMAX\tTN\tTIME\tEPOCH_BEST\tMILP_OBJVAL\tMILP_CALCTIME")
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

    for N in range(5,16):
        for LV in range(4,7):
            for GV in range(3,6):
                test(N, M, [LV], [GV], GAMMA, EPSILON, R_WEIGHTS, layer_dims, weight_decay, lr, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, OUTPUT_DIR)
    
if __name__ == '__main__':
    main()