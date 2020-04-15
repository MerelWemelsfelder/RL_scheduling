import numpy as npf
import random, time
from itertools import chain, combinations
from RL import *
import scipy.stats
from MILP import MILP_instance, MILP_solve

def powerset(iterable):
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))

def heuristic_best_job(tau, LV, GV, N):
    heur_job = dict()

    for i in range(LV):
        heur_j = dict()
        for j in range(N):
            j_total = 0
            for q in range(GV):
                j_total += tau[j][q][i]
            heur_j[j] = j_total
        heur_job[i] = heur_j

    return heur_job

def heuristic_best_resource(heur_j):
    heur_r = dict()
    for j in heur_j[0].keys():
        heur_r[j] = dict()
        for r in heur_j.keys():
            heur_r[j][r] = heur_j[r][j]
    return heur_r

def heuristic_order(delta, LV, GV, N):
    all_jobs = list(range(N))
    heur_order = dict()             # key = resource i
    for i in range(LV):
        r_dict = dict()             # key = job j
        for j in range(N):
            j_dict = dict()         # key = job o
            other = all_jobs.copy()
            other.remove(j)
            for o in other:
                counter = 0
                spare = 0
                for q in range(GV-1):
                    dj = delta[j][q+1][i]
                    do = delta[o][q][i]
                    blocking = dj-do
                    if blocking < 0:
                        spare += blocking
                    if blocking > 0:
                        if spare >= blocking:
                            spare -= blocking
                        else:
                            blocking -= spare
                            counter += blocking
                j_dict[o] = counter
            r_dict[j] = j_dict
        heur_order[i] = r_dict
    return heur_order

def calculate_reward(RL):
    t_all = []
    c_all = []
    T_all = []

    for job in RL.jobs:
        t_all.append(job.t)
        c_all.append(job.c)

        Tj = max(job.c - job.D, 0)                      # tardines of job j
        T_all.append(Tj)

    Cmax = max(c_all) - min(t_all)                      # makespan
    Tmax = max(T_all)                                   # maximum tardiness
    Tmean = np.mean(T_all)                              # mean tardiness
    Tn = sum(T>0 for T in T_all)                        # number of tardy jobs

    return Cmax

def update_policy_JEPS(resource, states, actions, r_best, time_max, GAMMA):
    for z in range(time_max-1):
        s = resource.h[z][0]
        a = resource.h[z][1]
        if a != None:
            s_index = states.index(s)     # previous state
            a_index = actions.index(a)    # taken action
            for job in s:
                if job == a:
                    if STACT == "st_act":
                        resource.policy[s_index,a_index] = resource.policy[s_index,a_index] + (GAMMA * (1 - resource.policy[s_index,a_index]))
                    if STACT == "act":
                        resource.policy[a_index] = resource.policy[a_index] + (GAMMA * (1 - resource.policy[a_index]))
                else:
                    if STACT == "st_act":
                        resource.policy[s_index,a_index] = (1 - GAMMA) * resource.policy[s_index,a_index]
                    if STACT == "act":
                        resource.policy[a_index] = (1 - GAMMA) * resource.policy[a_index]
    return resource

def make_schedule(RL):
    schedule = dict()
    for resource in RL.resources:
        schedule[resource.i] = resource.schedule
    return schedule

def find_schedule(M, LV, GV, N, delta, ALPHA, GAMMA, EPSILON, heur_job, heur_res, heur_order, EPOCHS, METHOD, STACT):
    if STACT == "st_act":                       # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([2**N, N+1])     # states, actions
    if STACT == "act":                          # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([N+1])           # actions

    RL = MDP(LV, GV, N, policy_init)            # initialize MDP
    r_best = 99999
    best_schedule = dict()
    epoch_best_found = 0
    timer_start = time.time()
    for epoch in range(EPOCHS):
        if epoch%100==0:
            print(epoch)

        DONE = False
        z = 0
        RL.reset()
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(z, GV, N, delta, ALPHA, GAMMA, EPSILON, STACT, heur_job, heur_res, heur_order)
            z += 1

        r = calculate_reward(RL)
        if r < r_best:
            r_best = r
            best_schedule = make_schedule(RL)
            epoch_best_found = epoch

            if METHOD == "JEPS":
                resources = RL.resources
                states = RL.states
                actions = RL.actions

                for i in range(len(resources)):
                    resource = update_policy_JEPS(resources[i], states, actions, r_best, z, GAMMA)
                    RL.resources[i] = resource

    timer_finish = time.time()
    calc_time = timer_finish - timer_start
    return r_best, best_schedule, epoch_best_found, calc_time, RL

def write_log(OUTPUT_DIR, METHOD, STACT, N, LV, GV, EPOCHS, ALPHA, GAMMA, EPSILON, makespan, calc_time, epoch):
    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("\n"+METHOD+","+STACT+","+str(N)+","+str(LV)+","+str(GV)+","+str(EPOCHS)+","+str(ALPHA)+","+str(GAMMA)+","+str(EPSILON)+","+str(makespan)+","+str(calc_time)+","+str(epoch))    
    file.close() 

def main():
    M = 1       # number of work stations
    LV = 4      # number of resources
    GV = 3      # number of units per resource
    N = 10      # number of jobs

    ALPHA = 0.2     # learning rate (0<α≤1): the extent to which Q-values are updated every timestep
    GAMMA = 0.7     # discount factor (0≤γ≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)   
    EPSILON = 0.2   # probability of choosing a random action (= exploring)

    METHOD = "JEPS"
    STACT = "act"

    EPOCHS = 1000       # set number of epochs to train RL model
    OUTPUT_DIR = '../output/'

    file = open(OUTPUT_DIR+"log.csv",'a')
    file.write("METHOD,STACT,N,LV,GV,EPOCHS,ALPHA,GAMMA,EPSILON,MAKESPAN,TIME,EPOCH_BEST")
    file.close() 

    print("START TESTING")
    # for LV in range(1,10):                  # number of resources
    #     for GV in range(1,5):               # number of units per resource
    #         for N in range(1,100):          # number of jobs
    ins = MILP_instance(M, LV, GV, N)
    # best_schedule, best_makespan = MILP_solve(M, LV, GV, N)
    # print(best_schedule, best_makespan)
    delta = np.round(ins.lAreaInstances[0].tau)
    print(delta)

    heur_job = heuristic_best_job(delta, LV, GV, N)
    heur_res = heuristic_best_resource(heur_job)
    heur_order = heuristic_order(delta, LV, GV, N)

    # print(str(LV)+","+str(GV)+","+str(N)+","+str(EPSILON)+","+str(GAMMA))

    makespan, schedule, epoch, calc_time, RL = find_schedule(M, LV, GV, N, delta, ALPHA, GAMMA, EPSILON, heur_job, heur_res, heur_order, EPOCHS, METHOD, STACT)
    print(schedule)
    print(makespan)
    print(calc_time)
    write_log(OUTPUT_DIR, METHOD, STACT, N, LV, GV, EPOCHS, ALPHA, GAMMA, EPSILON, makespan, calc_time, epoch)

if __name__ == '__main__':
    main()