import numpy as np
import random
from itertools import chain, combinations
from RL import *
from settings import *

def powerset(iterable):
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))

# GENERATE RANDOM DURATIONS
def gen_delta(l_1, g_1, n):
    delta = dict()
    heur_job = dict()

    for i in range(l_1):
        r_dict = dict()
        heur_j = dict()

        for j in range(n):
            ls = []
            j_total = 0
            for q in range(g_1):
                d = random.sample(list(range(1,6)), 1)[0]
                j_total += d
                ls.append(d)

            r_dict[j] = ls
            heur_j[j] = j_total

        delta[i] = r_dict
        heur_job[i] = heur_j

    # delta = resource: {job: [delta_0, delta_1, ..]}
    return delta, heur_job

def heuristic_best_resource(heur_j):
    heur_r = dict()
    for j in heur_j[0].keys():
        heur_r[j] = dict()
        for r in heur_j.keys():
            heur_r[j][r] = heur_j[r][j]
    return heur_r

def heuristic_order(delta, l_1, g_1, n):
    all_jobs = list(range(n))
    heur_order = dict()             # key = resource i
    for i in range(l_1):
        r_dict = dict()             # key = job j
        for j in range(n):
            j_dict = dict()         # key = job o
            other = all_jobs.copy()
            other.remove(j)
            for o in other:
                counter = 0
                spare = 0
                for q in range(g_1-1):
                    dj = delta[i][j][q+1]
                    do = delta[i][o][q]
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

def update_policy_JEPS(resource, states, actions, r_best, time_max, gamma):
    for time in range(time_max-1):
        s = resource.h[time][0]
        a = resource.h[time][1]
        if a != None:
            # print(s)
            # print(a)
            s_index = states.index(s)     # previous state
            a_index = actions.index(a)    # taken action
            # print(resource.policy[a_index])
            for job in s:
                if job == a:
                    if state_action == "st_act":
                        resource.policy[s_index,a_index] = resource.policy[s_index,a_index] + (gamma * (1 - resource.policy[s_index,a_index]))
                    if state_action == "act":
                        resource.policy[a_index] = resource.policy[a_index] + (gamma * (1 - resource.policy[a_index]))
                else:
                    if state_action == "st_act":
                        resource.policy[s_index,a_index] = (1 - gamma) * resource.policy[s_index,a_index]
                    if state_action == "act":
                        resource.policy[a_index] = (1 - gamma) * resource.policy[a_index]

    # print("resource "+str(resource.i))
    # print(resource.policy)
    return resource

def make_schedule(RL):
    schedule = dict()
    for resource in RL.resources:
        schedule[resource.i] = resource.schedule
    return schedule

def main():
    delta, heur_job = gen_delta(l_1, g_1, n)
    heur_res = heuristic_best_resource(heur_job)
    heur_order = heuristic_order(delta, l_1, g_1, n)

    if state_action == "st_act":                # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([2**n, n+1])     # states, actions
    else:                                       # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([n+1])           # actions

    RL = MDP(l_1, g_1, n, policy_init)          # initialize MDP
    r_best = 99999
    best_schedule = dict()
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}")

        DONE = False
        time = 0
        RL.reset()
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            RL, DONE = RL.step(time, g_1, n, delta, alpha, gamma, epsilon, heur_job, heur_res, heur_order)
            time += 1

        r = calculate_reward(RL)
        if r < r_best:
            r_best = r
            best_schedule = make_schedule(RL)
            print("changed")
            if method == "JEPS":
                resources = RL.resources
                states = RL.states
                actions = RL.actions

                for i in range(len(resources)):
                    resource = update_policy_JEPS(resources[i], states, actions, r_best, time, gamma)
                    RL.resources[i] = resource

    print("\nBEST SCHEDULE")
    print(best_schedule)
    print("makespan = "+str(r_best))

    print("POLICIES:")
    for resource in RL.resources:
        print("resource "+str(resource.i)+":")
        print(resource.policy)

if __name__ == '__main__':
    main()