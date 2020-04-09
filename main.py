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
    for i in range(l_1):
        r_dict = dict()
        for j in range(n):
            ls = []
            for q in range(g_1):
                ls.append((q, random.sample(list(range(1,6)), 1)[0]))
            r_dict[j] = ls
        delta[i] = r_dict
    return delta

def calculate_reward(RL):
    t_all = []
    c_all = []
    T_all = []

    for job in RL.jobs:
        t_all.append(job.t)
        c_all.append(job.c)

        Tj = max(job.c - job.D, 0)                      # tardines of job j
        T_all.append(Tj)

    Cmax = max(c_all) - min(t_all)  # makespan
    Tmax = max(T_all)                                   # maximum tardiness
    Tmean = np.mean(T_all)                                 # mean tardiness
    Tn = sum(T>0 for T in T_all)                        # number of tardy jobs

    return Cmax

def update_policy_JEPS(RL, r_best, time_max, gamma):
    r = calculate_reward(RL)
    resources = RL.resources
    if r < r_best:
        for resource in resources:
            for time in range(time_max-1):
                s = resource.h[time][0]
                a = resource.h[time][1]
                if a != None:
                    print(s)
                    print(a)
                    s_index = RL.states.index(s)     # previous state
                    a_index = RL.actions.index(a)    # taken action
                    for job in s:
                        if job == a:
                            if state_action == "st_act":
                                resource.policy[s_index,a_index] = resource.policy[s_index,a_index] + (gamma * (1 - resource.policy[s_index,a_index]))
                            if state_action == "act":
                                as_indices = []
                                for job in s:
                                    as_indices.append(RL.actions.index(job))
                                resource.policy[a_index] = resource.policy[a_index] + (sum(resource.policy[as_indices]) - resource.policy[a_index])
                        else:
                            if state_action == "st_act":
                                resource.policy[s_index,a_index] = (1 - gamma) * resource.policy[s_index,a_index]
                            if state_action == "act":
                                resource.policy[a_index] = (1 - gamma) * resource.policy[a_index]
    return resources

def main():
    delta = gen_delta(l_1, g_1, n)

    if state_action == "st_act":                      #st_act for state-action pairs, act for only actions
        policy_init = np.zeros([2**n, n+1])     # states, actions
    else:                                       # st_act for state-action pairs, act for only actions
        policy_init = np.zeros([n+1])           # actions

    RL = MDP(l_1, g_1, n, policy_init)          # initialize MDP
    r_best = 99999
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}")

        DONE = False
        time = 0
        RL.reset()
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            world, DONE = RL.step(time, g_1, n, delta, alpha, gamma, epsilon)
            time += 1

        if method == "JEPS":
            RL.resources = update_policy_JEPS(RL, r_best, time, gamma)


    print("\nFINAL SCHEDULE")
    score = 0
    for resource in RL.resources:
        print("resource "+str(resource.i)+":")
        print(resource.schedule)
        score += resource.score

    print("\nscore: "+str(score))
    print("needed timesteps: "+str(time))
    print("makespan: "+str(RL.makespan))


    print("Q-TABLES:")
    for resource in RL.resources:
        print("resource "+str(resource.i)+":")
        print(resource.policy)

if __name__ == '__main__':
    main()