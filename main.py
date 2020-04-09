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

def main():
    if method == "Q_learning":
        policy_init = np.zeros([2**n, n+1])
    delta = gen_delta(l_1, g_1, n)

    # REINFORCEMENT LEARNING
    RL = MDP(l_1, g_1, n, policy_init)	# initialize MDP
    n_epochs = 100000		# set number of epochs to train RL model

    for epoch in range(n_epochs):
        DONE = False
        time = 0
        RL.reset()
        
        # take timesteps until processing of all jobs is finished
        while not DONE:
            time += 1
            world, DONE = RL.step(time, g_1, n, delta, alpha, gamma, epsilon)
        
        if epoch % 100 == 0:
            # clear_output(wait=True)
            print(f"Epoch: {epoch}")

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
        print(resource.q_table)

if __name__ == '__main__':
    main()