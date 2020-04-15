import numpy as np
import random
from itertools import chain, combinations
from settings import *
from tester import *

def update_policy_Q(resources, states, actions, STACT):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state
        
        if resource.last_action != None:
            s1 = states.index(resource.state)          # current state
            s0 = states.index(resource.prev_state)     # previous state
            a = actions.index(resource.last_action)    # taken action

            if STACT == "st_act":
                next_max = np.max(resource.policy[s1])    # max q-value of current state
                q_old = resource.policy[s0, a]
                q_new = (1 - ALPHA) * q_old + ALPHA * (resource.reward + GAMMA * next_max)
                resource.policy[s0, a] = q_new
            if STACT == "act":
                next_max = resource.policy[a]             # q-value of current state
                q_old = resource.policy[a]
                q_new = (1 - ALPHA) * q_old + ALPHA * (resource.reward + GAMMA * next_max)
                resource.policy[a] = q_new

        resource.reward = 0     # reset reward

    return resources

def update_history(resources, z):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state

        resource.h[z] = (resource.prev_state, resource.last_action)

    return resources

# RESOURCE
class Resource(object):
    def __init__(self, i, GV, policy_init):
        self.i = i                                      # index of r_i
        self.units = [Unit(i, q) for q in range(GV)]   # units in resource
        self.policy = policy_init.copy()                # initialize Q-table with zeros
        
    def reset(self, waiting):
        self.units = [unit.reset(waiting) for unit in self.units]
        
        self.reward = 0             # counts rewards per timestep
        self.prev_state = None      # previous state
        self.state = waiting        # current state (= waiting jobs)
        self.last_action = None     # action that was executed most recently
        self.last_job = None        # job that was processed most recently
        
        self.schedule = []          # stores tuples (job, starting time) for this resource
        self.h = dict()
        
        return self

# UNIT
class Unit(object):
    def __init__(self, i, q):
        self.i = i              # resource index of r_i
        self.q = q              # unit index of u_iq

    def reset(self, waiting):
        self.processing = None  # job that is being processed
        self.c = None           # completion time of current job
        
        # state of unit = jobs waiting to be processed on unit
        if self.q == 0:
            self.state = waiting
        else:
            self.state = []
            
        return self

# JOB
class Job(object):
    def __init__(self, j):
        self.j = j
        self.B = 1                                          # release date
        self.D = random.sample(list(range(10,30)), 1)[0]    # due date
        
    def reset(self):
        self.done = False       # whether all processing is completed
        self.t = None           # time start processing
        self.c = None           # time completion processing
        
        return self

# MDP
class MDP(object):
    # INITIALIZE MDP ENVIRONMENT
    def __init__(self, LV, GV, N, policy_init):
        self.jobs = [Job(j) for j in range(N)]
        self.actions = self.jobs.copy()
        self.actions.append("do_nothing")
        self.states = [list(state) for state in list(powerset(self.jobs))]
        self.resources = [Resource(i, GV, policy_init) for i in range(LV)]
        
    # RESET MDP ENVIRONMENT
    def reset(self):
        self.jobs = [j.reset() for j in self.jobs]
        waiting = self.jobs.copy()
        self.resources = [resource.reset(waiting) for resource in self.resources]
        
        self.makespan = None
        self.DONE = False

    # TAKE A TIMESTEP
    def step(self, z, GV, N, delta, ALPHA, GAMMA, EPSILON, STACT, heur_job, heur_res, heur_order):

        for resource in self.resources:
            resource.reward -= 1                            # timestep penalty
            resource.prev_state = resource.state.copy()     # update previous state
            
            # REACHED COMPLETION TIMES
            for q in range(GV):
                unit = resource.units[q]
                if unit.c == z:
                    job = unit.processing           # remember what job it was processing
                    unit.processing = None          # set unit to idle
                    
                    if q < (GV-1):                 # if this is not the last 
                        nxt = resource.units[q+1]   # next unit in the resource
                        nxt.state.append(job)       # add job to waiting list for next unit
                    else:
                        job.done = True             # set job to done
                        job.c = z                # save completion time of job

        # CHECK WHETHER ALL JOBS ARE FINISHED
        if all([job.done for job in self.jobs]):
            start = min([job.t for job in self.jobs])
            end = max([job.c for job in self.jobs])
            self.makespan = end - start
            
            return self, True
                        
        # PRESUME OPERATIONS BETWEEN UNITS
        for resource in self.resources:
            for unit in resource.units[1:]:         # for all units exept q=0
                if len(unit.state) > 0:             # check if there is a waithing list
                    if unit.processing == None:     # check if unit is currently idle
                        job = unit.state.pop(0)     # pick first waiting job
                        unit.processing = job       # set unit to processing selected job
                        duration = delta[job.j][unit.q][resource.i]
                        unit.c = z + duration       # set completion time
                        
        # START PROCESSING OF NEW JOBS
        first_units = set([resource.units[0] for resource in self.resources])
        for unit in first_units:
            resource = self.resources[unit.i]
            
            if unit.processing == None:
                waiting = unit.state.copy()           # create action set
                actions = waiting.copy()
                actions.append("do_nothing")          # add option to do nothing
                
                # choose random action with probability EPSILON,
                # otherwise choose action with highest Q-value
                if random.uniform(0, 1) < EPSILON:
                    print("EPSILON")
                    job = random.sample(actions, 1)[0]                                      # explore
                else:
                    print("HIGHEST")
                    s_index = self.states.index(waiting)
                    a_indices = [job.j for job in waiting]
                    a_indices.append(N)
                    if STACT == "st_act":
                        j = a_indices[np.argmax(resource.policy[s_index, a_indices])]       # exploit
                    if STACT == "act":
                        j = a_indices[np.argmax(resource.policy[a_indices])]
                    job = self.actions[j]

                resource.last_action = job                  # update last executed action
                if job != "do_nothing":
                    # give a reward for choosing the job with shortest processing time on resource i
                    best_job = min(heur_job[resource.i], key=heur_job[resource.i].get)
                    if job.j == best_job:
                        resource.reward += 3

                    # give a reward for choosing the job which has its shortest processing time on resource i
                    best_resource = min(heur_res[job.j], key=heur_res[job.j].get)
                    if resource.i == best_resource:
                        resource.reward += 5

                    if resource.last_job != None:
                        # give negative reward for blocking due to scheduling order
                        blocking = heur_order[resource.i][job.j][resource.last_job.j]
                        resource.reward -= blocking

                    resource.last_job = job                 # update last processed job
                    unit.state.remove(job)                  # remove job from all waiting lists
                    unit.processing = job                   # set unit to processing job
                    job.t = z                               # set starting time job
                    duration = delta[job.j][unit.q][resource.i]
                    unit.c = z + duration                   # set completion time on unit
                    resource.schedule.append((job.j,z))     # add to schedule
            else:
                resource.last_action = None
                    
        if METHOD == "Q_learning":
            self.resources = update_policy_Q(self.resources, self.states, self.actions, STACT)
        if METHOD == "JEPS":
            self.resources = update_history(self.resources, z)
            
        return self, False



