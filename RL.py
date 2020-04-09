import numpy as np
import random
from itertools import chain, combinations
from settings import *
from main import *

def update_policy_Q(resources, states, actions):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state
        resource.score += resource.reward                   # update resource score
        
        if resource.last_action != None:
            s1 = states.index(resource.state)          # current state
            s0 = states.index(resource.prev_state)     # previous state
            a = actions.index(resource.last_action)    # taken action

            if state_action == "st_act":
                next_max = np.max(resource.policy[s1])    # max q-value of current state
                q_old = resource.policy[s0, a]
                q_new = (1 - alpha) * q_old + alpha * (resource.reward + gamma * next_max)
                resource.policy[s0, a] = q_new
            if state_action == "act":
                next_max = resource.policy[a]             # q-value of current state
                q_old = resource.policy[a]
                q_new = (1 - alpha) * q_old + alpha * (resource.reward + gamma * next_max)
                resource.policy[a] = q_new

    return resources

def update_history(resources, time):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state
        resource.score += resource.reward                   # update resource score

        resource.h[time] = (resource.prev_state, resource.last_action)

    return resources

# RESOURCE
class Resource(object):
    def __init__(self, i, g_v, policy_init):
        self.i = i                                      # index of r_i
        self.units = [Unit(i, q) for q in range(g_v)]   # units in resource
        self.policy = policy_init.copy()                # initialize Q-table with zeros
        
    def reset(self, waiting):
        self.units = [unit.reset(waiting) for unit in self.units]
        
        self.reward = 0             # counts rewards per timestep
        self.score = 0              # counts rewards per epoch
        self.prev_state = None      # previous state
        self.state = waiting        # current state (= waiting jobs)
        self.last_action = None     # action that was executed most recently
        
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
    def __init__(self, l_v, g_v, n, policy_init):
        self.jobs = [Job(j) for j in range(n)]
        self.actions = self.jobs.copy()
        self.actions.append("None")
        self.states = [list(state) for state in list(powerset(self.jobs))]
        self.resources = [Resource(i, g_v, policy_init) for i in range(l_v)]
        
    # RESET MDP ENVIRONMENT
    def reset(self):
        self.jobs = [j.reset() for j in self.jobs]
        waiting = self.jobs.copy()
        self.resources = [resource.reset(waiting) for resource in self.resources]
        
        self.makespan = None
        self.DONE = False

    # TAKE A TIMESTEP
    def step(self, time, g_v, n, delta, alpha, gamma, epsilon):
        
        for resource in self.resources:
            resource.reward = -1                            # timestep penalty
            resource.prev_state = resource.state.copy()     # update previous state
            
            # REACHED COMPLETION TIMES
            for q in range(g_v):
                unit = resource.units[q]
                if unit.c == time:
                    job = unit.processing           # remember what job it was processing
                    unit.processing = None          # set unit to idle
                    resource.reward += 1            # give reward for finished processing step
                    
                    if q < (g_v-1):                 # if this is not the last 
                        nxt = resource.units[q+1]   # next unit in the resource
                        nxt.state.append(job)       # add job to waiting list for next unit
                    else:
                        job.done = True             # set job to done
                        job.c = time                # save completion time of job
                        
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
                        duration = delta[resource.i][job.j][unit.q][1]
                        unit.c = time + duration    # set completion time
                        
        # START PROCESSING OF NEW JOBS
        first_units = set([resource.units[0] for resource in self.resources])
        for unit in first_units:
            resource = self.resources[unit.i]
            
            if unit.processing == None:
                waiting = unit.state.copy()           # create action set
                actions = waiting.copy()
                actions.append("None")                # add option to do nothing
                
                # choose random action with probability epsilon,
                # otherwise choose action with highest Q-value
                if random.uniform(0, 1) < epsilon:
                    job = random.sample(actions, 1)[0]          # explore
                else:
                    s_index = self.states.index(waiting)
                    a_indices = [job.j for job in waiting]
                    a_indices.append(n)
                    if state_action == "st_act":
                        j = a_indices[np.argmax(resource.policy[s_index, a_indices])]       # exploit
                    if state_action == "act":
                        j = a_indices[np.argmax(resource.policy[a_indices])]
                    job = self.actions[j]

                resource.last_action = job
                                
                if job != "None":
                    unit.state.remove(job)                  # remove job from all waiting lists
                    unit.processing = job                   # set unit to processing job
                    job.t = time                            # set starting time job
                    duration = delta[unit.i][job.j][unit.q][1]
                    unit.c = time + duration                # set completion time on unit
                    resource.schedule.append((job.j,time))  # add to schedule
            else:
                resource.last_action = None
                    
        if method == "Q_learning":
            self.resources = update_policy_Q(self.resources, self.states, self.actions)
        if method == "JEPS":
            self.resources = update_history(self.resources, time)
            
        return self, False



