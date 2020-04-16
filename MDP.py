import numpy as np
import random
from itertools import chain, combinations

from Q_learning import *
from JEPS import *

class Schedule(object):
    def __init__(self, B, D, N, LV, GV):
        self.B = B              # all release dates
        self.D = D              # all due dates
        self.T = np.zeros([N])  # tardiness for all jobs

        self.t = np.zeros([N])          # starting times of jobs
        self.t_q = np.zeros([N, GV])    # starting times of jobs on units
        self.c = np.zeros([N])          # completion times of jobs
        self.c_q = np.zeros([N, GV])    # completion times of jobs on units

        # processed jobs per resource, like: [[4,2], [0,5,3], ..]
        self.schedule = [[] for i in range(LV)]
        for i in range(LV):
            self.schedule[i] = []

    def objectives(self):
        self.Cmax = max(self.c) - min(self.t) # makespan
        self.Tmax = max(self.T)                 # maximum tardiness
        self.Tmean = np.mean(self.T)            # mean tardiness
        self.Tn = sum(T>0 for T in self.T)       # number of tardy jobs

        return self

class Resource(object):
    def __init__(self, i, GV, policy_init):
        self.i = i                                      # index of r_i
        self.units = [Unit(i, q) for q in range(GV)]    # units in resource
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

class Unit(object):
    def __init__(self, i, q):
        self.i = i              # resource index of r_i
        self.q = q              # unit index of u_iq

    def reset(self, waiting):
        self.processing = None  # job that is being processed
        self.c = None           # completion time of current job
        self.c_idle = None      # time of becoming idle after job completion
        
        # state of unit = jobs waiting to be processed on unit
        if self.q == 0:
            self.state = waiting
        else:
            self.state = []
            
        return self

class Job(object):
    def __init__(self, j, D, B):
        self.j = j      # job index
        self.B = B      # release date
        self.D = D      # due date
        
    def reset(self):
        self.done = False       # whether all processing is completed
        
        return self

class MDP(object):
    # INITIALIZE MDP ENVIRONMENT
    def __init__(self, LV, GV, N, policy_init, due_dates, release_dates):
        self.jobs = [Job(j, due_dates[j], release_dates[j]) for j in range(N)]
        self.actions = self.jobs.copy()
        self.actions.append("do_nothing")
        self.states = [list(state) for state in list(powerset(self.jobs))]
        self.resources = [Resource(i, GV, policy_init) for i in range(LV)]
        
    # RESET MDP ENVIRONMENT
    def reset(self, due_dates, release_dates, LV, GV, N):
        self.jobs = [j.reset() for j in self.jobs]
        waiting = self.jobs.copy()
        self.resources = [resource.reset(waiting) for resource in self.resources]
        
        self.schedule = Schedule(release_dates, due_dates, N, LV, GV)
        self.DONE = False

    # TAKE A TIMESTEP
    def step(self, z, GV, N, METHOD, delta, ALPHA, GAMMA, EPSILON, STACT, heur_job, heur_res, heur_order):

        for resource in self.resources:
            resource.reward -= 1                            # timestep penalty
            resource.prev_state = resource.state.copy()     # update previous state

            # REACHED COMPLETION TIMES
            for q in range(GV):
                unit = resource.units[q]
                if unit.c == z:
                    job = unit.processing
                    self.schedule.c_q[job.j][unit.q] = z
                    
                    # If job is finished at the last unit of a resource
                    if q == (GV-1):
                        job.done = True 
                        self.schedule.c[job.j] = z
                        self.schedule.T[job.j] = max(z - self.schedule.D[job.j], 0)
                
                if unit.c_idle == z:
                    job = unit.processing
                    unit.processing = None          # set unit to idle
                    if q < (GV-1):                  # if this is not the last 
                        nxt = resource.units[q+1]   # next unit in the resource
                        nxt.state.append(job)       # add job to waiting list for next unit

        # CHECK WHETHER ALL JOBS ARE FINISHED
        if all([job.done for job in self.jobs]):            
            return self, True
                        
        # RESUME OPERATIONS BETWEEN UNITS
        for resource in self.resources:
            for unit in resource.units[1:]:         # for all units exept q=0
                if len(unit.state) > 0:             # check if there is a waithing list
                    if unit.processing == None:     # check if unit is currently idle
                        job = unit.state.pop(0)                 # pick first waiting job
                        unit.processing = job                   # set unit to processing selected job
                        self.schedule.t_q[job.j][unit.q] = z    # add unit starting time job j

                        completion = z + delta[job.j][unit.q][resource.i]
                        unit.c = completion
                        unit.c_idle = completion + 1
                        resource.schedule.append((job.j,z))

                        
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
                    job = random.sample(actions, 1)[0]                                      # explore
                else:
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

                    self.schedule.t[job.j] = z                    # add starting time job j
                    self.schedule.t_q[job.j][unit.q] = z           # add unit starting time job j
                    self.schedule.schedule[unit.i].append(job.j)    # add operation to schedule

                    completion = z + delta[job.j][unit.q][resource.i]
                    unit.c = completion                     # set completion time on unit
                    unit.c_idle = completion + 1            # set when unit will be idle again
                    resource.schedule.append((job.j,z))     # add to schedule
            else:
                resource.last_action = None

        if METHOD == "Q_learning":
            self.resources = update_policy_Q(self.resources, self.states, self.actions, STACT)
        if METHOD == "JEPS":
            self.resources = update_history(self.resources, z)
            
        return self, False

def powerset(iterable):
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))