import numpy as np
import random
from itertools import chain, combinations

# from Q_learning import *
from JEPS import *
from NN import *

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
        self.Cmax = max(self.c) - min(self.t)   # makespan
        self.Tsum = sum(self.T)                 # total tardiness
        self.Tmax = max(self.T)                 # maximum tardiness
        self.Tmean = np.mean(self.T)            # mean tardiness
        self.Tn = sum(T>0 for T in self.T)      # number of tardy jobs

        return self

    def calc_reward(self, R_WEIGHTS):
        r = 0
        r += R_WEIGHTS["Cmax"] * self.Cmax
        r += R_WEIGHTS["Tsum"] * self.Tsum
        r += R_WEIGHTS["Tmax"] * self.Tmax
        r += R_WEIGHTS["Tmean"] * self.Tmean
        r += R_WEIGHTS["Tn"] * self.Tn

        return r

class Resource(object):
    def __init__(self, i, GV, policies):
        self.i = i                                      # index of r_i
        self.units = [Unit(i, q) for q in range(GV)]    # units in resource
        self.policy = policies[i]                       # initialize Q-table with zeros
        
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
        self.done = False
        
        return self

class MDP(object):
    # INITIALIZE MDP ENVIRONMENT
    def __init__(self, N, LV, GV, release_dates, due_dates, NN_weights, policies):
        self.jobs = [Job(j, due_dates[j], release_dates[j]) for j in range(N)]
        self.actions = self.jobs.copy()
        self.actions.append("do_nothing")
        self.states = [list(state) for state in list(powerset(self.jobs))]
        self.resources = [Resource(i, GV, policies) for i in range(LV)]
        self.policy_function = NeuralNetwork(NN_weights)
        
    # RESET MDP ENVIRONMENT
    def reset(self, N, LV, GV, release_dates, due_dates):
        self.jobs = [j.reset() for j in self.jobs]
        waiting = self.jobs.copy()
        self.resources = [resource.reset(waiting) for resource in self.resources]
        
        self.NN_inputs = []
        self.NN_predictions = []
        self.schedule = Schedule(release_dates, due_dates, N, LV, GV)
        self.DONE = False

    # TAKE A TIMESTEP
    def step(self, z, N, LV, GV, GAMMA, EPSILON, delta, heur_job, heur_res, heur_order, PHASE, METHOD):

        for resource in self.resources:
            # Update previous state for all resources
            resource.prev_state = resource.state.copy()

            # REACHED COMPLETION TIMES
            for q in range(GV):
                unit = resource.units[q]
                if unit.c == z:
                    # Add unit completion time to the schedule
                    job = unit.processing
                    self.schedule.c_q[job.j][unit.q] = z
                    
                    # If job is finished at the last unit of a resource
                    if q == (GV-1):
                        job.done = True 
                        self.schedule.c[job.j] = z
                        self.schedule.T[job.j] = max(z - self.schedule.D[job.j], 0)
                
                # Set unit to idle and add job to waiting list for the next unit
                if unit.c_idle == z:
                    job = unit.processing
                    unit.processing = None
                    if q < (GV-1):
                        nxt = resource.units[q+1] 
                        nxt.state.append(job)

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
                waiting = unit.state.copy()
                actions = waiting.copy()
                actions.append("do_nothing")
                
                # choose random action with probability EPSILON,
                # otherwise choose action with highest policy value
                if random.uniform(0, 1) < EPSILON:
                    job = random.sample(actions, 1)[0]
                else:
                    s_index = self.states.index(waiting)
                    a_indices = [job.j for job in waiting]
                    a_indices.append(N)

                    if (PHASE == "train") or (METHOD == "NN"):
                        values = []
                        for j in a_indices[:-1]:
                            inputs = generate_NN_input(resource.i, j, self.actions[j].D, resource.last_job, z, heur_job, heur_res, heur_order, N, LV, GV)
                            values.append(self.policy_function.predict(inputs))
                        inputs = generate_NN_input(0, N, 0, 0, z, heur_job, heur_res, heur_order, N, LV, GV)
                        values.append(self.policy_function.predict(inputs))

                        j = a_indices[np.argmax(values)]
                    elif (PHASE == "load") and (METHOD == "JEPS"):
                        j = a_indices[np.argmax(resource.policy[a_indices])]
                    
                    job = self.actions[j]

                resource.last_action = job                  # update last executed action
                if job != "do_nothing":
                    resource.last_job = job                 # update last processed job
                    unit.state.remove(job)                  # remove job from all waiting lists
                    unit.processing = job                   # set unit to processing job

                    self.schedule.t[job.j] = z                      # add starting time job j
                    self.schedule.t_q[job.j][unit.q] = z            # add unit starting time job j
                    self.schedule.schedule[unit.i].append(job.j)    # add operation to schedule

                    completion = z + delta[job.j][unit.q][resource.i]
                    unit.c = completion                     # set completion time on unit
                    unit.c_idle = completion + 1            # set when unit will be idle again
                    resource.schedule.append((job.j,z))     # add to schedule
            else:
                resource.last_action = None

        if (PHASE == "load") and (METHOD == "JEPS"):
            self.resources = update_history(self.resources, z)

        return self, False

def powerset(iterable):
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))