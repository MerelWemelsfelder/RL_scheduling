import numpy as np
import random
from itertools import chain, combinations

# from Q_learning import *
from JEPS import *
from NN import *

class Schedule(object):
    def __init__(self, N, M, LV, GV):
        self.T = np.zeros([N])          # tardiness for all jobs

        self.t = np.zeros([N])          # starting times of jobs
        self.c = np.zeros([N])          # completion times of jobs
        self.t_q = []
        self.c_q = []
        for v in range(M):
            self.t_q.append(np.zeros([N, GV[v]])) # starting times of jobs on units
            self.c_q.append(np.zeros([N, GV[v]])) # completion times of jobs on units

        # processed jobs per resource, like: [[4,2], [0,5,3], ..]
        self.schedule = [[] for v in range(M)]
        for v in range(M):
            self.schedule[v] = [[] for i in range(LV[v])]

    def objectives(self):
        self.Cmax = max(self.c) - min(self.t)   # makespan
        self.Tsum = sum(self.T)                 # total tardiness
        self.Tmax = max(self.T)                 # maximum tardiness
        self.Tmean = np.mean(self.T)            # mean tardiness
        self.Tn = sum(T>0 for T in self.T)      # number of tardy jobs

        return self

    def calc_reward(self, OBJ_FUN):
        r = 0
        r += OBJ_FUN["Cmax"] * self.Cmax
        r += OBJ_FUN["Tsum"] * self.Tsum
        r += OBJ_FUN["Tmax"] * self.Tmax
        r += OBJ_FUN["Tmean"] * self.Tmean
        r += OBJ_FUN["Tn"] * self.Tn

        return r

class WorkStation(object):
    def __init__(self, v, LV, GV):
        self.v = v                                                                  # index of w_v
        self.resources = [Resource(v, i, GV) for i in range(LV[v])]       # units in resource
        
    def reset(self, waiting):
        self.resources = [resource.reset(waiting) for resource in self.resources]
        self.jobs_to_come = waiting.copy()

        return self

class Resource(object):
    def __init__(self, v, i, GV):
        self.v = v                                          # index of w_v
        self.i = i                                          # index of r_i
        self.units = [Unit(v, i, q) for q in range(GV[v])]  # units in resource
        # self.policy = policies[v][i]                      # initialize Q-table with zeros
        
    def reset(self, waiting):
        self.units = [unit.reset(waiting) for unit in self.units]
        
        self.prev_state = None      # previous state
        self.state = waiting.copy()        # current state (= waiting jobs)
        self.last_action = None     # action that was executed most recently
        self.last_job = None        # job that was processed most recently
        self.h = dict()
        
        return self

class Unit(object):
    def __init__(self, v, i, q):
        self.v = v
        self.i = i              # resource index of r_i
        self.q = q              # unit index of u_iq

    def reset(self, waiting):
        self.processing = None  # job that is being processed
        self.c = None           # completion time of current job
        self.c_idle = None      # time of becoming idle after job completion
        
        # state of unit = jobs waiting to be processed on unit
        if self.v == 0 and self.q == 0:
            self.state = waiting.copy()
        else:
            self.state = []
            
        return self

class Job(object):
    def __init__(self, j, M, B, due_dates):
        self.j = j      # job index
        self.B = B      # release date
        self.D = []     # due dates
        for v in range(M):
            self.D.append(due_dates[v][j])
        
    def reset(self):
        self.done = False
        
        return self

class MDP(object):
    # INITIALIZE MDP ENVIRONMENT
    def __init__(self, N, M, LV, GV, release_dates, due_dates, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients):
        self.jobs = [Job(j, M, release_dates[j], due_dates) for j in range(N)]
        self.workstations = [WorkStation(v, LV, GV) for v in range(M)]

        self.NN = NeuralNetwork(
            Dense(NN_weights[0], NN_weights_gradients[0], NN_biases[0], NN_biases_gradients[0]), 
            Sigmoid(),
            Dense(NN_weights[1], NN_weights_gradients[1], NN_biases[1], NN_biases_gradients[1]), 
            Sigmoid(),
            Dense(NN_weights[2], NN_weights_gradients[2], NN_biases[2], NN_biases_gradients[2]),
            Sigmoid(),
            Dense(NN_weights[3], NN_weights_gradients[3], NN_biases[3], NN_biases_gradients[3]), 
            Sigmoid())
        self.loss = NLL()
        
    # RESET MDP ENVIRONMENT
    def reset(self, N, M, LV, GV, release_dates, due_dates):
        self.jobs = [j.reset() for j in self.jobs]
        self.workstations = [workstation.reset(self.jobs) for workstation in self.workstations]
        self.schedule = Schedule(N, M, LV, GV)
        self.DONE = False
        self.NN_inputs = []
        self.NN_predictions = []

    # TAKE A TIMESTEP
    def step(self, z, N, M, LV, GV, CONFIG, GAMMA, EPSILON, deltas, heur_job, heur_res, heur_blocking, heur_rev_blocking, PHASE, METHOD):

        for ws in self.workstations:
            for resource in ws.resources:
                # Update previous state for all resources
                resource.prev_state = resource.state.copy()

                # REACHED COMPLETION TIMES
                for q in range(GV[ws.v]):
                    unit = resource.units[q]
                    if unit.c == z:
                        # Add unit completion time to the schedule
                        job = unit.processing
                        self.schedule.c_q[ws.v][job.j][unit.q] = z
                        
                        # If job is finished at the last unit of a resource, of the last work station
                        if q == (GV[ws.v]-1):
                            if ws.v == (M-1):
                                job.done = True 
                                self.schedule.c[job.j] = z
                                self.schedule.T[job.j] += max(z - job.D[-1], 0)
                    
                    # Set unit to idle
                    if unit.c_idle == z:
                        job = unit.processing
                        unit.processing = None
                        # add job to waiting list for the next unit
                        if q < (GV[ws.v]-1):
                            nxt = resource.units[q+1] 
                            nxt.state.append(job)
                        # add job to waiting list for all first units of the next work station
                        else:
                            if ws.v < (M-1):
                                for resource in self.workstations[ws.v+1].resources:
                                    resource.units[0].state.append(job)


        # CHECK WHETHER ALL JOBS ARE FINISHED
        if all([job.done for job in self.jobs]):            
            return self, True
                        
        # RESUME OPERATIONS BETWEEN UNITS
        for ws in self.workstations:
            for resource in ws.resources:
                for unit in resource.units[1:]:         # for all units exept q=0
                    if len(unit.state) > 0:             # check if there is a waithing list
                        if unit.processing == None:     # check if unit is currently idle
                            job = unit.state.pop(0)                     # pick first waiting job
                            unit.processing = job                       # set unit to processing selected job
                            self.schedule.t_q[ws.v][job.j][unit.q] = z  # add unit starting time job j

                            completion = z + deltas[ws.v][job.j][unit.q][resource.i]
                            unit.c = completion
                            unit.c_idle = completion + 1
                        
        # START PROCESSING OF NEW JOBS
        for ws in self.workstations:
            first_units = set([resource.units[0] for resource in ws.resources])
            for unit in first_units:
                resource = ws.resources[unit.i]
                
                if unit.processing == None and len(unit.state) > 0:

                    # choose random action with probability EPSILON,
                    # otherwise choose action with highest policy value
                    if random.uniform(0, 1) < EPSILON:
                        job = random.sample(unit.state, 1)[0]
                    else:
                        a_indices = [job.j for job in unit.state]
                        # a_indices.append(N)

                        if (PHASE == "train") or (METHOD == "NN"):
                            values = []
                            for j in a_indices:
                                inputs = generate_NN_input(N, M, LV, GV, CONFIG, ws, resource, self.jobs, ws.v, resource.i, j, z, heur_job, heur_res, heur_blocking, heur_rev_blocking, deltas)
                                values.append(self.NN.forward(inputs))

                            j = a_indices[np.argmin(values)]

                        # elif (PHASE == "load") and (METHOD == "JEPS"):
                        #     j = a_indices[np.argmax(resource.policy[a_indices])]
                        
                        job = self.jobs[j]

                    inputs = generate_NN_input(N, M, LV, GV, CONFIG, ws, resource, self.jobs, ws.v, resource.i, job.j, z, heur_job, heur_res, heur_blocking, heur_rev_blocking, deltas)
                    self.NN_inputs.append(inputs)
                    self.NN_predictions.append(self.NN.forward(inputs))

                    resource.last_action = job                  # update last executed action
                    resource.last_job = job                 # update last processed job

                    for u in first_units:
                        u.state.remove(job)                 # remove job from all waiting lists
                    unit.processing = job                   # set unit to processing job

                    if job in ws.jobs_to_come:
                        ws.jobs_to_come.remove(job)

                    if ws.v == 0:
                        self.schedule.t[job.j] = z                      # add starting time job j
                    self.schedule.t_q[ws.v][job.j][unit.q] = z            # add unit starting time job j
                    self.schedule.schedule[ws.v][resource.i].append(job.j)    # add operation to schedule

                    completion = z + deltas[ws.v][job.j][unit.q][resource.i]
                    unit.c = completion                     # set completion time on unit
                    unit.c_idle = completion + 1            # set when unit will be idle again
                else:
                    resource.last_action = None

        return self, False

def powerset(iterable):
    return chain.from_iterable(combinations(iterable, r) for r in range(len(iterable)+1))