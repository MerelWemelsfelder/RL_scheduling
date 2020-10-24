from operator import attrgetter
import copy
from MDP import *

class Node(object):
    def __init__(self, index, parent, schedule, waiting, root, leaf):
        self.index = index
        self.children = []
        self.parent = parent

        self.visit_count = 1
        self.objval_min = 99999
        self.objval_avg = 99999
        self.schedule = schedule
        self.waiting = waiting

        self.isroot = root
        self.isleaf = leaf

class Monte_Carlo(object):

    def __init__(self, N, M, LV, GV, due_dates):
        # initialize jobs, resources and units
        self.jobs = [Job(j, M, 0, due_dates) for j in range(N)]
        self.resources = [Resource(0, i, GV) for i in range(LV[0])]

        # initialize root node and current node
        self.root = Node(0, None, Schedule(N, M, LV, GV), self.jobs.copy(), True, False)
        self.node = self.root
        
    def search(self, return_dict, N, M, LV, GV, deltas, EPSILON, OBJ_FUN):

        # EXPANSION OF ROOT NODE
        waiting = self.root.waiting.copy()
        job = random.sample(waiting, 1)[0]
        resource = random.sample(self.resources, 1)[0]

        # extend schedule
        schedule = copy.deepcopy(self.root.schedule.schedule)
        schedule[0][resource.i].append(job.j)
        new_schedule = Schedule(N, M, LV, GV)
        new_schedule.schedule = schedule

        # remove chosen job from waiting jobs
        waiting.remove(job)

        # check whether newly expanded node is a leaf
        isleaf = False
        if len(waiting) == 0:
            isleaf = True
        
        # create new node
        index = len(self.node.children)
        self.node.children.append(Node(index, self.node, new_schedule, waiting, False, isleaf))

        # set new node as current node
        self.node = self.node.children[index]

        while True:

            # SIMULATION

            if self.node.isleaf == True: 
                objval = calc_objval(self.node.schedule, self.jobs, N, LV, GV, deltas, OBJ_FUN)
                new_schedule = self.node.schedule
                self.node.objval_min = objval
                self.node.objval_avg = objval

            else:
                # simulate scheduling until complete
                waiting = self.node.waiting.copy()
                while len(waiting) > 0:
                    job = random.sample(waiting, 1)[0]
                    resource = random.sample(self.resources, 1)[0]
                    schedule[0][resource.i].append(job.j)
                    waiting.remove(job)

                # calculate objective value of new schedule
                new_schedule = Schedule(N, M, LV, GV)
                new_schedule.schedule = schedule
                objval = calc_objval(new_schedule, self.jobs, N, LV, GV, deltas, OBJ_FUN)

                self.node.objval_min = objval
                self.node.objval_avg = objval

            if objval < self.root.objval_min:
                return_dict["schedule"] = new_schedule

            # BACKPROPAGATION

            # update lower-bound and average objective values for all parent nodes
            while self.node.isroot == False:
                parent = self.node.parent
                parent.objval_min = min(parent.objval_min, self.node.objval_min)
                
                objval_tot = (parent.objval_avg * parent.visit_count) + objval
                parent.visit_count += 1
                parent.objval_avg = objval_tot / parent.visit_count
                
                self.node = parent

            # update root node
            self.root = self.node
            return_dict["root"] = self.root.objval_min

            # SELECTION

            while len(self.node.children) > 0:

                if random.uniform(0, 1) < EPSILON:
                    job = random.sample(self.node.waiting, 1)[0]
                    resource = random.sample(self.resources, 1)[0]

                    # extend schedule
                    schedule = copy.deepcopy(self.node.schedule.schedule)
                    schedule[0][resource.i].append(job.j)
                    new_schedule = Schedule(N, M, LV, GV)
                    new_schedule.schedule = schedule

                    # check whether chosen node was already expanded
                    children = [node.schedule.schedule for node in self.node.children]
                    # if it already existed, go to that node
                    if schedule in children:
                        self.node = self.node.children[children.index(schedule)]
                    
                    # EXPANSION
                    # if it does not exist yet, create a new node
                    else: 
                        # remove chosen job from waiting jobs
                        waiting = self.node.waiting.copy()
                        waiting.remove(job)

                        # check whether newly expanded node is a leaf
                        isleaf = False
                        if len(waiting) == 0:
                            isleaf = True
                        
                        # create new node
                        index = len(self.node.children)
                        self.node.children.append(Node(index, self.node, new_schedule, waiting, False, isleaf))

                        # set new node as current node
                        self.node = self.node.children[index]

                # with probability 1-epsilon
                else:
                    # choose existing child node with lowest objective value
                    self.node = min(self.node.children, key=attrgetter('objval_avg'))


def calc_objval(schedule, all_jobs, N, LV, GV, deltas, OBJ_FUN):

    # Compute starting and finishing times of all jobs for this schedule
    for i in range(LV[0]):
        jobs = schedule.schedule[0][i]
        N_i = len(jobs)

        if N_i > 0:
            j = jobs[0]
            z = 0
            for q in range(GV[0]):
                schedule.t_q[0][j][q] = z
                z += deltas[0][j][q][i]
                schedule.c_q[0][j][q] = z
                z += 1

            if N_i > 1:
                for q in range(GV[0]):
                    for j_index in range(1, len(jobs)):
                        j = jobs[j_index]

                        schedule.t_q[0][j][q] = max(schedule.c_q[0][jobs[j_index-1]][q], schedule.c_q[0][j][q-1]) + 1
                        schedule.c_q[0][j][q] = schedule.t_q[0][j][q] + deltas[0][j][q][i]

    # Calculate objective value for this schedule
    for j in range(N):
        schedule.t[j] = schedule.t_q[0][j][0]
        schedule.c[j] = schedule.c_q[0][j][GV[0]-1]

        due_date = all_jobs[j].D[-1]
        schedule.T[j] = max(schedule.c[j] - due_date, 0)

    schedule = schedule.objectives()
    objval = schedule.calc_reward(OBJ_FUN)

    return objval