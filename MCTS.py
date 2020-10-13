from operator import attrgetter
import copy
from MDP import *

class Node(object):
    def __init__(self, index, parent, schedule, waiting, root, leaf):
        self.index = index
        self.children = []
        self.parent = parent

        self.objval = 99999
        self.schedule = schedule
        self.waiting = waiting

        self.isroot = root
        self.isleaf = leaf

class Monte_Carlo(object):
    # INITIALIZE TREE
    def __init__(self, N, M, LV, GV, due_dates):
        # initialize jobs, resources and units
        self.jobs = [Job(j, M, 0, due_dates) for j in range(N)]
        self.resources = [Resource(0, i, GV) for i in range(LV[0])]

        # initialize root node and current node
        self.root = Node(0, None, Schedule(N, M, LV, GV), self.jobs.copy(), True, False)
        self.node = self.root
        
    # SEARCH FOR OPTIMAL SOLUTION
    def search(self, budget, N, M, LV, GV, deltas, EPSILON, OBJ_FUN):

        for b in range(budget):

            # print(self.root.objval)

            # if current node is a leaf (complete schedule) start backpropagating
            if self.node.isleaf == True:
                objval = self.calc_objval(N, LV, GV, deltas, OBJ_FUN)

		    	# Update lower-bound objective values for all parent nodes
                while self.node.isroot == False:
                    parent = self.node.parent
                    parent.objval = min(parent.objval, self.node.objval)
                    self.node = parent

                self.root = self.node

		    # current node is not a leaf
            else:
		    	# if the current node does not have children yet, expand randomly
                if len(self.node.children) == 0:
                    job = random.sample(self.node.waiting, 1)[0]
                    resource = random.sample(self.resources, 1)[0]

                    # print(job.j)
                    # print(resource.i)
                    
		        	# extend schedule
                    schedule = copy.deepcopy(self.node.schedule.schedule)
                    schedule[0][resource.i].append(job.j)
                    new_schedule = Schedule(N, M, LV, GV)
                    new_schedule.schedule = schedule

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

		        # if the current node already has at least one child
                else:
		        	# with probability epsilon, generate or choose a random node
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
                        self.node = min(self.node.children, key=attrgetter('objval'))

        return self.root

    def calc_objval(self, N, LV, GV, deltas, OBJ_FUN):

		# Compute starting and finishing times of all jobs for this schedule
        for i in range(LV[0]):
            jobs = self.node.schedule.schedule[0][i]
            N_i = len(jobs)

            if N_i > 0:
                j = jobs[0]
                z = 0
                for q in range(GV[0]):
                    self.node.schedule.t_q[0][j][q] = z
                    z += deltas[0][j][q][i]
                    self.node.schedule.c_q[0][j][q] = z
                    z += 1

                if N_i > 1:
                    for q in range(GV[0]):
                        for j_index in range(1, len(jobs)):
                            j = jobs[j_index]

                            self.node.schedule.t_q[0][j][q] = max(self.node.schedule.c_q[0][jobs[j_index-1]][q], self.node.schedule.c_q[0][j][q-1]) + 1
                            self.node.schedule.c_q[0][j][q] = self.node.schedule.t_q[0][j][q] + deltas[0][j][q][i]

		# Calculate objective value for this schedule
        for j in range(N):
            self.node.schedule.t[j] = self.node.schedule.t_q[0][j][0]
            self.node.schedule.c[j] = self.node.schedule.c_q[0][j][GV[0]-1]

            due_date = self.jobs[j].D[-1]
            self.node.schedule.T[j] = max(self.node.schedule.c[j] - due_date, 0)

        self.node.schedule = self.node.schedule.objectives()
        self.node.objval = self.node.schedule.calc_reward(OBJ_FUN)

        return self.node.objval