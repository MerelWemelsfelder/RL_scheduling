import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/merel/Documents/studie/IS/thesis/Scheduling under Uncertainty')

import numpy as np
import instance_functions
import solve_techniques as st
import schedule_tools
from plotting_functions import plot_best_schedule

def MILP_instance(M, LV, GV, N):
	plantLayout = {'q': M,
	               'lm': [LV],
	               'lU': [GV],
	               'probsPlant': [],
	               'probsBatch': [1]}
	n = N
	lR = np.zeros((plantLayout['q'], n))
	ins = instance_functions.Instance(n=n, randomSeed=2, lR=lR, **plantLayout)

	return ins


def MILP_solve(M, LV, GV, N):
	ins = MILP_instance(M, LV, GV, N)
	model, decVars = st.solve_MILP(ins, startSol=None, saveModel=True)	# HIERHEEN
	modelTight, decVarsTight = plot_best_schedule(ins, model, decVars, tightStartTimes=True)
	schedule = schedule_tools.Schedule(ins, decVarsTight)
	objVal = model.objVal

	# evaluate schedule with simulation (sim)
	objValSim, s = schedule.evaluate(returnStartValues=True)

	if (objVal - objValSim)/objValSim > 0.00001:
	    print('Warning: MILP and schedule simulation give different objective'
	          ' values')

	(s, p, delta, gamma, WT, fMax, FMax, f) = decVars
	print(s[0])
	print(f[0])
	
	return schedule, objVal



	################## PRINT INFORMATION ################

	# in the output the "Explored .. nodes (.. simplex iterations) in .. seconds" tells computation time

	# translation JEPS -> MILP
	# M -> q, v -> a
	# LV -> lm, i -> j
	# GV -> lU, q -> ?
	# N -> n

	# processing times in np.array of dimension [N, GV, LV]
	# print(ins.lAreaInstances[0].tau)

	# see descriptions below			
	# print(ins.lAreaInstances[0].machineSettings)
	# print(ins.lAreaInstances[0].machineSettingSwitchTime)
	# print(ins.lAreaInstances[0].t)
	# print(ins.lAreaInstances[0].tWait)
	# print(ins.lAreaInstances[0].t_Tr)

	# (s, p, delta, gamma, WT, fMax, FMax, f) = decVars
	# print(s[0])
	# print(f[0])

	# s_{aiu} = start time batch i at unit u of area a
	# p_{aij} = 1 if batch i is produced on production line j at area a, 0 else
	# delta_{aik} = 1 if batch i starts before k at area a, 0 else
	# weighted tardiness (WT)
	# fMax[a][j] = makespan of production line j at area a
	# FMax = total makespan decision variable
	# f_{aiu} = finishing time batch i on unit u at area a



	# for M in range(3):
	# 	for LV in range(100):
	# 		for GV in range(20):
	# 			for N in range(50):
	# 				plantLayout = {'q': M, 'lm': [LV], 'lU': [GV], 'probsPlant': [], 'probsBatch': [1]}
	# 				n = N
	# 				lR = np.zeros((plantLayout['q'], n))
	# 				ins = instance_functions.Instance(n=n, randomSeed=2, lR=lR, **plantLayout)

	# 				print(ins.lAreaInstances[M].tau)

	# print processing times for all jobs on all units
	# for a in range(ins.q):
	#   for j in range(ins.lm[a]):
	#     for u in range(ins.lU[a]):
	#       for i in range(ins.n):
	#         print("v: "+str(a)+", i: "+str(j)+", q: "+str(u)+", j: "+str(i)+", D_j: "+str(ins.d[i]))
	#         print("duration: "+str(ins.lAreaInstances[a].tau[i, u, j]))

	# ins.m = m  # number of GMLs
	# ins.U = U  # number of units per GML

	# # batches
	# ins.n = n  # number of batches
	# ins.M = M if M is not None else self.generate_feasible_prod_lines(modus='random')  # list of feasible GMLs sets for batches
	# ins.complementM = list(map(lambda M: list(set(range(self.m)).difference(M)), self.M))  # list of infeasible GMLs lists for batches
	# ins.tauMean = self.load_tau_mean()  # expectations of the batches at the units
	# ins.tauVar = self.load_tau_var()  # variance of the batches at the units
	# ins.tauLoc = self.load_tau_loc()  # location of distribution of the batches at the units
	# ins.tauDistribution = self.load_tau_distribution()  # the distribution of the batches
	# ins.tau = self.set_tau('mean', inplace=False)  # the production duration of the batches
	# ins.t_Tr = self.load_t_Tr()  # internal transport times @ GML units
	# ins.machineSettings = self.load_machine_settings()  # machine settings required for batches
	# ins.machineSettingSwitchTime = self.load_machine_setting_switch_time()  # machine setting switching times
	# ins.t, self.tWait = self.load_t()  # switching times @ GML units
	# ins.d = self.load_due_dates()  # due dates of the batches
	# ins.w = self.load_due_date_weights()  # importance of batches
	# ins.T = np.nansum(self.tau) + np.nansum(self.t_Tr) + np.nansum(self.t) + np.nansum(self.tWait)  # large modeling constant
