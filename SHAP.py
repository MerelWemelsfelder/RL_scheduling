import shap
import pickle
import itertools
import math
import copy
import numpy as np
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from NN import *

def number_of_coalitions(S, cutoff):
	l = 1
	nc = len(S)
	ncs = []
	while nc < cutoff:
		ncs.append(True)
		l += 1
		nc = len(list(itertools.combinations(S, l)))
		
	ncs = ncs + ([False] * (len(S)-((l-1)*2)-1))

	for l in range(len(S)-(l-1),len(S)+1):
		ncs.append(True)

	return ncs


def find_shapley_values(OUTPUT_DIR, X_test, y_test, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, zero_weights, nc_cutoff):
	# N = set of all features
	# n = total number of features
	# f = a particular feature
	# S = N\{f}
	# s = coalition of elements in S
	# v_si = output of coalition s including f
	# v_s = output of coalition s without f
	# phi = shapley value of feature f

	n_features = len(X_test[0])
	diff_values = dict()
	diff_avg_values = dict()
	better_avg_values = dict()

	N = list(range(n_features))
	ncs = number_of_coalitions(N[1:], nc_cutoff)

	# Select feature f from all features
	for f in range(n_features):

		S = N.copy()
		S.remove(f)
		S = np.array(S)

		diff_i = 0
		better_i = 0

		# Sample coalitions
		coalitions = []
		for l in range(len(S)):
			if ncs[l]:
				comb = list(itertools.combinations(S, l+1))
				coalitions += [np.array(c) for c in comb]
			else:
				for c in range(nc_cutoff):
					coalitions.append(S[np.random.choice(S.shape[0], (l+1), replace=False)])

		for s in coalitions:

			# Sample 10 random elements from X_test to use for this coalition
			samples = np.random.choice(X_test.shape[0], 10, replace=False)
			X = X_test[samples, :]
			y_true = y_test[samples]

			# Initialize a neural network with value 0 for all weights of the first layer
			NN = NeuralNetwork(
				Dense(zero_weights.copy(), zero_weights.copy(), NN_biases[0], NN_biases_gradients[0]), 
				Sigmoid(),
				Dense(NN_weights[1], NN_weights_gradients[1], NN_biases[1], NN_biases_gradients[1]), 
				Sigmoid(),
				Dense(NN_weights[2], NN_weights_gradients[2], NN_biases[2], NN_biases_gradients[2]),
				Sigmoid(),
				Dense(NN_weights[3], NN_weights_gradients[3], NN_biases[3], NN_biases_gradients[3]), 
				Sigmoid())
			loss = NLL()
			
			# Fill trained weights for features in coalition
			for si in s:
				NN.layers[0].W[si] = NN_weights[0][si]
				NN.layers[0].grad_W[si] = NN_weights_gradients[0][si]

			# Make predictions with coalition
			y_pred_0 = []
			for x in X:
				y_pred_0.append(NN.forward(x))

			# Add weights for feature f
			NN.layers[0].W[f] = NN_weights[0][f]
			NN.layers[0].grad_W[f] = NN_weights_gradients[0][f]

			# Make predictions with coalition + f
			y_pred_1 = []
			for x in X:
				y_pred_1.append(NN.forward(x))

			# IDEA: not only use difference between predictions ex- and including feature f, but also 
			# comparing both results to y_true, and make shapley value illustrate to what extent
			# the value moves either toward or away from correct value (instead of higher/lower)

			# TODO: check whether all y0, y1 and y_test values are positive

			diff = np.average([y1 - y0 for (y1, y0) in zip(y_pred_1, y_pred_0)])
			diff_i += diff

			better = []
			for i in range(len(y_true)):
				y = y_true[i]
				y0 = y_pred_0[i]
				y1 = y_pred_1[i]

				if y0 < y:
					if y1 < y:
						b = y1 - y0
					else:
						b = 2*y - y1 - y0
				else:
					if y < y1:
						b = y0 - y1
					else:
						b = y1 + y0 - 2*y

				better.append(b)

			better_i += np.average(better)

			# shapley_s = ((math.factorial(len(s)) * math.factorial(len(N) - len(s) - 1)) / math.factorial(len(N))) * diff
			# shapley_i += shapley_s

			print("f="+str(f)+", len(s)="+str(len(s))+": "+str(round(diff_i,5))+", "+str(round(better_i,5)))

		diff_avg_i = diff_i / len(coalitions)
		better_avg_i = better_i / len(coalitions)
 
		diff_file = open(OUTPUT_DIR+"batch/diff_values.txt",'a')
		diff_file.write("f="+str(f)+": "+str(np.array(diff_i))+"\n")
		diff_file.close()

		diff_avg_file = open(OUTPUT_DIR+"batch/diff_avg_values.txt",'a')
		diff_avg_file.write("f="+str(f)+": "+str(np.array(diff_avg_i))+"\n")
		diff_avg_file.close()

		better_avg_file = open(OUTPUT_DIR+"batch/better_avg_values.txt",'a')
		better_avg_file.write("f="+str(f)+": "+str(np.array(better_avg_i))+"\n")
		better_avg_file.close()

		diff_values[i] = diff_i
		diff_avg_values[i] = diff_avg_i
		better_avg_values[i] = better_avg_i
		print("total diff of feature "+str(f)+": "+str(round(diff_i,6)))
		print("total average diff of feature "+str(f)+": "+str(round(diff_avg_i,6)))
		print("total average better of feature "+str(f)+": "+str(round(better_avg_i,6)))

	np.save(OUTPUT_DIR+"batch/diff_values", np.array(diff_values))
	np.save(OUTPUT_DIR+"batch/diff_avg_values", np.array(diff_avg_values))
	np.save(OUTPUT_DIR+"batch/better_avg_values", np.array(better_avg_values))


def main():

	# SETTINGS
	layer_dims = [32, 25, 16, 7, 1]
	OUTPUT_DIR = '/home/merel/Documents/studie/IS/thesis/RL_scheduling/output/'
	nc_cutoff = 500

	with open(OUTPUT_DIR+"batch/"+str(layer_dims)+'-weights.pickle','rb') as f:
	    NN_weights = pickle.load(f)
	with open(OUTPUT_DIR+"batch/"+str(layer_dims)+'-biases.pickle','rb') as f:
	    NN_biases = pickle.load(f)
	with open(OUTPUT_DIR+"batch/"+str(layer_dims)+'-weights_grad.pickle','rb') as f:
	    NN_weights_gradients = pickle.load(f)
	with open(OUTPUT_DIR+"batch/"+str(layer_dims)+'-biases_grad.pickle','rb') as f:
	    NN_biases_gradients = pickle.load(f)
	zero_weights = np.zeros([len(NN_weights[0]), len(NN_weights[0][0])])

	# LOAD TRAINING & TEST SET
	# with open(OUTPUT_DIR+"batch/X.txt") as f:
	#     X = f.readlines()
	#     X = [x.strip().split(";") for x in X]
	#     for f in range(0,len(X)):
	#         X[f] = [float(x) for x in X[f]]
	# with open(OUTPUT_DIR+"batch/y.txt") as f:
	# 	y = f.readlines()
	# 	y = [float(y.strip("\n")) for y in y]

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	# np.save(OUTPUT_DIR+"batch/X_train", np.array(X_train))
	# np.save(OUTPUT_DIR+"batch/X_test", np.array(X_test))
	# np.save(OUTPUT_DIR+"batch/y_train", np.array(y_train))
	# np.save(OUTPUT_DIR+"batch/y_test", np.array(y_test))

	X_test = np.load(OUTPUT_DIR+"batch/X_test.npy")
	y_test = np.load(OUTPUT_DIR+"batch/y_test.npy")

	find_shapley_values(OUTPUT_DIR, X_test, y_test, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, zero_weights, nc_cutoff)


if __name__ == '__main__':
    main()


# DIFF: low < 0.1 -- medium < 1.0
# f=0: -0.2931205261448963		- medium
# f=1: 0.06723030314761318		- low
# f=2: 0.32360026494438193		- medium
# f=3: 0.21015336437079843		- medium
# f=4: 0.32406958478207765		- medium
# f=5: -0.26349143496242444		- medium
# f=6: -0.23852620615871778		- medium
# f=7: 0.05512610066739152		- low
# f=8: 0.06796122447449435		- low
# f=9: 0.04454211541899682		- low
# f=10: 0.023927973040587733	- low
# f=11: 0.4001432215424781		- medium
# f=12: 0.30136906298368354		- medium
# f=13: 0.022840039079346598	- low
# f=14: -0.00826167658410708	- low
# f=15: 0.0271523132544655		- low
# f=16: -0.007097320365745913	- low
# f=17: -2.70351892957486
# f=18: -0.44340845453815814	- medium
# f=19: -3.3074700518495113
# f=20: -10.062856537858153
# f=21: -3.930087043684729
# f=22: -9.56936581532314
# f=23: -2.430477731533527
# f=24: -0.3835427350817103		- medium
# f=25: 12.844856430148445
# f=26: -0.5914360170840477		- medium
# f=27: 0.016558266308200446	- low
# f=28: -2.558425573876863
# f=29: -0.6500350990413577		- medium
# f=30: -5.00443777330971
# f=31: -4.9978934983261825



# BETTER: very negative < 1.0e-06 -- negative < 1.0e-05
# f=0: 3.491762813406065e-07
# f=1: -1.4136474006832152e-06	- very negative
# f=2: -2.7440698354224497e-06	- very negative
# f=3: -3.0328280899131398e-06	- very negative
# f=4: -3.2223713277115624e-06	- very negative
# f=5: 4.3936562334861e-07
# f=6: 1.552050558610059e-07
# f=7: -1.6782417746831518e-06	- very negative
# f=8: -1.5116531877682684e-06	- very negative
# f=9: -4.816197094592117e-07	- negative
# f=10: 2.3791868452243596e-08
# f=11: -2.204379917376377e-06	- very negative
# f=12: -2.095880148271648e-06	- very negative
# f=13: 1.0519364136001882e-07
# f=14: 1.4595674892089165e-07
# f=15: -9.382195885752116e-08	- negative
# f=16: 6.519658402582135e-08
# f=17: -1.069337127807254e-05	- very negative
# f=18: -3.5690932355114486e-06	- very negative
# f=19: 8.457576030820249e-06
# f=20: 3.2021027192506124e-05
# f=21: 7.344464697852667e-07
# f=22: 2.6120550162787852e-05
# f=23: -1.354371446650573e-06	- very negative
# f=24: -3.0079487997845064e-06	- very negative
# f=25: 1.5176424959806346e-05
# f=26: 2.444417894995412e-06
# f=27: -2.5993490879295565e-06	- very negative
# f=28: -9.842448319028212e-07	- negative
# f=29: 2.2520569618462554e-06
# f=30: 2.097553355466416e-05
# f=31: 1.8290777027947816e-05

0 5 6 10 13 14 16 26 29
19 20 21 22 25 30 31

17 19 20 21 22 23 25 28 30 31

# ALL FEATURES
# 0 time_res_minmax
# 1 time_res_stdev
# 2 time_res_stdev_occupied
# 3 time_res_stdev_blocking
# 4 time_res_stdev_occ_block
# 5 time_job_minmax
# 6 time_job_minmax_coming
# 7 time_job_stdev_all
# 8 time_job_stdev_coming
# 9 time_job_stdev_blocking_all
# 10 time_job_stdev_blocking_coming
# 11 blocking_res_stdev
# 12 rev_blocking_res_stdev
# 13 blocking_job_all_stdev
# 14 rev_blocking_job_all_stdev
# 15 blocking_job_coming_stdev
# 16 rev_blocking_job_coming_stdev
# 17 blocking
# 18 rev_blocking 
# 19 u0_occupied
# 20 future_blockings_mean
# 21 future_rev_blockings_mean
# 22 future_blockings_median
# 23 future_rev_blockings_median
# 24 already_processing
# 25 T_expected
# 26 time_to_duedate
# 27 relative_duedate
# 28 n
# 29 m
# 30 lv
# 31 gv