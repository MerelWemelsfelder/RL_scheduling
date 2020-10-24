import shap
import pickle
import itertools
import math
import copy
import numpy as np
from itertools import chain, combinations
from sklearn.model_selection import train_test_split
from NN import *
from utils import *

def number_of_coalitions(S, cutoff):
	l = 1
	nc = len(S)
	ncs = []
	while (nc < cutoff) and (nc > 0):
		ncs.append(True)
		l += 1
		nc = len(list(itertools.combinations(S, l)))
		
	if l < len(S):
		ncs = ncs + ([False] * (len(S)-((l-1)*2)-1))

		for l in range(len(S)-(l-1),len(S)+1):
			ncs.append(True)

	return ncs


def find_shapley_values(OUTPUT_DIR, folder, X_test, y_test, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, zero_weights, coal_cutoff):
	# N = set of all features
	# n = total number of features
	# f = a particular feature
	# S = N\{f}
	# c = coalition of elements in S
	# v_si = output of coalition c including f
	# v_s = output of coalition c without f
	# phi = shapley value of feature f

	n_features = len(X_test[0])
	# diff_values = dict()
	# diff_avg_values = dict()
	# better_avg_values = dict()

	N = list(range(n_features))
	ncs = number_of_coalitions(N[1:], coal_cutoff)

	# Select feature f from all features
	for f in N:

		S = N.copy()
		S.remove(f)
		S = np.array(S)

		diff_f = 0
		better_f = 0

		# Sample coalitions
		coalitions = []
		for l in range(len(S)):
			if ncs[l]:
				comb = list(itertools.combinations(S, l+1))
				coalitions += [np.array(c) for c in comb]
			else:
				for c in range(coal_cutoff):
					coalitions.append(S[np.random.choice(S.shape[0], (l+1), replace=False)])

		for c in coalitions:

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
				Sigmoid())
			loss = NLL()
			
			# Fill trained weights for features in coalition
			for ci in c:
				NN.layers[0].W[ci] = NN_weights[0][ci]
				NN.layers[0].grad_W[ci] = NN_weights_gradients[0][ci]

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

			diff = np.average([y1 - y0 for (y1, y0) in zip(y_pred_1, y_pred_0)])
			diff_f += diff

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

			better_f += np.average(better)

			# shapley_s = ((math.factorial(len(c)) * math.factorial(len(N) - len(c) - 1)) / math.factorial(len(N))) * diff
			# shapley_f += shapley_s

			print("f="+str(f)+", len(c)="+str(len(c))+": "+str(round(diff_f,5))+", "+str(round(better_f,5)))

		diff_avg_f = diff_f / len(coalitions)
		better_avg_f = better_f / len(coalitions)
 
		diff_file = open(OUTPUT_DIR+"batch/"+folder+"diff_values.txt",'a')
		diff_file.write("f="+str(f)+": "+str(np.array(diff_f))+"\n")
		diff_file.close()

		diff_avg_file = open(OUTPUT_DIR+"batch/"+folder+"diff_avg_values.txt",'a')
		diff_avg_file.write("f="+str(f)+": "+str(np.array(diff_avg_f))+"\n")
		diff_avg_file.close()

		diff_file = open(OUTPUT_DIR+"batch/"+folder+"better_values.txt",'a')
		diff_file.write("f="+str(f)+": "+str(np.array(better_f))+"\n")
		diff_file.close()

		better_avg_file = open(OUTPUT_DIR+"batch/"+folder+"better_avg_values.txt",'a')
		better_avg_file.write("f="+str(f)+": "+str(np.array(better_avg_f))+"\n")
		better_avg_file.close()

		# diff_values[f] = diff_f
		# diff_avg_values[f] = diff_avg_f
		# better_avg_values[f] = better_avg_f
		print("total diff of feature "+str(f)+": "+str(round(diff_f,6)))
		print("total average diff of feature "+str(f)+": "+str(round(diff_avg_f,6)))
		print("total average better of feature "+str(f)+": "+str(round(better_avg_f,6)))

	# np.save(OUTPUT_DIR+"batch/"+folder+"diff_values", np.array(diff_values))
	# np.save(OUTPUT_DIR+"batch/"+folder+"diff_avg_values", np.array(diff_avg_values))
	# np.save(OUTPUT_DIR+"batch/"+folder+"better_avg_values", np.array(better_avg_values))


def main():

	# SETTINGS
	OUTPUT_DIR = '/home/merel/Documents/studie/IS/thesis/RL_scheduling/output/'
	coal_cutoff = 500

	INPUT_CONFIGS = ["all_vars", "sim_select", "minmax", "generalizability", "absolute", "relative", "diff", "better", "diff_better"]
	CONFIG = 0
	layer_dims = find_layer_dims(CONFIG)
	folder = INPUT_CONFIGS[CONFIG]+"/"

	with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+'-weights.pickle','rb') as f:
	    NN_weights = pickle.load(f)
	with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+'-biases.pickle','rb') as f:
	    NN_biases = pickle.load(f)
	with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+'-weights_grad.pickle','rb') as f:
	    NN_weights_gradients = pickle.load(f)
	with open(OUTPUT_DIR+"batch/"+folder+str(layer_dims)+'-biases_grad.pickle','rb') as f:
	    NN_biases_gradients = pickle.load(f)
	zero_weights = np.zeros([len(NN_weights[0]), len(NN_weights[0][0])])

	X_test = np.load(OUTPUT_DIR+"batch/"+folder+"X_test.npy")
	y_test = np.load(OUTPUT_DIR+"batch/"+folder+"y_test.npy")

	find_shapley_values(OUTPUT_DIR, folder, X_test, y_test, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, zero_weights, coal_cutoff)


if __name__ == '__main__':
    main()


# DIFF: 2,3,4,11,12,17,18,19,20,21,22,23,25,26,28,29,30,31
# BETTER: 1,2,3,4,7,8,10,11,12,25
# DIFF_BETTER: 2,3,4,11,12,25

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