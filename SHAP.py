import shap
import pickle
import itertools
import math
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from NN import *

def find_shapley_values(OUTPUT_DIR, X_test, y_test, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, zero_weights):
	# N = set of all features
	# n = total number of features
	# i = a particular feature
	# S = N\{i}
	# s = coalition of elements in S
	# v_si = output of coalition s including i
	# v_s = output of coalition s without i
	# phi = shapley value of feature i

	n_features = len(X_test[0])
	shapley_values = dict()
	diff_values = dict()

	# Select feature i from all features
	for i in range(n_features):
		print("i: "+str(i))

		N = list(range(n_features))
		n = len(N)
		S = N.copy()
		S.remove(i)

		shapley_i = 0
		diff_i = 0
		# Loop over all possible feature coalitions not including i
		for l in range(1,len(S)+1):
			print("coalition length: "+str(l)+"/"+str(len(S)+1))
			for s in itertools.combinations(S, l):
				s = list(s)

				# Sample 1000 random elements from X_test
				X = X_test[np.random.choice(X_test.shape[0], 1000, replace=False), :]

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

				# print(NN.layers[0].W)

				# Make predictions with coalition
				y_pred_0 = []
				for x in X:
					y_pred_0.append(NN.forward(x))

				# Add weights for feature i
				NN.layers[0].W[i] = NN_weights[0][i]
				NN.layers[0].grad_W[i] = NN_weights_gradients[0][i]

				# print(NN.layers[0].W)

				# Make predictions with coalition + i
				y_pred_1 = []
				for x in X:
					y_pred_1.append(NN.forward(x))

				# IDEA: not only use difference between predictions ex- and including feature i, but also 
				# comparing both results to y_true, and make shapley value illustrate to what extent
				# the value moves either toward or away from correct value (instead of higher/lower)

				diff = np.average([y1 - y0 for (y1, y0) in zip(y_pred_1, y_pred_0)])
				diff_i += diff
				shapley_s = ((math.factorial(len(s)) * math.factorial(n - len(s) - 1)) / math.factorial(n)) * diff
				shapley_i += shapley_s

				print("shapley of coalition "+str(s)+": "+str(round(shapley_s,6))+" (diff: "+str(round(diff,6))+")")

			diff_file = open(OUTPUT_DIR+"batch/diff_values.txt",'a')
			diff_file.write("i="+str(i)+", l="+str(l)+": "+str(np.array(diff_i))+"\n")
			diff_file.close()

			shapley_file = open(OUTPUT_DIR+"batch/shapley_values.txt",'a')
			shapley_file.write("i="+str(i)+", l="+str(l)+": "+str(np.array(shapley_i))+"\n")
			shapley_file.close()

		diff_values[i] = diff_i
		shapley_values[i] = shapley_i
		print("total diff of feature "+str(i)+": "+str(round(shapley_i,6)))
		print("shapley of feature "+str(i)+": "+str(round(shapley_i,6)))
		

	np.save(OUTPUT_DIR+"batch/diff_values", np.array(diff_values))
	np.save(OUTPUT_DIR+"batch/shapley_values", np.array(shapley_values))


def main():
	# INPUT FEATURES
	# time_res_minmax, time_res_stdev, time_res_stdev_occupied, time_res_stdev_blocking, time_res_stdev_occ_block, 
	# time_job_minmax, time_job_minmax_coming, time_job_stdev_all, time_job_stdev_coming, time_job_stdev_blocking_all, time_job_stdev_blocking_coming, 
	# blocking_res_stdev, rev_blocking_res_stdev, blocking_job_all_stdev, rev_blocking_job_all_stdev, blocking_job_coming_stdev, rev_blocking_job_coming_stdev, blocking, rev_blocking, 
	# u0_occupied, future_blockings_mean, future_rev_blockings_mean, future_blockings_median, future_rev_blockings_median, 
	# already_processing, T_expected, time_to_duedate, relative_duedate, n, m, lv, gv

	# SETTINGS
	layer_dims = [32, 25, 16, 7, 1]
	OUTPUT_DIR = '/home/merel/Documents/studie/IS/thesis/output/'

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
	#     for i in range(0,len(X)):
	#         X[i] = [float(x) for x in X[i]]
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

	find_shapley_values(OUTPUT_DIR, X_test, y_test, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, zero_weights)


if __name__ == '__main__':
    main()