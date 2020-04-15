M = 1		# number of work stations
LV = 4		# number of resources in w_1
GV = 3		# number of units per resource in w_1
N = 10		# number of jobs

ALPHA = 0.2		# learning rate (0<α≤1): the extent to which Q-values are updated every timestep
GAMMA = 0.6		# discount factor (0≤γ≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)   
EPSILON = 0.2	# probability of choosing a random action (= exploring)

EPOCHS = 1000       # set number of epochs to train RL model

METHOD = "JEPS" 	# Q_learning / JEPS
STACT = "act"	# st_act for state-action pairs, act for only actions