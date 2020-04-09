m = 1		# number of work stations
l_1 = 3		# number of resources in w_1
g_1 = 4		# number of units per resource in w_1
n = 5		# number of jobs

alpha = 0.2		# learning rate (0<α≤1): the extent to which Q-values are updated every timestep
gamma = 0.6		# discount factor (0≤γ≤1): how much importance to give to future rewards (1 = long term, 0 = greedy)   
epsilon = 0.2	# probability of choosing a random action (= exploring)

n_epochs = 1000                       # set number of epochs to train RL model

method = "JEPS" 	# Q_learning / JEPS
state_action = "act"	# st_act for state-action pairs, act for only actions