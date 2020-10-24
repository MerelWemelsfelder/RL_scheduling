from main import test
from NN import *
from utils import *
from sklearn.model_selection import train_test_split

def generate_batch_data(INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR):

    PHASE = "train"         # train / load
    TIMEOUT = 3*60            # allowed computation time in sec
    MILP_TIMEOUT = 30*60    # allowed computation time for MILP in sec
    EPSILON = 1.0
    EPSILON_DECREASE = 0.025
    GAMMA = 0.7

    M = 1
    GV = [6]

    N = 5
    for LV in [2, 3]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    N = 7
    for LV in [2, 4]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    N = 11
    for LV in [4, 6]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    N = 13
    for LV in [3, 7]:
        test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))


    # N = 5
    # for LV in [2, 3]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 7
    # for LV in [2, 4, 6]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 9
    # for LV in [3, 5, 7]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 11
    # for LV in [4, 6, 8]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 13
    # for LV in [3, 5, 7, 9]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 15
    # for LV in [3, 5, 8, 10]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 17
    # for LV in [4, 6, 9, 11]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # N = 19
    # for LV in [4, 6, 10, 13]:
    #     test(N, M, [LV], GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, random.randrange(0,2000))

    # LOAD TRAINING & TEST SET
    folder = INPUT_CONFIGS[CONFIG]+"/"
    with open(OUTPUT_DIR+"batch/"+folder+"X.txt") as f:
        X = f.readlines()
        X = [x.strip().split(";") for x in X]
        for f in range(0,len(X)):
            X[f] = [float(x) for x in X[f]]
    with open(OUTPUT_DIR+"batch/"+folder+"y.txt") as f:
      y = f.readlines()
      y = [float(y.strip("\n")) for y in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    np.save(OUTPUT_DIR+"batch/"+folder+"X_train", np.array(X_train))
    np.save(OUTPUT_DIR+"batch/"+folder+"X_test", np.array(X_test))
    np.save(OUTPUT_DIR+"batch/"+folder+"y_train", np.array(y_train))
    np.save(OUTPUT_DIR+"batch/"+folder+"y_test", np.array(y_test))

def batch_training(NN_weights, NN_weights_gradients, NN_biases, NN_biases_gradients, OUTPUT_DIR, layer_dims, weight_decay, GAMMA, INPUT_CONFIGS, CONFIG):

    NN = NeuralNetwork(
        Dense(NN_weights[0], NN_weights_gradients[0], NN_biases[0], NN_biases_gradients[0]), 
        Sigmoid(),
        Dense(NN_weights[1], NN_weights_gradients[1], NN_biases[1], NN_biases_gradients[1]), 
        Sigmoid(),
        Dense(NN_weights[2], NN_weights_gradients[2], NN_biases[2], NN_biases_gradients[2]),
        Sigmoid())
    loss = NLL()

    NN = batch_train_NN(NN, loss, OUTPUT_DIR, weight_decay, GAMMA, INPUT_CONFIGS, CONFIG)
    params = NN.get_params()
    NN_weights = [params[i] for i in range(0,len(params),2)]
    NN_biases = [params[i] for i in range(1,len(params),2)]
    grads = NN.get_params_gradients()
    NN_weights_gradients = [grads[i] for i in range(0,len(grads),2)]
    NN_biases_gradients = [grads[i] for i in range(1,len(grads),2)]
    write_NN_weights(OUTPUT_DIR, layer_dims, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, INPUT_CONFIGS, CONFIG)

def feature_set_selection(INPUT_CONFIGS, CONFIG, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):
    M = 1
    GV = [6]
    RSEED = 500
    EPOCHS = 90000
    TIMEOUT = 3*60
    
    if PHASE == "train":
        EPSILON = 1.0
        EPSILON_DECREASE = 0.025
        GAMMA = 0.7
    elif PHASE == "load":
        EPSILON = 0.5
        EPSILON_DECREASE = 0.1
        GAMMA = 0.3

    N = 6
    for LV in[[2], [3]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

    N = 9
    for LV in [[3], [5]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

    N = 12
    for LV in [[4], [6]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

    N = 15
    for LV in [[5], [7]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

def network_architectures(INPUT_CONFIGS, CONFIG, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):
    M = 1
    GV = [6]
    EPSILON = 1.0
    EPSILON_DECREASE = 0.025
    GAMMA = 0.3
    RSEED = 500

    N = 7
    for LV in [[3], [6]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

    N = 11
    for LV in [[4], [8]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

    N = 14
    for LV in [[5], [10]]:
        test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

def performance_tests_JSSP(INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):
    M = 1
    GV = [6]
    EPSILON_DECREASE = 0.1
    RSEED = 10

    for EPSILON in [0.8]:
        for TIMEOUT in [30, 60, 3*60, 30*60]:        
            N = 4
            LV = [2]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

            N = 8
            LV = [4]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

            N = 18
            LV = [8]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

            N = 32
            LV = [13]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

            N = 47
            LV = [17]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

            N = 64
            LV = [21]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

            N = 81
            LV = [26]
            test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

def performance_tests_FFSSP(INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT):
    GV = [4]
    EPSILON_DECREASE = 0.1  
    RSEED = 10

    for EPSILON in [0.8, 0.5, 0.2]:
        for TIMEOUT in [30, 60, 3*60, 30*60]:
            for M in [2, 3, 5]:
                N = 4
                LV = [2 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

                N = 8
                LV = [4 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

                N = 18
                LV = [8 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

                N = 32
                LV = [13 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

                N = 47
                LV = [17 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

                N = 64
                LV = [21 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)

                N = 81
                LV = [26 for i in range(M)]
                GV = [4 for i in range(M)]
                test(N, M, LV, GV, INPUT_CONFIGS, CONFIG, GAMMA, EPSILON, EPSILON_DECREASE, OBJ_FUN, layer_dims, weight_decay, NN_weights, NN_biases, NN_weights_gradients, NN_biases_gradients, PHASE, METHOD, EPOCHS, MCTS_BUDGET, OUTPUT_DIR, TIMEOUT, MILP_TIMEOUT, RSEED)
