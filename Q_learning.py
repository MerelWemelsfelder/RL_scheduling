
def update_policy_Q(resources, states, actions, STACT):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state
        
        if resource.last_action != None:
            s1 = states.index(resource.state)          # current state
            s0 = states.index(resource.prev_state)     # previous state
            a = actions.index(resource.last_action)    # taken action

            if STACT == "st_act":
                next_max = np.max(resource.policy[s1])    # max q-value of current state
                q_old = resource.policy[s0, a]
                q_new = (1 - ALPHA) * q_old + ALPHA * (resource.reward + GAMMA * next_max)
                resource.policy[s0, a] = q_new
            if STACT == "act":
                next_max = resource.policy[a]             # q-value of current state
                q_old = resource.policy[a]
                q_new = (1 - ALPHA) * q_old + ALPHA * (resource.reward + GAMMA * next_max)
                resource.policy[a] = q_new

        resource.reward = 0     # reset reward

    return resources

def heuristic_best_job(tau, LV, GV, N):
    heur_job = dict()

    for i in range(LV):
        heur_j = dict()
        for j in range(N):
            j_total = 0
            for q in range(GV):
                j_total += tau[j][q][i]
            heur_j[j] = j_total
        heur_job[i] = heur_j

    return heur_job

def heuristic_best_resource(heur_j):
    heur_r = dict()
    for j in heur_j[0].keys():
        heur_r[j] = dict()
        for r in heur_j.keys():
            heur_r[j][r] = heur_j[r][j]
    return heur_r

def heuristic_order(delta, LV, GV, N):
    all_jobs = list(range(N))
    heur_order = dict()             # key = resource i
    for i in range(LV):
        r_dict = dict()             # key = job j
        for j in range(N):
            j_dict = dict()         # key = job o
            other = all_jobs.copy()
            other.remove(j)
            for o in other:
                counter = 0
                spare = 0
                for q in range(GV-1):
                    dj = delta[j][q+1][i]
                    do = delta[o][q][i]
                    blocking = dj-do
                    if blocking < 0:
                        spare += blocking
                    if blocking > 0:
                        if spare >= blocking:
                            spare -= blocking
                        else:
                            blocking -= spare
                            counter += blocking
                j_dict[o] = counter
            r_dict[j] = j_dict
        heur_order[i] = r_dict
    return heur_order