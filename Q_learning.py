def update_policy_Q(resources, states, actions, STACT, ALPHA, GAMMA):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state
        
        if resource.last_action != None:
            s1 = states.index(resource.state)          # current state
            s0 = states.index(resource.prev_state)     # previous state
            a = actions.index(resource.last_action)    # taken action

            if STACT == "st_act":
                next_max = np.max(resource.policy[s1])    # max q-value of current state
                q_old = resource.policy[s0, a]
                q_new = (1 - GAMMA) * q_old + GAMMA * (resource.reward + ALPHA * next_max)
                resource.policy[s0, a] = q_new
            elif (STACT == "act") or (STACT == "NN"):
                next_max = resource.policy[a]             # q-value of current state
                q_old = resource.policy[a]
                q_new = (1 - GAMMA) * q_old + GAMMA * (resource.reward + ALPHA * next_max)
                resource.policy[a] = q_new

        resource.reward = 0     # reset reward

    return resources