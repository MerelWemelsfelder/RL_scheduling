
def update_history(resources, z):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state

        resource.h[z] = (resource.prev_state, resource.last_action)

    return resources

def update_policy_JEPS(resource, states, actions, r_best, time_max, GAMMA, STACT):
    for z in range(time_max-1):
        s = resource.h[z][0]
        a = resource.h[z][1]
        if a != None:
            s_index = states.index(s)     # previous state
            a_index = actions.index(a)    # taken action
            for job in s:
                if job == a:
                    if STACT == "st_act":
                        resource.policy[s_index,a_index] = resource.policy[s_index,a_index] + (GAMMA * (1 - resource.policy[s_index,a_index]))
                    elif (STACT == "act") or (STACT == "NN"):
                        resource.policy[a_index] = resource.policy[a_index] + (GAMMA * (1 - resource.policy[a_index]))
                else:
                    if STACT == "st_act":
                        resource.policy[s_index,a_index] = (1 - GAMMA) * resource.policy[s_index,a_index]
                    elif (STACT == "act") or (STACT == "NN"):
                        resource.policy[a_index] = (1 - GAMMA) * resource.policy[a_index]
    return resource

