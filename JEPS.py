# Update the history of actions that were taken during current epoch
def update_history(resources, z):
    for resource in resources:
        resource.state = resource.units[0].state.copy()     # update resource state

        # dictionary with current timestep as key, and (state, action) pair as value
        resource.h[z] = (resource.prev_state, resource.last_action)

    return resources

# Update all resource's policies according to JEPS
def update_policy_JEPS(resource, actions, time_max, GAMMA):
    for z in range(time_max-1):
        s = resource.h[z][0]
        a = resource.h[z][1]

        # Would be None if the first unit of this resource was already
        # processing a job and therefore not able to execute an action
        if a != None:
            a_index = actions.index(a)
            for job in s:
                if job == a:
                    # Update the policy value for the action that was executed
                    resource.policy[a_index] = resource.policy[a_index] + (GAMMA * (1 - resource.policy[a_index]))
                else:
                    # Update the policy values for the actions that were available but not executed
                    resource.policy[a_index] = (1 - GAMMA) * resource.policy[a_index]
    return resource

