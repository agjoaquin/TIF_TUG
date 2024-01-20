import numpy as np

def filter_u_EMA(stack_ini, alfa):
    stack_filter = np.zeros(0)
    while (np.size(stack_filter)<np.size(stack_ini)):
        if(np.size(stack_filter)<2):
            stack_filter = np.append(stack_filter,stack_ini[np.size(stack_filter)])
        else:
            stack_filter = np.append(stack_filter,
            alfa * stack_ini[np.size(stack_filter)-1] + (1-alfa)*stack_filter[np.size(stack_filter)-2])
    return stack_filter

def derivate_stack(stack_ini, delta_t):
    stack_der = np.zeros(0)
    while (np.size(stack_der)<np.size(stack_ini)):
        if(np.size(stack_der)<2):
            stack_der = np.append(stack_der,0)
        else:
            stack_der = np.append(stack_der,
            (stack_ini[np.size(stack_der)] - stack_ini[np.size(stack_der)-1]) / delta_t) 

    return stack_der