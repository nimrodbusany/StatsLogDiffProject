import numpy as np

def create_sample_realization(number_of_trace, p1):
    a = np.ones(int(p1 * number_of_trace))
    b = np.zeros(number_of_trace)
    b[:len(a)] += a
    return b

def sample_traces(log, n):
    '''
    returns a list of n randomally selected traces from log (sampling with return)
    :param n: number of traces to draw
    :param log:
    :return: list of traces
    '''
    ind = np.random.random_integers(low= 0, high= len(log) -1, size= n)
    return [log[i] for i in ind]
