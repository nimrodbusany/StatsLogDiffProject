import numpy as np


def sample_traces(log, n):
    '''
    returns a list of n randomally selected traces from log (sampling with return)
    :param n: number of traces to draw
    :param log:
    :return: list of traces
    '''
    ind = np.random.random_integers(low= 0, high= len(log) -1, size= n)
    return [log[i] for i in ind]
