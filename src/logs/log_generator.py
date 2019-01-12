from src.graphs.graph import DGraph
from src.main.config import RESULTS_PATH
import numpy as np

class LogGeneator:

    @classmethod
    def _generate_single_split_model(cls, split_bias_probability=0.0):
        '''

        :param split_bias_probability: how far from random (i.e., 50/50) should the split be); must be <= 0.5
        :return: the model:
        1->(a, 0.5+bias)->2
        1->(a, 0.5-bias)->3
        2->(b, 1)->4
        3->(c, 1)->4
        '''
        if split_bias_probability > 0.5:
            raise AssertionError('split bais must be smaller then 0.5')

        g = DGraph()
        g.add_node(1, 'n1')
        g.add_node(2, 'n2')
        g.add_node(3, 'n3')
        g.add_node(4, 'n4')
        g.add_edge(1, 2, 0.5 + split_bias_probability)
        g.add_edge_attribute(1, 2, 'label', 'a')
        g.add_edge(1, 3, 0.5 - split_bias_probability)
        g.add_edge_attribute(1, 3, 'label', 'a')
        g.add_edge(2, 4, 1)
        g.add_edge_attribute(2, 4, 'label', 'b')
        g.add_edge(3, 4, 1)
        g.add_edge_attribute(3, 4, 'label', 'c')
        return g

    @classmethod
    def _choose_element_at_random(cls, elements, probabilities = None):
        '''
        :param elements: : a list of elemnts
        :param probabilities: a list of probabilities per elements (must sum to 1); assumed uniformed of non is provided
        :return: element from the list
        '''
        if sum(probabilities) != 1:
            raise AssertionError('probabilities must some to 1.0')
        N = len(elements)
        if not probabilities:
            probabilities = []
            for i in range(N):
                probabilities.append(1.0 / N)
        ind = np.random.choice(N, p=probabilities, size=1)
        return elements[ind[0]]

    @classmethod
    def _generate_trace(cls, g, edge_label_attribute='label', node_label_attribute=None, edge_prob_attribute='weight'):
        '''
        :param g: a directed graph,
        Assumption 1: must have at least one sink state
        Assumption 2: sink states are interpreted as terminal states
        Assumption 3: it is assumed that any state in g can reach a terminal state
        :return: a trace
        '''
        init_nodes = [n[0] for n in g.get_initial_nodes()]
        terminal_nodes = [n[0] for n  in g.get_sink_nodes()]
        if len(init_nodes) == 0:
            raise AssertionError('To generate a trace, at least one node must be provided!')

        current_node = init_nodes[0]
        if len(init_nodes) > 1:
            # if multiple initial nodes exists, choose one at random: TODO: use node weight when possible
            current_node = cls._choose_element_at_random(init_nodes)

        trace = []
        while current_node not in terminal_nodes:
            # if required, add nodes label
            if node_label_attribute:
                trace.append(g.node_attr(current_node, node_label_attribute))

            # select next edge randomaly by weight
            edges = g.out_edges(current_node)
            weights = [g.get_edge_data(e)[edge_prob_attribute] for e in edges]
            e = cls._choose_element_at_random(list(edges), weights)

            # if required, add edge label
            if edge_label_attribute:
                trace.append(g.get_edge_data(e)[edge_label_attribute])
            current_node = e[1]

        return trace

    @classmethod
    def produce_log_from_single_split_models(cls, bias = 0.0, size =10, add_dummy_initial= True, add_dummy_terminal= True):
        '''
            return logs from the model:
            1->(a, 0.5+bias)->2
            1->(a, 0.5-bias)->3
            2->(b, 1)->4
            3->(c, 1)->4
            :param split_bias_probability: how far from random (i.e., 50/50) should the split be); must be <= 0.5
            :return:
            '''
        g = cls._generate_single_split_model(bias)
        traces = []
        for i in range(size):
            t = cls._generate_trace(g)
            if add_dummy_initial:
                t.insert(0, 'I')
            if add_dummy_terminal:
                t.append('T')
            traces.append(tuple(t))
        return traces

    @classmethod
    def produce_toy_logs(cls, bias= 0.1, N=1000):
        '''
        produce a toy logs from two models, one completely random choice, the other with a bias
        a, b
        # a, c
        # d->e
        # d->f
        # g->j
        # g->h
        :return:  two logs according to distrubutions, first with no bias, second with bias.
        '''
        if bias < 0 or bias > 0.5:
            raise ValueError('bias must be between [0, 0.5]')

        if N <= 0:
            raise ValueError('N must be > 0')

        traces = [('I', 'a', 'b'), ('I', 'a', 'c')]
        random_ind = np.random.choice(2, p=[0.5, 0.5], size=N)
        random_ind2 = np.random.choice(2, p=[0.5+bias, 0.5-bias], size=N)
        random_log = [traces[i] for i in random_ind]
        random_log2 = [traces[i] for i in random_ind2]
        return random_log, random_log2

if __name__ == '__main__':

    l1, l2 = LogGeneator.produce_toy_logs()
    print(l1)
    print('-----------------')
    print(l2)

    # g = LogGeneator.generate_single_split_model()
    # g.write_dot(RESULTS_PATH + '/exmaple_1.dot', True)
    # ks_dict = {}
    # for i in range(100):
    #     t = tuple(LogGeneator.generate_trace(g))
    #     ks_dict[t] = ks_dict.get(t, 0) + 1
    # print (ks_dict)