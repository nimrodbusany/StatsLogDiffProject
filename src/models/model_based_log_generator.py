import numpy as np
from src.graphs.graphs import DGraph
from src.main.config import RESULTS_PATH
import random
from src.ktails.ktails import INIT_LABEL, TERM_LABEL
import os
from src.logs.log_writer import LogWriter


class LogGenerator:

    def __init__(self, model, edge_prob_attribute='transition_probability', aggregated_probs=True):
        self.model = model
        self.init_nodes = [n[0] for n in model.get_initial_nodes()]
        self.terminal_nodes = [n[0] for n in model.get_sink_nodes()]
        self.aggregated_probs = aggregated_probs
        self.nodes_2_edges_dict = {}
        for node in model.nodes():
            edges = list(self.model.out_edges(node))
            weights = [model.get_edge_data(e)[edge_prob_attribute] for e in edges]
            if aggregated_probs:
                agg_weights = []
                for i in range(len(weights)):
                    agg_weights.append(sum(weights[:i+1]))
            self.nodes_2_edges_dict[node] = {"edges": edges, "transition_probability": list(agg_weights)}

        if len(self.init_nodes) == 0:
            raise AssertionError('To generate a trace, at least one node must be provided!')

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

    def _generate_trace(self, edge_label_attribute='label', node_label_attribute=None, edge_prob_attribute='weight'):
        '''
        :param g: a directed graph,
        Assumption 1: must have at least one sink state
        Assumption 2: sink states are interpreted as terminal states
        Assumption 3: it is assumed that any state in g can reach a terminal state
        :return: a trace
        '''
        current_node = self.init_nodes[0]
        if len(self.init_nodes) > 1:
            # if multiple initial nodes exists, choose one at random: TODO: use node weight when possible
            # current_node = self._choose_element_at_random()
            raise NotImplemented("currently unsupported")

        trace = []
        while current_node not in self.terminal_nodes: ## TODO: allow looping out of terminal states, when possible!
            # if required, add nodes label
            if node_label_attribute:
                trace.append(self.model.node_attr(current_node, node_label_attribute))

            # select next edge randomaly by weight
            node_tup = self.nodes_2_edges_dict[current_node]
            edges = node_tup["edges"] #self.model.out_edges(current_node)
            weights = node_tup["transition_probability"] ##[g.get_edge_data(e)[edge_prob_attribute] for e in edges]
            e = self._choose_element_at_random(edges, weights)

            # if required, add edge label
            if edge_label_attribute:
                trace.append(self.model.get_edge_data(e)[edge_label_attribute])
            current_node = e[1]

        return tuple(trace)



    def _choose_at_random_by_method(self, N, probabilities):
        if not self.aggregated_probs:
            return np.random.choice(N, p=probabilities, size=1)[0]
        else:
            if probabilities[-1] > 0.99:
                probabilities[-1] = 1
            else:
                raise ValueError('BAD probs'+str(probabilities))
            import random
            num = random.random()
            i = 0
            while num > probabilities[i]:
                i += 1

            return i

    # [ 0.2, 0.5, 1] 0.1 -> 0; 0.4 -> 1; 0.6;
    def _choose_element_at_random(self, elements, probabilities = None, \
                                  perfrom_checks=False):
        '''
        :param elements: : a list of elemnts
        :param probabilities: a list of probabilities per elements (must sum to 1); assumed uniformed of non is provided
        :return: element from the list
        '''

        if perfrom_checks:
            if abs(1 - sum(probabilities)) > 0.001:
                raise AssertionError('probabilities must some to 1.0, but sum to '+ str(sum(probabilities) + ' values: ' + str(probabilities)))
        if min(probabilities) < 0:
            if min(probabilities) < -0.001:
                raise AssertionError('probabilities must non negative: ' + str(probabilities))
            else:
                for i in range(probabilities):
                    probabilities[i] = 0
        N = len(elements)
        if not probabilities:
            probabilities = []
            sum_=0
            for i in range(N):
                if self.aggregated_probs:
                    sum_ += 1.0 / N
                    probabilities.append(sum_)
                else:
                    probabilities.append(1.0 / N)
        ind = self._choose_at_random_by_method(N, probabilities=probabilities)
        return elements[ind]

    @classmethod
    def produce_log_from_single_model(cls, model, size_ =10, add_dummy_initial= True, add_dummy_terminal= True):
        traces = []
        for i in range(size_):
            t = cls._generate_trace(model)
            if add_dummy_initial:
                t.insert(0, INIT_LABEL)
            if add_dummy_terminal:
                t.append(TERM_LABEL)
            traces.append(tuple(t))
        return traces


    @classmethod
    def produce_log_from_single_split_models(cls, bias = 0.0, size_ =10, add_dummy_initial= True, add_dummy_terminal= True):
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
        return cls.produce_log_from_single_model(g, bias, size_ , add_dummy_initial, add_dummy_terminal)


    @classmethod
    def produce_toy_logs(cls, bias= 0.1, N=1000):
        '''
        produce a toy logs from two ktails_models, one completely random choice, the other with a bias
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

    @staticmethod
    def produce_log_from_model(model, traces2produce=1000, transition_probability_attribute='prob'):
        log = []
        log_gen = LogGenerator(model)
        for i in range(traces2produce):
            trace = log_gen._generate_trace(edge_prob_attribute=transition_probability_attribute)
            log.append(trace)
        return log


def produce_mutated_log():

    from src.models.protocol_models_to_logs import ProtocolModel
    model_dir = 'C:/Users/USER/PycharmProjects/StatsLogDiffProject/models/'
    ordset_path = model_dir + "stamina/ordSet.net.dot"
    zip_path = model_dir + "/zeller/ZipOutputStream.dot"
    smtp_path = model_dir + "/zeller/SMTPProtocol.dot"
    # csv_path = model_dir + "stamina/cvs.net.dot"
    ssh_path = model_dir + "stamina/ssh.net.dot"
    tcp_path = model_dir + "stamina/tcpip.net.dot"
    room_controller_path = model_dir + "stamina/roomcontroller.net.dot"

    models_paths = [room_controller_path, ssh_path, tcp_path, ordset_path, zip_path, smtp_path, ]
    models = ['roomcontroller', 'ssh', 'tcpip', 'ordSet', 'ZipOutputStream', 'SMTPProtocol']
    for model_ind in range(len(models_paths)):
        model_path = models_paths[model_ind]
        model = models[model_ind]
        for itr in range(3):
            print('processing', model, 'run', itr)
            model_output_path = "C:/Users/USER/PycharmProjects/StatsLogDiffProject/results/user_study_logs/" + model + "/series_" + str(itr) + "/"
            if not os.path.exists(model_output_path):
                os.makedirs(model_output_path)
            for diff in [0.05, 0.1, 0.2]:
                model_generator = ProtocolModel(model_path, 0, assign_transtion_probs=True)
                log_gen = LogGenerator(model_generator.graph)
                print('producing log 1')
                log1 = LogGenerator.produce_log_from_model(model_generator.graph, 5000)
                print('producing log 2')
                log2 = LogGenerator.produce_log_from_model(model_generator.graph, 5000)
                print('producing log 3')
                log3 = LogGenerator.produce_log_from_model(model_generator.graph, 5000)
                nodes = list(model_generator.graph.nodes())
                random.shuffle(nodes)
                mutation = ""
                for n in nodes:
                    print('edges of node', n)
                    edges_tuples = log_gen.nodes_2_edges_dict[n]
                    if len(edges_tuples['edges']) > 2:
                        e1 = None
                        e2 = None
                        for i in range(len(edges_tuples['edges'])):
                            e = edges_tuples['edges'][i]
                            p = edges_tuples['transition_probability'][i]
                            if p > diff:
                                if e1 == None:
                                    e1 = e
                                else:
                                    e2 = e

                        if e1 and e2:
                            e1_data = model_generator.graph.get_edge_data(e1)
                            e2_data = model_generator.graph.get_edge_data(e2)
                            mutation += "Before mutation:\n"
                            mutation += " e1: " + str(e1) + ", label: " + e1_data['label'] + ", prob: " + str(e1_data['transition_probability']) + "\n"
                            mutation += " e2: " + str(e2) + ", label: " + e2_data['label'] + ", prob: " + str(e2_data[
                                'transition_probability']) + "\n"
                            model_generator.graph.get_edge_data(e1)['transition_probability'] -= diff
                            model_generator.graph.get_edge_data(e2)['transition_probability'] += diff
                            mutation += "After mutation:\n"
                            mutation += " e1: " + str(e1) + ", label: " + e1_data['label'] + ", prob: " + str(e1_data[
                                'transition_probability']) + "\n"
                            mutation += " e2: " + str(e2) + ", label: " + e2_data['label'] + ", prob: " + str(e2_data[
                                'transition_probability']) + "\n"
                            break

                print('producing log 4')
                log4 = LogGenerator.produce_log_from_model(model_generator.graph, 5000)
                print('producing log 5')
                log5 = LogGenerator.produce_log_from_model(model_generator.graph, 5000)
                print('producing log 6')
                log6 = LogGenerator.produce_log_from_model(model_generator.graph, 5000)
                log_set_dir = model_output_path + "/" + "logset_" + str(diff) + "/"
                if not os.path.exists(log_set_dir):
                    os.makedirs(log_set_dir)
                LogWriter.write_log(log1, log_set_dir + "log_tmp1.txt")
                LogWriter.write_log(log2, log_set_dir + "log_tmp2.txt")
                LogWriter.write_log(log3, log_set_dir + "log_tmp3.txt")
                LogWriter.write_log(log4, log_set_dir + "log_tmp4.txt")
                LogWriter.write_log(log5, log_set_dir + "log_tmp5.txt")
                LogWriter.write_log(log6, log_set_dir + "log_tmp6.txt")
                with open(log_set_dir + 'mutation.txt', 'w') as fw:
                    fw.write(mutation)

if __name__ == '__main__':

    produce_mutated_log()
    # l1, l2 = LogGenerator.produce_toy_logs()
    # g = LogGenerator.generate_single_split_model()
    # g.write_dot(RESULTS_PATH + '/exmaple_1.dot', True)
    # ks_dict = {}
    # for i in range(100):
    #     t = LogGenerator._generate_trace(g)
    #     ks_dict[t] = ks_dict.get(t, 0) + 1
    # print (ks_dict)