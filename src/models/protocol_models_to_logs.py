from src.graphs.graphs import DGraph
import random
import networkx as nx
import os
from model_based_log_generator import LogGeneator
from log_writer import LogWriter


TRANSITION_PROBABILITY_ATTRIBUTE = 'transition_probability'
TRANSITION_LABEL_ATTRIBUTE = 'label'


class ProtocolModel:


    @staticmethod
    def _add_sink_node(graph):

        terminal_nodes = [n for n in graph.dgraph.nodes() if
                          nx.get_node_attributes(graph.dgraph, 'shape')[n] == 'doublecircle']
        graph.add_node('terminal', {'shape': 'plaintext'})
        for n in terminal_nodes:
            graph.add_edge(n, 'terminal')
            graph.add_edge_attribute(n, 'terminal', 'label', '')
        shapes = nx.get_node_attributes(graph.dgraph, 'shape')
        shapes['terminal'] = 'plaintext'

    @staticmethod
    def _assign_weights(graph):

        edges2weights = {}
        for n in graph.nodes():
            out_edges = list(graph.out_edges(n))
            sum_ = 0
            cpy_edges = out_edges.copy()
            ## set probability  to all edges such that all probabilities sums to one
            while len(out_edges) > 1:
                selected_edge = random.choice(out_edges)
                out_edges.remove(selected_edge)
                weight = round(random.uniform(0.01, 0.95 - sum_), 3)
                sum_ += weight
                edges2weights[selected_edge] = weight
                if weight == 0:
                    raise ValueError('weight cannot be zero, check for error!')
            if len(out_edges) == 1:
                edges2weights[out_edges[0]] = 1 - sum_
                if 1 - sum_ == 0:
                    raise ValueError('weight cannot be zero, check for error!')
                total = sum([edges2weights[edge] for edge in cpy_edges])
                if total < 0.999 or total > 1.001:
                    raise ('when generating probabilities, outgoing edges probabilities did not sum to one, but got', total)
                if total < 1.0:
                    edge = cpy_edges[cpy_edges.index(max(cpy_edges, key=lambda x: edges2weights[x]))]
                    edges2weights[edge] = edges2weights[edge] - (total - 1)

        nx.set_edge_attributes(graph.dgraph, edges2weights, TRANSITION_PROBABILITY_ATTRIBUTE)

    def __init__(self, model_path, id=0, assign_transtion_probs=True, add_sink_node=False):
        graph = DGraph.read_dot(model_path)
        if add_sink_node:
            self._add_sink_node(graph)
        if assign_transtion_probs:
            self._assign_weights(graph)
        self.graph = graph
        self.id = id

    def update_transition_probabilities(self, transition_prob_path):
        ## EXPECTED FORMAT of input file, each transition is described by:
        # (src, trg), transition_prob, label; e.g.,
        # ('1', '11'), 1, initialize_2_
        tran2prob = {}
        with open(transition_prob_path) as fr:
            for l in fr.readlines():
                l = l.strip().replace('\'', '').replace('(', '').replace(')', '').replace(' ', '').split(',')
                from_, to_, prob = l[0], l[1], float(l[2])
                tran2prob[(from_, to_)] = prob
        nx.set_edge_attributes(self.graph.dgraph, tran2prob, 'transition_probability' )

    def write_model(self, output_dir): ## TODO: may not work on windows, unless pygraphicviz is installed
        self.graph.write_dot(output_dir + 'm' + str(self.id) + ".dot")

    def write_transitions_probabilities(self, output_dir):
        with open(output_dir + 'm' + str(self.id) + '_transitions_' + ".csv", 'w') as fw:
            edges_prob = nx.get_edge_attributes(self.graph.dgraph, TRANSITION_PROBABILITY_ATTRIBUTE)
            edges_label = nx.get_edge_attributes(self.graph.dgraph, TRANSITION_LABEL_ATTRIBUTE)
            for edge in edges_prob:
                fw.write(",".join([str(edge), '%.3f' % edges_prob[edge], edges_label[edge]]) + '\n')


def produce_logs_from_stamina(traces2produce):

    MODELS_PATH = '../../models/stamina/'
    LOGS_OUTPUT_PATH = '../../data/logs/stamina/'
    models = ['ctas.net.simplified.no_loops.dot', 'cvs.net.no_loops.dot', 'roomcontroller.simplified.net.dot',
              'ssh.net.simplified.dot', 'tcpip.simplified.net.dot', 'zip.simplified.dot']
    dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip', 'zip']
    MODEL_TO_PRODUCE = 4

    for model_id in range(4, 5):
        print('processing', models[model_id])
        dir_ = LOGS_OUTPUT_PATH + dirs_[model_id] + "/"
        model_path = MODELS_PATH + models[model_id]
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        for instance_id in range(MODEL_TO_PRODUCE):
            print('processing instance:', instance_id, model_path)
            model_generator = ProtocolModel(model_path, instance_id)
            log = LogGeneator.produce_log_from_model(model_generator.graph,
                                                     transition_probability_attribute=TRANSITION_PROBABILITY_ATTRIBUTE, traces2produce=traces2produce)
            model_generator.write_transitions_probabilities(dir_)
            LogWriter.write_log(log, dir_ + 'l' + str(instance_id) + ".log")

def produce_logs_from_padel(): ## TODO
    pass

def produce_logs_from_david(): ## TODO
    pass


if __name__ == '__main__':

    TRACE2PRODUCE = 100000
    produce_logs_from_stamina(TRACE2PRODUCE)
    produce_logs_from_padel()
    produce_logs_from_david()


