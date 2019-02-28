import random
import networkx as nx
import numpy as np

from src.graphs.graphs import DGraph
from src.models.model_based_log_generator import LogGenerator
from src.logs.log_writer import LogWriter
from src.logs_parsers.simple_log_parser import SimpleLogParser
from src.utils.disk_operations import create_folder_if_missing

from src.utils.project_constants import *

__EXTRA_CHECKS__ = True

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
    def _assign_transition_probabilities(graph):

        edges2weights = {}
        for n in graph.nodes():
            out_edges = list(graph.out_edges(n))
            if len(out_edges) == 0:
                continue
            sum_ = 0

            min_edge_w = 0.05
            min_edge_w_requires = (len(out_edges) * min_edge_w)
            random_weight_remains = 1.0 - min_edge_w_requires
            for e in out_edges:
                e_w = random.random()
                edges2weights[e] = e_w
                sum_ += e_w
            total_ = 0
            for e in out_edges:
                edges2weights[e] /= sum_
                edges2weights[e] = min_edge_w + round(edges2weights[e], 3) * random_weight_remains
                total_ += edges2weights[e]
            if total_ < 1:
                edge = out_edges[out_edges.index(min(out_edges, key=lambda x: edges2weights[x]))]
                edges2weights[edge] += round((1.0 - total_), 3)
            if total_ > 1:
                edge = out_edges[out_edges.index(max(out_edges, key=lambda x: edges2weights[x]))]
                edges2weights[edge] += round((total_ - 1.0), 3)
            if __EXTRA_CHECKS__:
                for w in out_edges:
                    if edges2weights[w] <= min_edge_w:
                        raise AssertionError('edge with less than '+str(min_edge_w) + " weight is " + str(edges2weights[w]))
                total_ = sum([edges2weights[e] for e in out_edges])
                if not 0.99 < total_ < 1.01:
                    raise AssertionError('bad weights, total eqauls:' + str(total_))
        nx.set_edge_attributes(graph.dgraph, edges2weights, TRANSITION_PROBABILITY_ATTRIBUTE)

    def __init__(self, model_path, id=0, assign_transtion_probs=True, add_sink_node=False):
        graph = DGraph.read_dot(model_path)
        if add_sink_node:
            self._add_sink_node(graph)
        if assign_transtion_probs:
            self._assign_transition_probabilities(graph)
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


def produce_logs_from_stamina(traces2produce, models2produce):

    MODELS_PATH = '../../models/stamina/'
    LOGS_OUTPUT_PATH = '../../data/logs/stamina/'
    models = ['ctas.net.simplified.no_loops.dot', 'cvs.net.no_loops.dot', 'roomcontroller.simplified.net.dot',
              'ssh.net.simplified.dot', 'tcpip.simplified.net.dot', 'zip.simplified.dot']
    dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip', 'zip']
    # MODEL_TO_PRODUCE = 16

    for model_id in range(len(models)):
        print('processing', models[model_id])
        dir_ = LOGS_OUTPUT_PATH + dirs_[model_id] + "/"
        model_path = MODELS_PATH + models[model_id]
        create_folder_if_missing(dir_)

        for instance_id in range(models2produce):
            print('processing instance:', instance_id, model_path)
            model_generator = ProtocolModel(model_path, instance_id)
            log = LogGenerator.produce_log_from_model(model_generator.graph,
                                                      transition_probability_attribute=TRANSITION_PROBABILITY_ATTRIBUTE, traces2produce=traces2produce)
            model_generator.write_transitions_probabilities(dir_)
            LogWriter.write_log(log, dir_ + 'l' + str(instance_id) + ".log")


def produce_logs_from_david(traces2produce, models2produce):
    MODELS_PATH = '../../models/david/'
    LOGS_OUTPUT_PATH = '../../data/logs/david/'
    models = ['Columba.simplified.dot', 'Heretix.simplified.dot', 'JArgs.simplified.dot',
              'Jeti.Simplified.dot', 'jfreechart.Simplified.dot', 'OpenHospital.Simplified.dot',
              'RapidMiner.Simplified.dot', 'tagsoup.Simplified.dot']
    dirs_ = ['Columba', 'Heretix', 'JArgs', 'Jeti', 'jfreechart', 'OpenHospital', 'RapidMiner', 'tagsoup']
    # MODEL_TO_PRODUCE = 4

    for model_id in range(0, len(models)):
        print('processing', models[model_id])
        dir_ = LOGS_OUTPUT_PATH + dirs_[model_id] + "/"
        model_path = MODELS_PATH + models[model_id]
        create_folder_if_missing(dirs_)

        for instance_id in range(models2produce):
            print('processing instance:', instance_id, model_path)
            model_generator = ProtocolModel(model_path, instance_id, assign_transtion_probs=True)
            # transition_probs = nx.get_edge_attributes(model_generator.graph.dgraph, TRANSITION_PROBABILITY_ATTRIBUTE)
            # for e in transition_probs:
            #     from fractions import Fraction
            #     transition_probs[e] = float(Fraction(transition_probs[e].strip("\"")))
            # nx.set_edge_attributes(model_generator.graph.dgraph, transition_probs, TRANSITION_PROBABILITY_ATTRIBUTE)

            log = LogGenerator.produce_log_from_model(model_generator.graph,
                                                      transition_probability_attribute=TRANSITION_PROBABILITY_ATTRIBUTE,
                                                      traces2produce=traces2produce)
            model_generator.write_transitions_probabilities(dir_)
            LogWriter.write_log(log, dir_ + 'l' + str(instance_id) + ".log")



def compute_logs_statistics(logs_base_folder, dirs_, logs_in_folder):
    for dir_ in dirs_:
        model_means_ = []
        for instance_id in range(logs_in_folder):
            log_path = logs_base_folder + dir_ + "/l" + str(instance_id) + ".log"
            traces = SimpleLogParser.read_log(log_path)
            model_means_.append(np.mean([len(t) for t in traces]))
        print(dir_, np.mean(model_means_))

def compute_bear_stats():

    LOGS_OUTPUT_PATH = '../../data/bear/'
    dirs_ = ['mobile.log', 'desktop.log']
    traces = SimpleLogParser.read_log(LOGS_OUTPUT_PATH + dirs_[0])
    print('mobile', np.mean([len(t) for t in traces]))
    print('mobile alphabet', len(set([w for t in traces for w in t])))
    traces = SimpleLogParser.read_log(LOGS_OUTPUT_PATH + dirs_[1])
    print('desktop', np.mean([len(t) for t in traces]))
    print('desktop alphabet', len(set([w for t in traces for w in t])))


if __name__ == '__main__':

    TRACE2PRODUCE = 200000
    MODEL_TO_PRODUCE = 8

    # produce_logs_from_stamina(TRACE2PRODUCE, MODEL_TO_PRODUCE)
    produce_logs_from_david(TRACE2PRODUCE, MODEL_TO_PRODUCE)
    #
    # LOGS_OUTPUT_PATH = '../../data/logs/david/'
    # dirs_ = ['Columba', 'Heretix', 'JArgs', 'Jeti', 'jfreechart', 'OpenHospital', 'RapidMiner', 'tagsoup']
    # compute_logs_statistics(LOGS_OUTPUT_PATH, dirs_, LOGS_IN_FOLDER)

    # compute_bear_stats()


    # LOGS_OUTPUT_PATH = '../../data/logs/stamina/'
    # dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip', 'zip']
    # compute_logs_statistics(LOGS_OUTPUT_PATH, dirs_, LOGS_IN_FOLDER)


