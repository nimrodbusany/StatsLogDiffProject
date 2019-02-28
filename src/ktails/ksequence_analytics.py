import networkx as nx
from src.utils.disk_operations import create_folder_if_missing
from src.models.protocol_models_to_logs import ProtocolModel
from src.utils.project_constants import *


def produce_all_paths(graph):

    source = graph.get_initial_nodes()
    sink = graph.get_sink_nodes()
    if len(source) > 1 or len(sink) > 1:
        raise NotImplemented('only supporting graphs with sinlge source and single sink')
    source = source[0][0]
    sink = sink[0][0]
    paths = list(nx.all_simple_paths(graph.dgraph, source, sink))
    return paths

def compute_path_probability(graph, path):
    trace_prob = 1
    probs = nx.get_edge_attributes(graph, TRANSITION_PROBABILITY_ATTRIBUTE)
    for i in range(len(path) - 1):
        transition = (path[i], path[i + 1])
        trace_prob *= probs[transition]
    return trace_prob


def update_transition_probabilities_to_probabilities_by_path(graph, path, k_seq_transitions_probabilities, k):

    probs = nx.get_edge_attributes(graph, TRANSITION_PROBABILITY_ATTRIBUTE)
    labels = nx.get_edge_attributes(graph, TRANSITION_LABEL_ATTRIBUTE)

    trace_prob = 1
    path_labels = []
    for i in range(len(path) - 1):
        transition = (path[i], path[i+1])
        path_labels.append(labels[transition])
        trace_prob *= probs[transition]

    for i in range(len(path_labels)):
        k_seq_transition = tuple(path_labels[i:i+k+1])
        k_seq_transitions_probabilities[k_seq_transition] = k_seq_transitions_probabilities.get(k_seq_transition, 0) + trace_prob


def finalize_transition_probabilities(k_seqs2probabilities, k):

    k_seqs2k_seqs = {}
    for k_plus_seq in k_seqs2probabilities:
        source_last_ind = min(k, len(k_plus_seq))
        source_k_seq = k_plus_seq[:source_last_ind]
        target_k_seq = k_plus_seq[1:]
        k_seqs2k_seqs[source_k_seq] = k_seqs2k_seqs.get(source_k_seq,{})
        k_seqs2k_seqs[source_k_seq][target_k_seq] = \
            k_seqs2k_seqs[source_k_seq].get(target_k_seq, 0) + k_seqs2probabilities[k_plus_seq]

    for source in k_seqs2k_seqs:
        targets = k_seqs2k_seqs[source]
        total_prob = sum([targets[target] for target in targets])
        for target in targets:
            targets[target] /= total_prob

    return k_seqs2k_seqs


def write_k_seq_probs_to_file(output_path, k_seqs2probabilities):

    with open(output_path, 'w') as fw:
        fw.write("k_sequence, probability\n")
        for k_seq in k_seqs2probabilities:
            fw.write(str(k_seq).replace(",", ";") + ', ' + str(k_seqs2probabilities[k_seq]) + '\n')


def compute_k_sequence_transition_probabilities(model, k):

    def validate_model_against_paths(model, paths):
        total_probs = 0
        for path in paths:
            total_probs += compute_path_probability(model.graph.dgraph, path)
        if total_probs < 0.99 or total_probs > 1.01:
            raise AssertionError("all paths must sum to 1, but equal " + str(total_probs), "; check model assumptions: no loops, valid probabilities")

    paths = produce_all_paths(model.graph)
    validate_model_against_paths(model, paths)
    k_seqs_transitions_probabilities = {}
    for path in paths:
        update_transition_probabilities_to_probabilities_by_path(model.graph.dgraph, path,
                                                                 k_seqs_transitions_probabilities, k)

    k_seqs2k_seqs = finalize_transition_probabilities(k_seqs_transitions_probabilities, k)
    return k_seqs2k_seqs

def ksequence_analytics_of_stamina(k):

    MODELS_PATH = '../../models/stamina/'
    LOGS_OUTPUT_PATH = '../../data/logs/stamina/'
    models = ['ctas.net.simplified.no_loops.dot', 'cvs.net.no_loops.dot','roomcontroller.simplified.net.dot',
              'ssh.net.simplified.dot', 'tcpip.simplified.net.dot', 'zip.simplified.dot']
    dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip', 'zip']

    for i in range(len(models)):
        print('processing:', models[i])
        dir_ = LOGS_OUTPUT_PATH + dirs_[i] + "/"
        model_path = MODELS_PATH + models[i]
        create_folder_if_missing(dir_)
        model = ProtocolModel(model_path)
        k_seqs2k_seqs = compute_k_sequence_transition_probabilities(model, k)
        write_k_seq_probs_to_file(dir_ + 'k_sequence_probs.csv', k_seqs2k_seqs)
        print("Total k_seqs:", len(k_seqs2k_seqs))
        # print("K-sequences dict:", k_seqs2probabilities)

        # model_generator.produce_logs()


def ksequence_analytics_of_pardel(k): ## TODO
    PARDEL_PATH = '../../models/pradel/'
    pass


def ksequence_analytics_of_david(k): ## TODO
    PARDEL_PATH = '../../models/david/'
    pass

def ksequence_analytics_of_zeller(k): ## TODO
    PARDEL_PATH = '../../models/david/'
    pass


if __name__ == '__main__':

    K = 3
    ksequence_analytics_of_david(K)
    ksequence_analytics_of_pardel(K)
    ksequence_analytics_of_stamina(K)
    ksequence_analytics_of_zeller(K)

