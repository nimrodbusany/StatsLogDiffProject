import networkx as nx
import graph_filtering
from BearLogParser import BearLogParser

INIT_LABEL = '"init"'
TERM_LABEL = '"term"'

def generate_equivalent_maps(traces, k, gen_past=True):

    ftr2ftrs = dict()
    ftr2past = dict()
    transitions2traces = dict()
    tr_id = 0
    for t in traces:
        t.insert(0, INIT_LABEL)
        t.append(TERM_LABEL)
        for i in range(len(t)):
            # if i == 0: ## do not unify futures of dummy init
            #     ftr = tuple([t[0]])
            # else:
            ftr = tuple([w.lower() for w in t[i:i+k]])
            next_ftr = tuple([w.lower() for w in t[i+1:i+k+1]])
            ## update data strucutre
            update_ftr2ftr(ftr, ftr2ftrs, next_ftr)
            update_transitions2traces(ftr, next_ftr, tr_id, transitions2traces)
            if gen_past:
                past_ftr = tuple([w.lower() for w in t[max(0, i-k-4):i]])
                update_ftr2pasts(ftr, past_ftr, ftr2past)
        tr_id += 1

    if not gen_past:
        return ftr2ftrs, transitions2traces
    return ftr2ftrs, ftr2past, transitions2traces


def update_transitions2traces(ftr, next_ftr, tr_id, transitions2traces):
    ## get state outgoing transitions
    ftr_outgoing_transitions_to_traces_map = transitions2traces.get(ftr, {})
    ## udpate outgoing transition visiting traces
    transition_traces = ftr_outgoing_transitions_to_traces_map.get(next_ftr, [])
    transition_traces.append(tr_id)
    ftr_outgoing_transitions_to_traces_map[next_ftr] = transition_traces
    transitions2traces[ftr] = ftr_outgoing_transitions_to_traces_map


def update_ftr2ftr(ftr, ftr2ftrs, next_ftr):
    futures = ftr2ftrs.get(ftr, set())
    futures.add(next_ftr)
    ftr2ftrs[ftr] = futures


def update_ftr2pasts(ftr, past_ftr, ftr2past):
    pasts = ftr2past.get(ftr, set())
    pasts.add(past_ftr)
    ftr2past[ftr] = pasts


def apply_past_equivelance(ftr2equiv_classes, pasts_equiv):

    pasts2id = {}
    for ftr in pasts_equiv:
        pasts = tuple(sorted(list(pasts_equiv[ftr])))
        if pasts in pasts2id:
            ## is seen past equiv class in the past, use existing id
            ftr2equiv_classes[ftr] = pasts2id[pasts]
        else:
            ## otherwise, define equivalent state id
            pasts2id[pasts] = ftr2equiv_classes[ftr]


def construct_graph_from_futures(ftr2ftrs, states2transition2traces, use_traces_as_set = False, pasts_equiv = None):

    def get_edge_weight(label2traces, use_traces_as_set):
        trs = []
        for l in label2traces:
            trs.extend(label2traces[l])
        return len(trs) if not use_traces_as_set else len(set(trs))

    ## map each equiv class to an id
    ftr2equiv_classes = dict(map(lambda x: (x[1], x[0]), enumerate(ftr2ftrs)))
    ftr2equiv_classes[tuple()] = len(ftr2equiv_classes)
    if pasts_equiv:
        apply_past_equivelance(ftr2equiv_classes, pasts_equiv)
    g = nx.DiGraph()
    init_id = -1
    for ftr in ftr2equiv_classes:
        id = ftr2equiv_classes[ftr]
        if id in g.nodes():
            continue
        shape = ""
        if len(ftr) > 0 and ftr[0] == INIT_LABEL:
            shape = "doublecircle"
            if init_id == -1: ## keep single representitive for init state
                init_id = id
            id = init_id
            ftr2equiv_classes[ftr] = id
        if len(ftr) == 0:
            shape = "diamond"
        if shape:
            g.add_node(id, shape=shape)
        else:
            g.add_node(id)

    ## add transitions, labels, weights
    edges_dic = {}
    for ftr in ftr2ftrs:
        for ftr2 in ftr2ftrs[ftr]:

            tar_src = (ftr2equiv_classes[ftr], ftr2equiv_classes[ftr2])
            edge_data = edges_dic.get(tar_src)
            edge_label = tuple([ftr[0] if ftr else ""])
            edge_traces = states2transition2traces[ftr][ftr2]
            if edge_data is None:
                label2traces = {edge_label[0]: edge_traces.copy()}
                # w = len(label2traces) if not use_traces_as_set else len(set(label2traces))
                label_transitions_traces = []
                for l in label2traces:
                    label_transitions_traces.extend(label2traces[l])
                w = len(label_transitions_traces) if not use_traces_as_set else len(set(label_transitions_traces))
                edge_data = (edge_label, w, label2traces)
            else:
                if edge_data[0] != edge_label and not pasts_equiv:
                    raise AssertionError("two states are expected to be connected with "
                                         "a single labels, but two different appear!")
                label2traces = edge_data[2]
                if edge_label in label2traces:
                    label2traces[edge_label].extend(edge_traces)
                else:
                    label2traces[edge_label] = edge_traces
                w = get_edge_weight(label2traces, use_traces_as_set)
                edge_data = (edge_data[0], w, label2traces)
            edges_dic[tar_src] = edge_data

    for e, data in edges_dic.items():
        g.add_edge(e[0], e[1], label=data[0], weight=data[1], traces=data[2])
    return g

def normalized_transition_count_dict(transitions_count):

    for ftr in transitions_count:
        transitions = transitions_count.get(ftr)
        total_out_transition = sum(transitions, key=lambda x: len(x))
        validation_count = 0
        for tr in transitions:
            transitions[tr] = len(transitions[tr]) / float(total_out_transition)
            validation_count += len(transitions[tr])

        if not (0.99 < validation_count < 1.01):
            raise AssertionError("transition probabilities do not add to one in future:", ftr, validation_count)


def ktails(traces, k=2, graph_simplification=0, use_traces_as_set=False):
    ''' 0- no past simplification, 1- simplify by graph, 2 - simplify by ks '''
    ## generate equiv classes

    ftr2ftrs, states2transitions2traces = generate_equivalent_maps(traces, k, False)
    g = construct_graph_from_futures(ftr2ftrs, states2transitions2traces, use_traces_as_set=use_traces_as_set, pasts_equiv=None)
    return ftr2ftrs, states2transitions2traces, g



def write2file(g, path):

    def remove_attribute(G, tnode, attr):
        G.node[tnode].pop(attr, None)

    g = g.copy()
    for n in g.nodes():
        remove_attribute(g, n, "contraction")
    for e in g.edges:
        del g.get_edge_data(e[0], e[1])['traces']
    nx.drawing.nx_pydot.write_dot(g, path)

DOT_SUFFIX = ".dot"

def run(traces, k, graph_output ="", fname =""):

    total_events = sum(map(lambda t: len(t), traces))
    print("Done reading log, total traces/events: ", len(traces), "/", total_events)
    print("starting model construction phase")

    ## run k-tails
    ftr2ftrs, states2transitions2traces, g = ktails(traces, k=k)
    print("done building mode, graph has node/edges", len(g.nodes), len(g.edges))
    k_tag = "_k_" + str(k)
    if graph_output:
        write2file(g, graph_output + fname + k_tag + DOT_SUFFIX)
    return ftr2ftrs, states2transitions2traces, g

def change_tuples_to_list(log):
    '''
    changes the traces representation from list of tuples to list of lists
    :return: a list
    '''
    new_log = []
    for tr in log:
        new_log.append(list(tr))
    return new_log


if __name__ == '__main__':

    ## read log
    # k = 11
    # ks = [20, 40, 80]
    # ks = [1, 2, 3, 4, 6, 8, 10]
    LOG_SUFFIX = '.log'
    MODEL_SUFFIX = '_model.dot'
    LOG_PATH = '../../data/bear/findyourhouse_long.log'
    LOG_OUT_PATH = '../../data/bear/filtered_logs/'
    GRAPH_OUTPUT = "../../data/bear_models/bear_models"
    ks = [1]
    log_parser = BearLogParser(LOG_PATH)
    traces = log_parser.process_log(True)
    # log1_traces = log_parser.get_traces_of_browser(traces, "Mozilla/4.0")
    # log2_traces = log_parser.get_traces_of_browser(traces, "Mozilla/5.0")
    # log1_filename = 'mozzila4'
    # log2_filename = 'mozzila5'

    log1_filename = 'desktop'
    log2_filename = 'mobile'
    log1_traces = log_parser.get_desktop_traces(traces)
    log2_traces = log_parser.get_mobile_traces(traces)

    # events2keep = set(['search','sales_anncs',
    #                    'sales_page, facebook',
    #                    'sales_page, page_1',
    #                    'sales_page, page_2',
    #                    'sales_page, page_3',
    #                    'sales_page, page_4',
    #                    'sales_page, page_5',
    #                    'sales_page, page_6',
    #                    'sales_page, page_7',
    #                    'sales_page, page_8',
    #                    'sales_page, page_9',
    #                    ])
    # filter_traces_mozilla4 = log_parser.filter_events(events2keep, mozilla4_traces, True)
    # filter_traces_mozilla5 = log_parser.filter_events(events2keep, mozilla5_traces, True)

    new_name_mapping = {'sales_page, page_1': 'sales_page', 'sales_page, page_2': 'sales_page', 'sales_page, page_3': 'sales_page',
    'sales_page, page_4': 'sales_page', 'sales_page, page_5': 'sales_page', 'sales_page, page_6': 'sales_page',
    'sales_page, page_7': 'sales_page', 'sales_page, page_8': 'sales_page', 'sales_page, page_9': 'sales_page',
                        'renting_page, page_1': 'renting_page', 'renting_page, page_2': 'renting_page',
                        'contacts_requested': 'contact_requested'}

    filter_traces_log1 = log_parser.abstract_events(new_name_mapping, log1_traces)
    filter_traces_log2 = log_parser.abstract_events(new_name_mapping, log2_traces)

    log1_traces = log_parser.get_traces_as_lists_of_event_labels(filter_traces_log1)
    log2_traces = log_parser.get_traces_as_lists_of_event_labels(filter_traces_log2)

    from LogWriter import LogWriter
    LogWriter.write_log(log1_traces, LOG_OUT_PATH + log1_filename + LOG_SUFFIX)
    LogWriter.write_log(log2_traces, LOG_OUT_PATH + log2_filename + LOG_SUFFIX)
    # mozilla4_traces = change_tuples_to_list(mozilla4_traces)
    # mozilla5_traces = change_tuples_to_list(mozilla5_traces)
    # traces = log_parser.get_traces_as_lists_of_event_labels

    for k in ks:
        ftr2ftrs4, states2transitions2traces4, g4 = run(log1_traces, k, GRAPH_OUTPUT, log1_filename + MODEL_SUFFIX)
        ftr2ftrs5, states2transitions2traces5, g5 = run(log2_traces, k, GRAPH_OUTPUT, log2_filename + MODEL_SUFFIX)

        filtering_str = ""
        low_probability_filter = None  ##  0.05
        if low_probability_filter:
            print("FILTER APPLIED: low prob filter!")
            g4 = graph_filtering.filter_low_probability_transitions(g4, low_probability_filter)
            g5 = graph_filtering.filter_low_probability_transitions(g5, low_probability_filter)
            filtering_str += "_lp_" + str(low_probability_filter)

        simple_filter = 20
        if simple_filter:
            print("FILTER APPLIED: simple filter!")
            g4 = graph_filtering.simple_filter_graph(g4, simple_filter, False)
            g5 = graph_filtering.simple_filter_graph(g5, simple_filter, False)
            filtering_str += "_sim_" + str(simple_filter)

        write2file(g4, GRAPH_OUTPUT + log1_filename + filtering_str + '_k' + str(k) + DOT_SUFFIX)
        write2file(g5, GRAPH_OUTPUT + log2_filename + filtering_str + '_k' + str(k) + DOT_SUFFIX)
        print("done running with k=", k)

