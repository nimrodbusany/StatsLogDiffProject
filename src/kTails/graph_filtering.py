import networkx as nx
import numpy as np


def find_init_terminal(G):

    init = set()
    term = set()
    for node, shape in nx.get_node_attributes(G, "shape").items():
        if shape == "doublecircle":
            init.add(node)
        if shape == "diamond":
            term.add(node)
    return init, term


def remove_non_reachable_states_from(G, init_states):

    reachable_states = set()
    for init_state in init_states:
        reachable_states.update(set(nx.descendants(G, init_state)))
        reachable_states.add(init_state)
    all_nodes = set(G.nodes)
    nodes2remove = all_nodes.difference(reachable_states)
    for n in nodes2remove:
        G.remove_node(n)


def remove_states_no_reaching(G, terminals):

    while True:
        nodes2remove = set()
        for n in G.nodes:
            if n in terminals:
                continue
            reachable_states = set(nx.descendants(G, n))
            if not terminals & reachable_states:
                nodes2remove.add(n)

        for n in nodes2remove:
            G.remove_node(n)
        if not nodes2remove:
            break


def remove_non_reacable_states(G):

    ## filter states not reachable from init or that do not reach terminal
    print("filtering non-reachable: ", len(G.nodes()), len(G.edges()))
    init_states, terminals = find_init_terminal(G)
    remove_non_reachable_states_from(G, init_states)
    print("nodes, edges - init filter: ", len(G.nodes()), len(G.edges()))
    remove_states_no_reaching(G, terminals)
    print("nodes, edges - terminal filter: ", len(G.nodes()), len(G.edges()))

    ## Sainity check: check that all states are still reachable from source
    # nodes_before_last_check = len(G.nodes())
    # remove_non_reachable_states_from(G, init_states)
    # nodes_after_last_check = len(G.nodes())
    # if nodes_before_last_check != nodes_after_last_check:
    #     raise ValueError("failed filtering test!")
    return G


def simple_filter_graph(G, threshold, precentage = True):

    ## remove low weight edges
    thr_val = threshold
    if precentage:
        weights = []
        for edge_data in G.edges.data():
            weights.append(edge_data[2]["weight"])

        weights = np.array(weights)
        thr_val = np.percentile(weights, threshold)

    edges2remove = []
    for edge_data in G.edges.data():
        w = edge_data[2]["weight"]
        if w <= thr_val:
            edges2remove.append(edge_data)
    for edge in edges2remove:
        G.remove_edge(edge[0], edge[1])
    print("weak edges to remove:", len(edges2remove))
    return remove_non_reacable_states(G)


def remove_weakest_edges(G, cmp):

    edges = []
    for edge_data in G.edges.data():
        if edge_data[0] in cmp and edge_data[1] in cmp:
            edges.append(edge_data)

    min_edge = min(edges, key= lambda x: x[2]["weight"])
    min_weight = min_edge[2]["weight"]
    edges2remove = []
    for edge in edges:
        w = edge[2]["weight"]
        if w == min_weight:
            G.remove_edge(edge[0], edge[1])
            edges2remove.append(edge)
    return edges2remove

def filter_low_probability_transitions(G, min_prob_threshold, max_occurances_to_filter = None):
    '''

    :param G: graph
    :param threshold: filter transition if has less than $ probabitliy to occur
    :param max_occurances_to_filter: only filter transition with less than $ occurrences
    :return: filtered graph
    '''
    nodes_before_filter = len(G.nodes())
    edges_before_filter = len(G.edges())
    for n in G.nodes():
        edges = G.edges(n, data=True)
        outgoing_edges = list(filter(lambda e: e[0] == n, edges))
        total_out = sum([e[2]["weight"] for e in outgoing_edges])
        trans_2_prob = []
        for e in outgoing_edges:
            trans_2_prob.append((e, e[2]["weight"]/total_out, e[2]["weight"]))
        # probabilities = np.array([e[1] for e in trans2prob])
        # thr_val = np.percentile(probabilities, min_prob_threshold)
        outgoing_edges_2_filter = \
            list(filter(lambda e: e[1] < min_prob_threshold, trans_2_prob))
        if max_occurances_to_filter:
            outgoing_edges_2_filter = \
                list(filter(lambda e: e[2] < max_occurances_to_filter, trans_2_prob))
        for e in outgoing_edges_2_filter:
            G.remove_edge(e[0][0], e[0][1])
    G = remove_non_reacable_states(G)
    nodes_after_filter = len(G.nodes())
    edges_after_filter = len(G.edges())
    print ("filtering kept ", nodes_after_filter, "nodes from", nodes_before_filter)
    print("filtering kept ", edges_after_filter, "edges from", edges_before_filter)
    return G

def connected_components_filter_graph(DG, threshold= 150):


    ## the idea is to break to graph to small connencted components by pilling away edges
    ## problem: connectivity between init and exit is not maintained
    G = DG.to_undirected()
    edges2remove = []
    stop = False
    while not stop:

        components = nx.connected_components(G)
        stop = True
        for cmp in components:
            if len(cmp) > threshold:
                edges2remove.extend(remove_weakest_edges(G, cmp))
                stop = False

    for edge in edges2remove:
        try:  ## In G we lose directions, must try in both direction
            DG.remove_edge(edge[0], edge[1])
        except:
            DG.remove_edge(edge[1], edge[0])
    return remove_non_reacable_states(DG)


# def generate_traces2transitions_maps(edges_dic):
#
#     traces2transitions = {}
#     transitions2traces = {}
#     for src_state, trg_state in edges_dic:
#         # for trg_state in states2transitions2traces[src_state]:
#         for tr in edges_dic[(src_state, trg_state)][2]:
#
#                 transition = (src_state, trg_state)
#                 ## add transition to trace list
#                 visited_transitions = traces2transitions.get(tr, set())
#                 visited_transitions.add(transition)
#                 traces2transitions[tr] = visited_transitions
#
#                 ## add traces to transition list
#                 visiting_traces = transitions2traces.get(transition,set())
#                 visiting_traces.add(tr)
#                 transitions2traces[transition] = visiting_traces
#     return transitions2traces, traces2transitions


def generate_traces2transitions_maps(g):

    transitions2traces = nx.get_edge_attributes(g, "traces").copy()
    transitions2traces_modified_dict = {}
    for trans in transitions2traces:
        trs = set()
        for trs_ in transitions2traces[trans]:
            trs.update(set(transitions2traces[trans][trs_]))
        transitions2traces_modified_dict[trans] = trs

    traces2transitions = {}
    for transition, traces in transitions2traces_modified_dict.items():
        for trace in traces:
            trace_transitions = traces2transitions.get(trace, [])
            trace_transitions.append(transition)
            traces2transitions[trace] = trace_transitions
    return transitions2traces_modified_dict, traces2transitions


def coverage_based_filter_graph(g, accuracy = 0.95):

    print("starting filtering, nodes and edges before: ", len(g.nodes()), len(g.edges()))
    transitions2traces, traces2transitions = generate_traces2transitions_maps(g)

    total_traces = len(traces2transitions)
    traces2remove = total_traces * (1 - accuracy)

    transition2remove = set()
    removed_traces = set()

    while len(removed_traces) < traces2remove:

        transition = min(transitions2traces, key=lambda x: len(transitions2traces[x]))
        traces2remove_set = transitions2traces.pop(transition)

        ## update other transitions
        for other_tran in transitions2traces:
            transitions2traces[other_tran].difference_update(traces2remove_set)
            if len(transitions2traces[other_tran]) == 0:
                transition2remove.add(other_tran)

        ## be conservative, only remove if not exceeding the required coverage
        if len(removed_traces) < traces2remove:
            transition2remove.add(transition)
            removed_traces.update(traces2remove_set)

    for trans in transition2remove:
        g.remove_edge(trans[0], trans[1])

    nx.set_edge_attributes(g, transitions2traces, "traces")
    traces_left = set()
    for e in transitions2traces:
        traces_left.update(transitions2traces[e])
    print("traces left:", len(traces_left))
    assert(len(traces_left) + len(removed_traces) == total_traces)

    print ("total of ", len(removed_traces), " traces removed:", removed_traces)
    print("finished filtering, nodes and edges after: ", len(g.nodes()), len(g.edges()))
    g = remove_non_reacable_states(g)
    for e in g.edges():
        g[e[0]][e[1]]['weight'] = len(transitions2traces[e])
    return g


def most_simplifying_coverage_based_filter_graph(DG, edges_dic, accuracy = 0.95):
    pass