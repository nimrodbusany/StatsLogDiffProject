import networkx as nx
from src.utils.project_constants import *

INIT_LABEL = 'init'
TERM_LABEL = 'term'
DOT_SUFFIX = ".dot"

class KSequence: ## TODO incorporate to code; make object hashable

    def __init__(self, events):
        self.events = events.copy()

class kTailsRunner:

    def __init__(self, traces, k):
        self.traces = traces.copy()
        self.k = k
        self.ftr2ftrs = {}
        self.ftr2past = {}
        self.states2transitions2traces = {}
        self.g = None
        self.state2transitions2prob = None

    def get_graph(self):
        return self.g

    def generate_equivalent_maps(self, gen_past=True, add_dummy_init=True, add_dummy_terminal=True):

        def _update_transitions2traces(ftr, next_ftr, tr_id):
            ## get state outgoing transitions
            ftr_outgoing_transitions_to_traces_map = self.states2transitions2traces.get(ftr, {})
            ## udpate outgoing transition visiting traces
            transition_traces = ftr_outgoing_transitions_to_traces_map.get(next_ftr, [])
            transition_traces.append(tr_id)
            ftr_outgoing_transitions_to_traces_map[next_ftr] = transition_traces
            self.states2transitions2traces[ftr] = ftr_outgoing_transitions_to_traces_map

        def _update_ftr2ftr(ftr, next_ftr):
            futures = self.ftr2ftrs.get(ftr, set())
            futures.add(next_ftr)
            self.ftr2ftrs[ftr] = futures

        def _update_ftr2pasts(ftr, past_ftr):
            pasts = self.ftr2past.get(ftr, set())
            pasts.add(past_ftr)
            self.ftr2past[ftr] = pasts

        self.ftr2ftrs = dict()
        self.ftr2past = dict()
        self.states2transitions2traces = dict()
        for tr_id in range(len(self.traces)):
            t = self.traces[tr_id]
            if add_dummy_init:
                t = tuple([INIT_LABEL] + list(t))
            if add_dummy_terminal:
                t = tuple(list(t) + [TERM_LABEL])
            for i in range(len(t)):
                # if i == 0: ## do not unify futures of dummy init
                #     ftr = tuple([t[0]])
                # else:
                ftr = t[i:i+self.k]
                next_ftr = t[i+1:i+self.k+1]
                ## update data strucutre
                _update_ftr2ftr(ftr, next_ftr)
                _update_transitions2traces(ftr, next_ftr, tr_id)
                if gen_past:
                    # past_ftr = t[max(0, i-self.k):i]
                    past_ftr = t[max(0,i-1):i+self.k-1]
                    _update_ftr2pasts(ftr, past_ftr)
            tr_id += 1

        if not gen_past:
            return self.ftr2ftrs, self.states2transitions2traces
        return self.ftr2ftrs, self.ftr2past, self.states2transitions2traces

    def apply_past_equivelance(self, ftr2equiv_classes, pasts_equiv):

        pasts2id = {}
        for ftr in pasts_equiv:
            pasts = tuple(sorted(list(pasts_equiv[ftr])))
            if pasts in pasts2id:
                ## is seen past equiv class in the past, use existing id
                ftr2equiv_classes[ftr] = pasts2id[pasts]
            else:
                ## otherwise, define equivalent state id
                pasts2id[pasts] = ftr2equiv_classes[ftr]

    def construct_graph_from_futures(self, use_traces_as_set = False, pasts_equiv = None):

        def get_edge_weight(label2traces, use_traces_as_set):
            trs = []
            for l in label2traces:
                trs.extend(label2traces[l])
            return len(trs) if not use_traces_as_set else len(set(trs))

        ## map each equiv class to an id
        self.ftr2equiv_classes = dict(map(lambda x: (x[1], x[0]), enumerate(self.ftr2ftrs)))
        self.ftr2equiv_classes[tuple()] = len(self.ftr2equiv_classes)
        if pasts_equiv:
            self.apply_past_equivelance(self.ftr2equiv_classes, pasts_equiv)
        g = nx.DiGraph()
        init_id = -1
        for ftr in self.ftr2equiv_classes:
            id = self.ftr2equiv_classes[ftr]
            if id in g.nodes():
                continue
            shape = ""
            if len(ftr) > 0 and ftr[0] == INIT_LABEL:
                shape = "doublecircle"
                if init_id == -1: ## keep single representitive for init state
                    init_id = id
                id = init_id
                self.ftr2equiv_classes[ftr] = id
            if len(ftr) == 0:
                shape = "diamond"
            if shape:
                g.add_node(id, shape=shape, label=ftr)
            else:
                g.add_node(id, label=ftr)

        self.ftr2transitions = {}
        ## add transitions, labels, weights
        edges_dic = {}
        for ftr in self.ftr2ftrs:
            for ftr2 in self.ftr2ftrs[ftr]:

                tar_src = (self.ftr2equiv_classes[ftr], self.ftr2equiv_classes[ftr2])
                self.ftr2transitions[(ftr, ftr2)] = tar_src
                edge_data = edges_dic.get(tar_src)
                edge_label = ftr[0] if ftr else ""
                edge_traces = self.states2transitions2traces[ftr][ftr2]
                if edge_data is None:
                    label2traces = {edge_label: edge_traces.copy()}
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

    def get_ftr_state_id(self, ftr):
        return self.ftr2transitions.get(ftr)

    def get_transition_state_id(self, ftr, next_ftr):
        return self.ftr2equiv_classes.get((ftr, next_ftr))

    def normalized_transition_count_dict(self, transitions_count):

        for ftr in transitions_count:
            transitions = transitions_count.get(ftr)
            total_out_transition = sum(transitions, key=lambda x: len(x))
            validation_count = 0
            for tr in transitions:
                transitions[tr] = len(transitions[tr]) / float(total_out_transition)
                validation_count += len(transitions[tr])

            if not (0.99 < validation_count < 1.01):
                raise AssertionError("transition probabilities do not add to one in future:", ftr, validation_count)


    def run_ktails(self, use_traces_as_set=False, add_dummy_init=True, add_dummy_terminal=True, graph_simplification=0):
        ''' 0- no past simplification, 1- simplify by graph, 2 - simplify by ks '''
        ## generate equiv classes
        if graph_simplification==0:
            self.generate_equivalent_maps(False, add_dummy_init, add_dummy_terminal)
            self.g = self.construct_graph_from_futures(use_traces_as_set=use_traces_as_set, pasts_equiv=None)
        else:
            ftr2ftrs, ftr2past, states2transitions2traces = self.generate_equivalent_maps(True, add_dummy_init, add_dummy_terminal)
            self.g = self.construct_graph_from_futures(use_traces_as_set=use_traces_as_set, pasts_equiv=ftr2past)
        return self.g.copy()

    def infer_transition_probabilities(self):

        if self.state2transitions2prob:
            return self.state2transitions2prob.copy()

        self.state2transitions2prob = {}
        for state in self.states2transitions2traces:
            visiting_traces = sum([len(self.states2transitions2traces[state][transition]) for transition in
                                   self.states2transitions2traces[state]])
            self.state2transitions2prob[state] = {}
            for transition in self.states2transitions2traces[state]:
                self.state2transitions2prob[state][transition] = len(self.states2transitions2traces[state][transition]) / float(
                    visiting_traces)
            total_prob = 0
            for transition in self.state2transitions2prob[state]:
                total_prob += self.state2transitions2prob[state][transition]
            if not (0.99 <= total_prob <= 1.01):
                raise ValueError("prob of outgoing transition must sum to 1, but sum to" + str(total_prob))
        return self.state2transitions2prob.copy()

    def reset_color_markings(self):
        colors = nx.get_edge_attributes(self.g, 'color')
        for edge in colors:
            colors[edge] = 'black'
        nx.set_edge_attributes(self.g, colors, 'color')

    def overlay_transition_probabilities_over_graph(self):

        labels = nx.get_edge_attributes(self.g, 'label')
        edge_penwidth = {}
        if not self.state2transitions2prob:
            self.infer_transition_probabilities()


        for state in self.state2transitions2prob:
            for transition in self.state2transitions2prob[state]:
                src = self.ftr2equiv_classes[state]
                trg = self.ftr2equiv_classes[transition]
                labels[(src, trg)] = str(labels[(src, trg)]) + " pr: "+ str(round(self.state2transitions2prob[state][transition], 2)) + "; vs: "+ str(len(self.states2transitions2traces[state][transition]))
                edge_penwidth[(src, trg)] = 1 + 2 * round(self.state2transitions2prob[state][transition], 2)

        nx.set_edge_attributes(self.g, labels, 'label')
        nx.set_edge_attributes(self.g, edge_penwidth, 'penwidth')

    def overlay_trace_over_graph(self, difference, trace_id):
        edge_colors = {}
        edge_labels = nx.get_edge_attributes(self.g, 'label')
        edge_traces = nx.get_edge_attributes(self.g, TRACES_ATTR_NAME)
        edge_style = {}

        ## mark diff transition
        source = tuple(ev for ev in difference[SOURCE_ATTR_NAME])
        target = tuple(ev for ev in difference[TARGET_ATTR_NAME])
        diff_edge = self.ftr2transitions[(source, target)]
        pairwise_comparison = difference[PAIRWISE_COMPARISON_ATTR_NAME]
        test_info = next(iter(pairwise_comparison.values()))
        edge_labels[diff_edge] = source[0] + " pr1/pr2:" + str(round(test_info[P1_ATTR_NAME], 2)) + ";" \
        + str(round(test_info[P2_ATTR_NAME], 2)) \
        +  "; vs1/vs2:" + str(test_info[M1_ATTR_NAME]) + ";" + str(test_info[M2_ATTR_NAME]) \
        + '\\npvalue' + str(round(difference[PVALUE_ATTR_NAME], 2))




        ## mark trace transitions
        for edge in edge_traces:
            traces_passing_edge = next(iter(edge_traces[edge].values()))
            if trace_id in traces_passing_edge:
                edge_colors[edge] = 'red'
                edge_style[edge] = "dashed"
        edge_style[diff_edge] = "solid"
        nx.set_edge_attributes(self.g, edge_colors, 'color')
        nx.set_edge_attributes(self.g, edge_labels, 'label')
        nx.set_edge_attributes(self.g, edge_style, 'style')


    def write2file(self, path):
        from src.graphs.diff_graph_overlay import write2file
        write2file(self.g, path)


    @staticmethod
    def run(traces, k, graph_output ="", fname =""):

        total_events = sum(map(lambda t: len(t), traces))
        print("Done reading log, total traces/events: ", len(traces), "/", total_events)
        print("starting model construction phase")

        ## run k-tails
        kTails_runner = kTailsRunner(traces, k)
        g = kTails_runner.run_ktails()
        print("done building mode, graph has node/edges", len(g.nodes), len(g.edges))
        k_tag = "_k_" + str(k)
        if graph_output:
            kTails_runner.write2file(graph_output + fname + k_tag + DOT_SUFFIX)
        return kTails_runner

    def _change_tuples_to_list(self):
        '''
        changes the traces representation from list of tuples to list of lists
        :return: a list
        '''
        new_log = []
        for tr in self.log:
            new_log.append(list(tr))
        return new_log


