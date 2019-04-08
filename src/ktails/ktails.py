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
        ftr2equiv_classes_cpy = ftr2equiv_classes.copy()
        import time
        start = time.time()
        print('starting past minimization')
        self.apply_past_equivelance_v1(ftr2equiv_classes, pasts_equiv)
        end = time.time()
        print('done running v1', (end - start))
        print('   ---    ')
        start2 = time.time()
        self.apply_past_equivelance_v2(ftr2equiv_classes_cpy, pasts_equiv)
        end2 = time.time()
        print('done running v2', (end2 - start2))
        if abs(len(set(ftr2equiv_classes.values())) - len(set(ftr2equiv_classes_cpy.values()))) > self.k:
            raise Exception("BAD IMPLEMENTATION, YOU FOOL!!!! "  + str(len(set(ftr2equiv_classes.values()))) + " - " +
                        str(len(set(ftr2equiv_classes_cpy.values()))))

    def apply_past_equivelance_v1(self, ftr2equiv_classes, pasts_equiv):

        state_backard_transitions = {}
        state_forward_transitions = {}
        state2id = {}
        total_states, total_transitions = len(self.ftr2ftrs), 0
        starting_states = set()
        for ftr_state in self.ftr2ftrs:
            if ftr_state[0] == 'init':
                starting_states.add(ftr_state)
            for next_state in self.ftr2ftrs[ftr_state]:
                state_backard_transitions.setdefault(ftr2equiv_classes[next_state], set()).add(
                    (ftr2equiv_classes[ftr_state], ftr_state[0]))
                state_forward_transitions.setdefault(ftr2equiv_classes[ftr_state], set()).add(
                    (ftr2equiv_classes[next_state], ftr_state[0]))
                state2id[ftr2equiv_classes[ftr_state]] = ftr2equiv_classes[ftr_state]
                state2id[ftr2equiv_classes[next_state]] = ftr2equiv_classes[next_state]
                total_transitions += 1
        for ftr_state in starting_states:
            state_backard_transitions.setdefault(ftr2equiv_classes[ftr_state], set()).add(
                (0, ftr_state[0]))
            state_forward_transitions.setdefault(0, set()).add(
                (ftr2equiv_classes[ftr_state], ftr_state[0]))
        max_id = max(state2id) + 1
        states_queue = set(state2id.keys())
        states_queue.remove(0) ## HACK: remove dummy initial node with not past, find nicer solution
        minimization_iterations, total_equiv_states_across_rounds = 0, 0
        states_reads = 0
        states_seend_coutners = {}
        while states_queue:
            states_reads += len(states_queue)
            minimization_iterations += 1
            past_equiv_classes = {}
            for state in states_queue:
                states_seend_coutners[state] = states_seend_coutners.get(state, 0) + 1
                state_incoming_transitions = frozenset((state2id[q], a) for (q, a) in state_backard_transitions[state])
                past_equiv_classes.setdefault(state_incoming_transitions, set()).add(state)
            states_queue = set()
            for equiv_class in past_equiv_classes.values():
                equiv_id = max_id
                if len(equiv_class) > 1 and len(set([state2id[st] for st in equiv_class])) > 1:
                    total_equiv_states_across_rounds += 1
                    for equiv_state in equiv_class:
                        state2id[equiv_state] = equiv_id
                        states_queue.update([tup[0] for tup in state_forward_transitions[equiv_state]])
                    max_id = max_id + 1

        for state in ftr2equiv_classes:
            ftr2equiv_classes[state] = state2id[ftr2equiv_classes[state]]
        print('total states, transitions:', total_states, total_transitions)
        print('total iterations, states_reads, total equiv states:', minimization_iterations, states_reads, total_equiv_states_across_rounds, max_id)
        print('max state visits', max(states_seend_coutners.values())) ## , [states_seend_coutners[st] for st in states_seend_coutners if states_seend_coutners[st] > 1]

    def apply_past_equivelance_v2(self, ftr2equiv_classes, pasts_equiv):


        state_back_transitions = {}
        starting_states = set()
        for ftr_state in self.ftr2ftrs:
            if ftr_state[0] == 'init':
                starting_states.add(ftr_state)
            for next_state in self.ftr2ftrs[ftr_state]:
                state_back_transitions.setdefault(ftr2equiv_classes[next_state], set()).add((ftr2equiv_classes[ftr_state], ftr_state[0]))

        for ftr_state in starting_states:
            state_back_transitions.setdefault(ftr2equiv_classes[ftr_state], set()).add(
                (0, ftr_state[0]))

        max_id = max(ftr2equiv_classes.values()) + 1
        state2id = dict([(v, v) for v in ftr2equiv_classes.values()])
        iterate = True
        past_equiv_classes = {}
        minimization_iterations, total_equiv_states_across_rounds, seen_states = 0 , 0, 0
        while iterate:
            minimization_iterations+=1
            prev_equiv_classes = past_equiv_classes.copy()
            iterate = False
            for state in state_back_transitions:
                state_incoming_transitions = frozenset((state2id[q], a) for (q, a) in state_back_transitions[state])
                past_equiv_classes.setdefault(state_incoming_transitions, set()).add(state)
            seen_states += len(state_back_transitions)
            for equiv_state in past_equiv_classes:
                if past_equiv_classes.get(equiv_state, set()) != prev_equiv_classes.get(equiv_state, set()):
                    if len(past_equiv_classes[equiv_state]) > 1 and len(set([state2id[st] for st in past_equiv_classes[equiv_state]])) > 1:
                        total_equiv_states_across_rounds+=1
                        max_id += 1
                        for state in past_equiv_classes[equiv_state]:
                            state2id[state] = max_id
                        iterate = True


        for state in ftr2equiv_classes:
            ftr2equiv_classes[state] = state2id[ftr2equiv_classes[state]]
        print('total iterations, total equiv states, seen_states', minimization_iterations, total_equiv_states_across_rounds, seen_states)

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
        g = nx.MultiDiGraph()
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

        ## add transitions, labels, weights
        edges_dic = {}
        for ftr in self.ftr2ftrs:
            for ftr2 in self.ftr2ftrs[ftr]:

                tar_src = (self.ftr2equiv_classes[ftr], self.ftr2equiv_classes[ftr2])
                edge_data = edges_dic.get(tar_src)
                edge_label = tuple([ftr[0] if ftr else ""])
                edge_traces = self.states2transitions2traces[ftr][ftr2]
                if edge_data is None:
                    label2traces = {edge_label[0]: edge_traces.copy()}
                    label_transitions_traces = []
                    for l in label2traces:
                        label_transitions_traces.extend(label2traces[l])
                    w = len(label_transitions_traces) if not use_traces_as_set else len(set(label_transitions_traces))
                    edge_data = (edge_label, w, label2traces)
                else:
                    label2traces = edge_data[2]
                    if edge_label in label2traces:
                        label2traces[edge_label].extend(edge_traces)
                    else:
                        label2traces[edge_label] = edge_traces
                    w = get_edge_weight(label2traces, use_traces_as_set)
                    new_labels_arr = []
                    new_labels_arr.extend(edge_data[0])
                    new_labels_arr.extend(edge_label)
                    edge_data = (list(set(new_labels_arr)), w, label2traces)

                edges_dic[tar_src] = edge_data

        id = 0
        for e, data in edges_dic.items():
            id+=1
            g.add_edge(e[0], e[1], id, label=data[0], weight=data[1], traces=data[2])
        return g


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
                labels[(src, trg)] = str(labels[(src, trg)]) + " p="+ str(round(self.state2transitions2prob[state][transition], 2))
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
        edge_labels[diff_edge] = source[0] + " p1, p2:" + str(round(test_info[P1_ATTR_NAME], 2)) + ", " \
        + str(round(test_info[P2_ATTR_NAME], 2)) + '\\npvalue' + str(round(difference[PVALUE_ATTR_NAME], 2))


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
        from diff_graph_overlay import write2file
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


