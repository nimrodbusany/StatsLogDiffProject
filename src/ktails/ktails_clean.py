import networkx as nx
from src.utils.project_constants import *

INIT_LABEL = 'init'
TERM_LABEL = 'term'
DOT_SUFFIX = ".dot"
VERBOSE = True
class KSequence: ## TODO incorporate to code; make object hashable

    def __init__(self, events):
        self.events = events.copy()

class kTailsRunner:

    def __init__(self, traces, k):
        self.traces = traces.copy()
        self.k = k
        self.ftr2ftrs = {}
        self.states2transitions2traces = {}
        self.g = None
        self.state2transitions2prob = None

    def get_graph(self):
        return self.g

    def generate_equivalent_maps(self, add_dummy_init=True, add_dummy_terminal=True):

        def update_transitions2traces(ftr, next_ftr, tr_id):
            ## get state outgoing transitions
            ftr_outgoing_transitions_to_traces_map = self.states2transitions2traces.get(ftr, {})
            ## udpate outgoing transition visiting traces
            transition_traces = ftr_outgoing_transitions_to_traces_map.get(next_ftr, [])
            transition_traces.append(tr_id)
            ftr_outgoing_transitions_to_traces_map[next_ftr] = transition_traces
            self.states2transitions2traces[ftr] = ftr_outgoing_transitions_to_traces_map

        def update_ftr2ftr(ftr, next_ftr):
            futures = self.ftr2ftrs.get(ftr, set())
            futures.add(next_ftr)
            self.ftr2ftrs[ftr] = futures

        self.ftr2ftrs = dict()
        self.states2transitions2traces = dict()
        for tr_id in range(len(self.traces)):
            t = self.traces[tr_id]
            if add_dummy_init:
                t = tuple([INIT_LABEL] + list(t))
            if add_dummy_terminal:
                t = tuple(list(t) + [TERM_LABEL])
            for i in range(len(t)):
                ftr = t[i:i+self.k]
                next_ftr = t[i+1:i+self.k+1]
                ## update data strucutre
                update_ftr2ftr(ftr, next_ftr)
                update_transitions2traces(ftr, next_ftr, tr_id)
            tr_id += 1

        return self.ftr2ftrs, self.states2transitions2traces

    def apply_past_equivelance(self):

        ## helper function: maps each state to its incoming transitions
        def flip_future_dict(ftr2equiv_classes):
            incoming_transitions_dict = {}
            for ftr_state in self.ftr2ftrs:
                for next_state in self.ftr2ftrs[ftr_state]:
                    incoming_transitions_dict.setdefault(ftr2equiv_classes[next_state], set()).add(
                        (ftr2equiv_classes[ftr_state], ftr_state[0]))
            return incoming_transitions_dict

        ## helper function: maps each state to its incoming transitions
        def flip_future_dict(ftr2equiv_classes):
            incoming_transitions_dict = {}
            for ftr_state in self.ftr2ftrs:
                for next_state in self.ftr2ftrs[ftr_state]:
                    incoming_transitions_dict.setdefault(ftr2equiv_classes[next_state], set()).add(
                        (ftr2equiv_classes[ftr_state], ftr_state[0]))
            return incoming_transitions_dict

        ## helper function: computes the past of a block
        def compute_block_past(concrete_states, incoming_transitions_dict):
            state_incoming_transitions = set()
            for concrete_state in concrete_states:
                incoming_transitions = frozenset(
                    (states2blocks[q], a) for (q, a) in incoming_transitions_dict.get(concrete_state, []))
                state_incoming_transitions = state_incoming_transitions.union(incoming_transitions)
            return frozenset(state_incoming_transitions)

        ftr2equiv_id = self.ftr2equiv_classes.copy()
        ## input: ftr2ftr, ftr2equiv_id
        # create auxiliary dictionaries, used for fast extraction
        equiv_classes2ftr = {ftr2equiv_id[ftr]: ftr for ftr in ftr2equiv_id} ## maps each equivalence state id to its future
        block_count = 0

        #  STEP 1: map each state to a single state block
        blocks2states = {} ## id -> equiv states,  maps id to a block of states
        states2blocks = {} ## Q -> maps each state to block id
        for ftr_id in equiv_classes2ftr.keys(): ## initially place every state on its own block
            block_id = 'b' + str(block_count)
            blocks2states[block_id] = set([ftr_id])
            states2blocks[ftr_id] = block_id
            block_count += 1

        incoming_transitions_dict = flip_future_dict(ftr2equiv_id)  ## maps each state to its incoming transitions
        states_queue = set(blocks2states.keys()) ## all states must be examined
        ## variables collected for stats
        if VERBOSE: minimization_iterations, new_states, states_reads, states_seen_coutners = 0, 0, 0, {}

        while True: ## loop until the blocks do not change

            if VERBOSE: states_reads += len(states_queue)
            if VERBOSE: minimization_iterations += 1

            past_equiv_classes = {} ## compute a dictionary of past equivalence classes of blocks: map pasts (set of block transitions) -> blocks (that share the past)
            for block in blocks2states: ## update blocks past_dict
                state_incoming_transitions = compute_block_past(blocks2states[block], incoming_transitions_dict)
                incoming_transitions_dict[block] = state_incoming_transitions
                past_equiv_classes.setdefault(state_incoming_transitions, set()).add(block)
                if VERBOSE: states_seen_coutners[block] = states_seen_coutners.get(block, 0) + 1

            prev_block_count = block_count
            for block_union in past_equiv_classes.values():
                if len(block_union) > 1:
                    block_count += 1
                    block_id = 'b' + str(block_count)
                    equiv_states = blocks2states.setdefault(block_id, set())
                    for block in block_union:
                        for concrete_state in blocks2states[block]:
                            equiv_states.add(concrete_state)
                            states2blocks[concrete_state] = block_id
                        blocks2states.pop(block)
                    if VERBOSE: new_states += 1

            if prev_block_count == block_count:
                break

        for block in blocks2states:
            for concrete_state in blocks2states[block]:
                ftr = equiv_classes2ftr[concrete_state]
                ftr2equiv_id[ftr] = block

        total_equiv_classes_after_reduction = len(set(ftr2equiv_id.values()))
        if VERBOSE:
            print('total iterations, states_reads, new_states, total equiv states, max state visits:', minimization_iterations,
                  states_reads, new_states, total_equiv_classes_after_reduction, max(states_seen_coutners.values()))

        if total_equiv_classes_after_reduction - len(set(self.ftr2equiv_classes.values())) > 0:
            raise Exception("Past minimization increased number of equivalence classes. This must be a bug!!!! "
                            + str(len(set(ftr2equiv_id.values()))) + " - " + str(len(set(self.ftr2equiv_classes.values()))))

        self.ftr2equiv_classes = ftr2equiv_id

    def construct_graph_from_futures(self, use_traces_as_set = False, past_minimization= False):

        def get_edge_weight(label2traces, use_traces_as_set):
            trs = []
            for l in label2traces:
                trs.extend(label2traces[l])
            return len(trs) if not use_traces_as_set else len(set(trs))

        ## map each equiv class to an id
        self.ftr2equiv_classes = dict(map(lambda x: (x[1], x[0]), enumerate(self.ftr2ftrs)))
        self.ftr2equiv_classes[tuple()] = len(self.ftr2equiv_classes)

        ## unify equiv class that share an id
        if past_minimization:
            self.apply_past_equivelance()

        ## construct graph according to equivalence classes
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


    def run_ktails(self, use_traces_as_set=False, add_dummy_init=True, add_dummy_terminal=True, past_minimization=False):
        self.generate_equivalent_maps(add_dummy_init, add_dummy_terminal)
        self.g = self.construct_graph_from_futures(use_traces_as_set=use_traces_as_set, past_minimization=past_minimization)
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


