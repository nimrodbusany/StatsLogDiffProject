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


        ftr2equiv_classes = self.ftr2equiv_classes.copy()
        incoming_transitions_dict = {}  ## maps each state to its incoming transitions
        outoing_transitions_dict = {}  ## maps each state to its outgoing transitions
        state2id = {}  ## Q -> Equiv_id maps each state to id of equivelance class

        for ftr_state in self.ftr2ftrs:  ## iterator over the future transitions maps,
            for next_state in self.ftr2ftrs[ftr_state]:
                incoming_transitions_dict.setdefault(ftr2equiv_classes[next_state], set()).add(
                    (ftr2equiv_classes[ftr_state], ftr_state[0]))
                state2id[ftr2equiv_classes[ftr_state]] = ftr2equiv_classes[ftr_state]
                state2id[ftr2equiv_classes[next_state]] = ftr2equiv_classes[next_state]

        max_id = max(state2id) + 1  ## new equivalence classes will recieve a higher id
        states_queue = set(state2id.keys())  ## all states must be examined
        states_queue.remove(0)  ## HACK: remove dummy initial node with not past, find nicer solution
        ## variables collected for stats
        if VERBOSE: minimization_iterations, new_states, states_reads, states_seen_coutners = 0, 0, 0, {}

        while states_queue:  ## while there are states to examine

            if VERBOSE: states_reads += len(states_queue)
            if VERBOSE: minimization_iterations += 1

            past_equiv_classes = {}
            for state in states_queue:  ## update past_dict
                state_incoming_transitions = frozenset((state2id[q], a) for (q, a) in incoming_transitions_dict.get(state, set()))
                past_equiv_classes.setdefault(state_incoming_transitions, set()).add(state)
                if VERBOSE: states_seen_coutners[state] = states_seen_coutners.get(state, 0) + 1

            states_queue = set()
            prev_max = max_id
            for equiv_class in past_equiv_classes.values():
                equiv_id = max_id
                if len(set([state2id[st] for st in equiv_class])) > 1:
                    for equiv_state in equiv_class:
                        state2id[equiv_state] = equiv_id
                    max_id += 1
                    if VERBOSE: new_states += 1

            states_queue = set(state2id.keys())
            if prev_max == max_id:
                break

        for state in ftr2equiv_classes:
            self.ftr2equiv_classes[state] = state2id[ftr2equiv_classes[state]]

        if VERBOSE:
            total_equiv_classes_after_reduction = len(set(ftr2equiv_classes.values()))
            print('total iterations, states_reads, new_states, total equiv states, max state visits:',
                  minimization_iterations,
                  states_reads, new_states, total_equiv_classes_after_reduction, max(states_seen_coutners.values()))



    # def apply_past_equivelance_optimized(self):
    #
    #
    #     ftr2equiv_classes = self.ftr2equiv_classes.copy()
    #     incoming_transitions_dict = {}  ## maps each state to its incoming transitions
    #     outoing_transitions_dict = {}  ## maps each state to its outgoing transitions
    #     state2id = {}  ## Q -> Equiv_id maps each state to id of equivelance class
    #
    #     for ftr_state in self.ftr2ftrs:  ## iterator over the future transitions maps,
    #         for next_state in self.ftr2ftrs[ftr_state]:
    #             incoming_transitions_dict.setdefault(ftr2equiv_classes[next_state], set()).add(
    #                 (ftr2equiv_classes[ftr_state], ftr_state[0]))
    #             outoing_transitions_dict.setdefault(ftr2equiv_classes[ftr_state], set()).add(
    #                 (ftr2equiv_classes[next_state], ftr_state[0]))
    #             state2id[ftr2equiv_classes[ftr_state]] = ftr2equiv_classes[ftr_state]
    #             state2id[ftr2equiv_classes[next_state]] = ftr2equiv_classes[next_state]
    #
    #     max_id = max(state2id) + 1  ## new equivalence classes will recieve a higher id
    #     states_queue = set(state2id.keys())  ## all states must be examined
    #     states_queue.remove(0)  ## HACK: remove dummy initial node with not past, find nicer solution
    #     K_BLOCK_MINIMIZATION = False
    #     ## variables collected for stats
    #     if VERBOSE: minimization_iterations, new_states, states_reads, states_seen_coutners = 0, 0, 0, {}
    #
    #     while states_queue:  ## while there are states to examine
    #
    #         if VERBOSE: states_reads += len(states_queue)
    #         if VERBOSE: minimization_iterations += 1
    #
    #         past_equiv_classes = {}
    #         for state in states_queue:  ## update past_dict
    #             state_incoming_transitions = frozenset((state2id[q], a) for (q, a) in incoming_transitions_dict.get(state, set()))
    #             past_equiv_classes.setdefault(state_incoming_transitions, set()).add(state)
    #             if VERBOSE: states_seen_coutners[state] = states_seen_coutners.get(state, 0) + 1
    #
    #         states_queue = set()
    #         prev_max = max_id
    #         for equiv_class in past_equiv_classes.values():
    #             equiv_id = max_id
    #             if len(set([state2id[st] for st in equiv_class])) > 1:
    #                 for equiv_state in equiv_class:
    #                     state2id[equiv_state] = equiv_id
    #                     if K_BLOCK_MINIMIZATION: states_queue.update(
    #                         [tup[0] for tup in outoing_transitions_dict[equiv_state]])
    #                 max_id += 1
    #                 if VERBOSE: new_states += 1
    #
    #         if not K_BLOCK_MINIMIZATION:  ## if no states were merged
    #             states_queue = set(state2id.keys())
    #             if prev_max == max_id:
    #                 break
    #
    #     for state in ftr2equiv_classes:
    #         ftr2equiv_classes[state] = state2id[ftr2equiv_classes[state]]
    #
    #     total_equiv_classes_after_reduction = len(set(ftr2equiv_classes.values()))
    #     if VERBOSE:
    #         print('total iterations, states_reads, new_states, total equiv states, max state visits:',
    #               minimization_iterations,
    #               states_reads, new_states, total_equiv_classes_after_reduction, max(states_seen_coutners.values()))
    #
    #     if total_equiv_classes_after_reduction - len(set(self.ftr2equiv_classes.values())) > 0:
    #         raise Exception("Past minimization increased number of equivalence classes. This must be a bug!!!! "
    #                         + str(len(set(ftr2equiv_classes.values()))) + " - " + str(
    #             len(set(self.ftr2equiv_classes.values()))))
    #
    #     self.ftr2equiv_classes = ftr2equiv_classes

    # def apply_past_equivelance_v2(self):
    #
    #     ## helper function: maps each state to its incoming transitions
    #     # def flip_future_dict(ftr2equiv_classes):
    #     #     incoming_transitions_dict = {}
    #     #     for ftr_state in self.ftr2ftrs:
    #     #         for next_state in self.ftr2ftrs[ftr_state]:
    #     #             incoming_transitions_dict.setdefault(ftr2equiv_classes[next_state], set()).add(
    #     #                 (ftr2equiv_classes[ftr_state], ftr_state[0]))
    #     #     return incoming_transitions_dict
    #
    #     ## helper function: computes the past of a block
    #     # def compute_block_past(concrete_states, incoming_transitions_dict):
    #     #     state_incoming_transitions = set()
    #     #     for concrete_state in concrete_states:
    #     #         incoming_transitions = frozenset(
    #     #             (states2blocks[q], a) for (q, a) in incoming_transitions_dict.get(concrete_state, []))
    #     #         state_incoming_transitions = state_incoming_transitions.union(incoming_transitions)
    #     #     return frozenset(state_incoming_transitions)
    #
    #     ftr2equiv_id = self.ftr2equiv_classes.copy()
    #     ## input: ftr2ftr, ftr2equiv_id
    #     # create auxiliary dictionaries, used for fast extraction
    #     equiv_classes2ftr = {ftr2equiv_id[ftr]: ftr for ftr in ftr2equiv_id} ## maps each equivalence state id to its future
    #     block_count = 0
    #
    #     #  STEP 1: map each state to a single state block
    #     blocks2states = {} ## id -> equiv states,  maps id to a block of states
    #     states2blocks = {} ## Q -> maps each state to block id
    #     for ftr_id in equiv_classes2ftr.keys(): ## initially place every state on its own block
    #         block_id = 'b' + str(block_count)
    #         blocks2states[block_id] = set([ftr_id])
    #         states2blocks[ftr_id] = block_id
    #         block_count += 1
    #
    #     blocks_incoming_transitions_dict = {}
    #     blocks_outgoing_transitions_dict = {}
    #
    #     for state in states2blocks:
    #         cur_block_id = states2blocks[state]
    #         ftr_state = equiv_classes2ftr[state]
    #         for next_state_ftr in self.ftr2ftrs.get(ftr_state, tuple()):
    #             next_state_ftr_id = ftr2equiv_id[next_state_ftr]
    #             next_block = states2blocks[next_state_ftr_id]
    #             blocks_incoming_transitions_dict.setdefault(next_block, set()).add(
    #                 (cur_block_id, ftr_state[0]))
    #             blocks_outgoing_transitions_dict.setdefault(cur_block_id, set()).add(
    #                 (next_block, ftr_state[0]))
    #
    #     ## variables collected for stats
    #     if VERBOSE: minimization_iterations, new_states, blocks_created, states_seen_coutners = 0, 0, 0, {}
    #
    #     past_equiv_classes = {}  ## compute a dictionary of past equivalence classes of blocks: map pasts (set of block transitions) -> blocks (that share the past)
    #     for block in blocks2states:  ## update blocks past_dict
    #         block_incoming_transitions = frozenset(blocks_incoming_transitions_dict.get(block, set()))
    #         past_equiv_classes.setdefault(block_incoming_transitions, set()).add(block)
    #         if VERBOSE: states_seen_coutners[block] = states_seen_coutners.get(block, 0) + 1
    #
    #     blocks2unify = [(block_past, block_states) for block_past, block_states in past_equiv_classes.items() if len(block_states) > 1]
    #
    #     while len(blocks2unify) > 0: ## loop until the blocks do not change
    #
    #         if VERBOSE: blocks_created = block_count
    #         if VERBOSE: minimization_iterations += 1
    #         print('iteration', minimization_iterations)
    #         blocks_past, block_states = blocks2unify.pop(0) ## remove first group of blocks
    #         block_states = block_states.copy()
    #         if len(block_states) == 0:
    #             continue
    #
    #         ## init new block
    #         block_count += 1
    #         union_block_id = 'b' + str(block_count)
    #         union_blocks_states = set()
    #         union_block_outgoing_transitions = blocks_outgoing_transitions_dict.setdefault(union_block_id, set())
    #         union_block_incoming_transitions = blocks_incoming_transitions_dict.setdefault(union_block_id, set())
    #         blocks2states[union_block_id] = union_blocks_states
    #
    #         for block in block_states: ## remove each of the unified blocks, and update past_equiv_dict
    #
    #             # past_equiv_classes[frozenset(union_block_incoming_transitions)].remove(block)
    #             try:
    #                 for concrete_state in blocks2states.pop(block): ## add block state to the union block
    #                     union_blocks_states.add(concrete_state)
    #             except KeyError:
    #                 print(block, blocks_past)
    #                 print('imashaa')
    #             block_incoming_transitions = blocks_incoming_transitions_dict.pop(block)
    #             block_outgoing_transitions = blocks_outgoing_transitions_dict.pop(block) ## remove block from outgoing transition dict
    #
    #             for incoming_transition in block_incoming_transitions: union_block_incoming_transitions.add(incoming_transition) ## TODO: can be removed - double check
    #             for outgoing_transition in block_outgoing_transitions: union_block_outgoing_transitions.add(outgoing_transition) ## add outgoing_transition to outgoing transition dict[union_state]
    #
    #             for outgoing_transition in block_outgoing_transitions:
    #
    #                 target_block = outgoing_transition[0]
    #                 target_block_incoming_transitions = blocks_incoming_transitions_dict.get(target_block, set()) ## TODO: skipping block in loop? should reprocess like the rest?
    #
    #                 if len(target_block_incoming_transitions) == 0:
    #                     continue
    #
    #                 for incomoing_transition in target_block_incoming_transitions:
    #                     if incomoing_transition[0] == block:
    #                         transition2update = incomoing_transition
    #                         break
    #
    #                 past_blocks = past_equiv_classes.get(frozenset(target_block_incoming_transitions))
    #
    #                 if past_blocks:
    #                     if len(past_blocks) == 1:
    #                         past_equiv_classes.pop(frozenset(target_block_incoming_transitions))
    #                     else:
    #                         if target_block in past_blocks:
    #                             past_blocks.remove(target_block)
    #                 target_block_incoming_transitions.remove(transition2update)
    #                 target_block_incoming_transitions.add((union_block_id, transition2update[1]))
    #                 updated_past_ = frozenset(target_block_incoming_transitions)
    #                 past_states = past_equiv_classes.setdefault(updated_past_, set())  ## TODO: remove set if empty!
    #                 past_states.add(target_block)
    #                 # if len(past_states) == 2: ## if > 2, was added by someone else!
    #                 #     blocks2unify.append((updated_past_, past_states))
    #
    #             for prev_transition in block_incoming_transitions:
    #                 prev_block = prev_transition[0]
    #                 transition2update = None
    #                 prev_block_outgoing_transitions = blocks_outgoing_transitions_dict.get(prev_block, set())
    #                 if len(prev_block_outgoing_transitions) == 0: ## TODO: skipping block in loop? should reprocess like the rest?
    #                     continue
    #
    #                 for transition in prev_block_outgoing_transitions:
    #                     if transition[0] == block:
    #                         transition2update = transition
    #                         break
    #                 blocks_outgoing_transitions_dict[prev_block].remove(transition2update)
    #                 blocks_outgoing_transitions_dict[prev_block].add((union_block_id, transition2update[1]))
    #
    #             if VERBOSE: new_states += 1
    #             # if block in past_equiv_classes[frozenset(blocks_past)]:
    #             #     past_equiv_classes[frozenset(blocks_past)].remove(block)
    #             # else:
    #             #     print(block, blocks_past)
    #
    #         past_equiv_classes[frozenset(blocks_past)] = set([union_block_id])
    #         blocks2unify = [(block_past, block_states) for block_past, block_states in past_equiv_classes.items() if
    #                         len(block_states) > 1]
    #
    #     for block in blocks2states:
    #         for concrete_state in blocks2states[block]:
    #             ftr = equiv_classes2ftr[concrete_state]
    #             ftr2equiv_id[ftr] = block
    #
    #     total_equiv_classes_after_reduction = len(set(ftr2equiv_id.values()))
    #     states_reads = 1
    #     if VERBOSE:
    #         print('total iterations, states_reads, new_states, total equiv states, max state visits:', minimization_iterations,
    #               states_reads, new_states, total_equiv_classes_after_reduction, max(states_seen_coutners.values()))
    #
    #     if total_equiv_classes_after_reduction - len(set(self.ftr2equiv_classes.values())) > 0:
    #         raise Exception("Past minimization increased number of equivalence classes. This must be a bug!!!! "
    #                         + str(len(set(ftr2equiv_id.values()))) + " - " + str(len(set(self.ftr2equiv_classes.values()))))
    #
    #     self.ftr2equiv_classes = ftr2equiv_id


    # def apply_past_equivelance_v1(self, ftr2equiv_classes):
    #
    #     state_backard_transitions = {}
    #     state_forward_transitions = {}
    #     state2id = {}
    #     total_states, total_transitions = len(self.ftr2ftrs), 0
    #     starting_states = set()
    #     for ftr_state in self.ftr2ftrs:
    #         if ftr_state[0] == 'init':
    #             starting_states.add(ftr_state)
    #         for next_state in self.ftr2ftrs[ftr_state]:
    #             state_backard_transitions.setdefault(ftr2equiv_classes[next_state], set()).add(
    #                 (ftr2equiv_classes[ftr_state], ftr_state[0]))
    #             state_forward_transitions.setdefault(ftr2equiv_classes[ftr_state], set()).add(
    #                 (ftr2equiv_classes[next_state], ftr_state[0]))
    #             state2id[ftr2equiv_classes[ftr_state]] = ftr2equiv_classes[ftr_state]
    #             state2id[ftr2equiv_classes[next_state]] = ftr2equiv_classes[next_state]
    #             total_transitions += 1
    #     for ftr_state in starting_states:
    #         state_backard_transitions.setdefault(ftr2equiv_classes[ftr_state], set()).add(
    #             (0, ftr_state[0]))
    #         state_forward_transitions.setdefault(0, set()).add(
    #             (ftr2equiv_classes[ftr_state], ftr_state[0]))
    #     max_id = max(state2id) + 1
    #     states_queue = set(state2id.keys())
    #     states_queue.remove(0)  ## HACK: remove dummy initial node with not past, find nicer solution
    #     minimization_iterations, total_equiv_states_across_rounds = 0, 0
    #     states_reads = 0
    #     states_seend_coutners = {}
    #     while states_queue:
    #         states_reads += len(states_queue)
    #         minimization_iterations += 1
    #         past_equiv_classes = {}
    #         for state in states_queue:
    #             states_seend_coutners[state] = states_seend_coutners.get(state, 0) + 1
    #             state_incoming_transitions = frozenset((state2id[q], a) for (q, a) in state_backard_transitions[state])
    #             past_equiv_classes.setdefault(state_incoming_transitions, set()).add(state)
    #         states_queue = set()
    #         for equiv_class in past_equiv_classes.values():
    #             equiv_id = max_id
    #             if len(equiv_class) > 1 and len(set([state2id[st] for st in equiv_class])) > 1:
    #                 total_equiv_states_across_rounds += 1
    #                 for equiv_state in equiv_class:
    #                     state2id[equiv_state] = equiv_id
    #                     states_queue.update([tup[0] for tup in state_forward_transitions[equiv_state]])
    #                 max_id = max_id + 1
    #
    #     for state in ftr2equiv_classes:
    #         ftr2equiv_classes[state] = state2id[ftr2equiv_classes[state]]
    #     print('total states, transitions:', total_states, total_transitions)
    #     print('total iterations, states_reads, total equiv states:', minimization_iterations, states_reads,
    #           total_equiv_states_across_rounds, max_id)
    #     print('max state visits', max(
    #         states_seend_coutners.values()))  ## , [states_seend_coutners[st] for st in states_seend_coutners if states_seend_coutners[st] > 1]

    def construct_graph_from_futures(self, use_traces_as_set = False, past_minimization= False):

        def get_edge_weight(label2traces, use_traces_as_set):
            trs = []
            for l in label2traces:
                trs.extend(label2traces[l])
            return len(trs) if not use_traces_as_set else len(set(trs))

        ## map each equiv class to an id
        self.ftr2equiv_classes = dict(map(lambda x: (x[1], x[0]), enumerate(self.ftr2ftrs)))
        self.ftr2equiv_classes[tuple()] = len(self.ftr2equiv_classes)

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


