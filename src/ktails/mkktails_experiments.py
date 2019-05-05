from src.logs_parsers.bear_log_parser import BearLogParser
from src.ktails.mk_ktails import kTailsRunner
from src.logs_parsers.simple_log_parser import SimpleLogParser
from src.automaton.automaton import check_relation
from PySimpleAutomata import DFA, NFA, automata_IO
from src.models.protocol_models_to_logs import ProtocolModel, TRANSITION_PROBABILITY_ATTRIBUTE
from src.logs.log_writer import LogWriter
from src.models.model_based_log_generator import LogGenerator
import networkx as nx
from time import time

from src.automaton.KanellakisAndSmolka import KanellakisAndSmolka, coarsen_graph
from PaigeAndTarjan import PaigeAndTarjan


def bear_based_experiments():

    ## read log
    # k = 11
    # ks = [20, 40, 80]
    # ks = [1, 2, 3, 4, 6, 8, 10]
    LOG_SUFFIX = '.log'
    MODEL_SUFFIX = '_model.dot'
    LOG_PATH = '../../data/bear/findyourhouse_long.log'
    LOG_OUT_PATH = '../../data/bear/filtered_logs/'
    GRAPH_OUTPUT = "../../data/bear_models/bear_models"
    ks = [1, 2, 3, 4]
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

    from log_writer import LogWriter
    LogWriter.write_log(log1_traces, LOG_OUT_PATH + log1_filename + LOG_SUFFIX)
    LogWriter.write_log(log2_traces, LOG_OUT_PATH + log2_filename + LOG_SUFFIX)
    # mozilla4_traces = change_tuples_to_list(mozilla4_traces)
    # mozilla5_traces = change_tuples_to_list(mozilla5_traces)
    # traces = log_parser.get_traces_as_lists_of_event_labels

    log1_traces_tups = []
    for tr in log1_traces:
        log1_traces_tups.append(tuple(tr))
    log1_traces = log1_traces_tups
    log2_traces_tups = []
    for tr in log2_traces:
        log2_traces_tups.append(tuple(tr))
    log2_traces = log2_traces_tups

    for k in ks:
        ktail_runner_4 = kTailsRunner(log1_traces, k)
        ktail_runner_5 = kTailsRunner(log2_traces, k)
        ktail_runner_4_past = kTailsRunner(log1_traces, k)
        ktail_runner_5_past = kTailsRunner(log2_traces, k)
        ktail_runner_4.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
        ktail_runner_5.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
        ktail_runner_4_past.run_ktails(add_dummy_init=False, add_dummy_terminal=False, graph_simplification=1)
        ktail_runner_5_past.run_ktails(add_dummy_init=False, add_dummy_terminal=False, graph_simplification=1)
        g4 = ktail_runner_4.get_graph()
        g5 = ktail_runner_5.get_graph()
        g4_past = ktail_runner_4_past.get_graph()
        g5_past = ktail_runner_5_past.get_graph()
        print(len(g4.nodes()), len(g4_past.nodes()), len(g5.nodes()), len(g5_past.nodes()))
        continue
        filtering_str = ""
        low_probability_filter = None  ##  0.05
        # if low_probability_filter:
        #     print("FILTER APPLIED: low prob filter!")
        #     g4 = graph_filtering.filter_low_probability_transitions(g4, low_probability_filter)
        #     g5 = graph_filtering.filter_low_probability_transitions(g5, low_probability_filter)
        #     filtering_str += "_lp_" + str(low_probability_filter)
        #
        # simple_filter = 20
        # if simple_filter:
        #     print("FILTER APPLIED: simple filter!")
        #     g4 = graph_filtering.simple_filter_graph(g4, simple_filter, False)
        #     g5 = graph_filtering.simple_filter_graph(g5, simple_filter, False)
        #     filtering_str += "_sim_" + str(simple_filter)

        ktail_runner_4.write2file(GRAPH_OUTPUT + log1_filename + filtering_str + '_k' + str(k) + DOT_SUFFIX)
        ktail_runner_5.write2file(GRAPH_OUTPUT + log2_filename + filtering_str + '_k' + str(k) + DOT_SUFFIX)
        print("done running with k=", k)


def log_based_experiments(log_path, ks, output_dir, model_name):

    traces = SimpleLogParser.read_log(log_path)
    traces_tups = []
    for tr in traces:
        traces_tups.append(tuple(tr))
    traces = traces_tups

    for k in ks:
        ktail_runner_ = kTailsRunner(traces, k)
        ktail_runner_past = kTailsRunner(traces, k)
        ktail_runner_.run_ktails(add_dummy_init=True, add_dummy_terminal=True, graph_simplification=0)
        ktail_runner_past.run_ktails(add_dummy_init=True, add_dummy_terminal=True, graph_simplification=1)
        g1 = ktail_runner_.get_graph()
        g2 = ktail_runner_past.get_graph()
        ktail_runner_.write2file(output_dir + model_name + '_' + str(k) + ".dot" )
        ktail_runner_past.write2file(output_dir + model_name + '_past_' + str(k) + ".dot")
        print(len(g1.nodes()), len(g2.nodes()), len(g1.edges()), len(g2.edges()))

def experiment_1():

    experiments = []
    # exp0 = ['../../data/logs/example/cvs/l0.log', [10], 'csv', "../../data/logs/example/cvs/ktails_models/"]
    # experiments.append(exp0)
    exp1 = ['../../data/bear/desktop.log', [5], 'desktop', "../../data/bear/ktails_models/"]
    experiments.append(exp1)
    exp2 = ['../../data/bear/mobile.log', [5], 'mobile', "../../data/bear/ktails_models/"]
    experiments.append(exp2)

    for log_path, ks, model_name, output_path in experiments:
        log_based_experiments(log_path, ks, output_path, model_name)


def simple_runner(traces, ks, k2s, output_dir, model_name, past_minimization=True, write_dot=False):

    ktail_runner_ = kTailsRunner(traces, ks, k2s)
    ktail_runner_.run_ktails(add_dummy_init=True, add_dummy_terminal=True, past_minimization=past_minimization)
    g1 = ktail_runner_.get_graph()
    if write_dot:
        ktail_runner_.write2file(output_dir + model_name + '_' + str(k) + ".dot")
    return g1

def minimize_nfa(G, path = "", PaigeMethod=True, check_equiv=True):

    for n1, n2, d in G.edges(data=True):
        d.pop('traces', None)

    ## to check, user third party module
    labels = set([x for x in nx.get_edge_attributes(G, 'label').values()])

    ## sainity check, in k-tail bisimluation does not change the model
    # blocks = k.compute_coarsest_partition(G)
    # non_singelton_blocks = [b for b in blocks if len(b) > 1]
    # if len(non_singelton_blocks):
    #     raise ValueError('bisumlation changed k-tails models', blocks)

    g1_r = G.reverse()
    if PaigeMethod:
        k = PaigeAndTarjan(labels)
        blocks = k.getCoarsestPartition(g1_r)
    else:
        k = KanellakisAndSmolka(labels)
        blocks = k.compute_coarsest_partition(g1_r)

    ## to check, user third party module
    g_abs = coarsen_graph(G, blocks)
    if path:
        nx.drawing.nx_pydot.write_dot(G, path + 'model_12.dot')
        nx.drawing.nx_pydot.write_dot(g_abs, path + 'model_minimized.dot')
        nfa1 = automata_IO.nfa_dot_importer(path + 'model_12.dot')
        nfa2 = automata_IO.nfa_dot_importer(path + 'model_minimized.dot')

    if check_equiv:
        print('isBisimilar:', k.isBisimilar(G, g_abs)) ## TODO: understand why graphs are not bisimular!
        print('IS EQUIV:', check_relation(nfa1, nfa2))
        if check_relation(nfa1, nfa2) != 0:
            raise ValueError('not equiv!!!!!')

    return g_abs


def remove_attributes(graph):
    for n1, d in graph.nodes(data=True):
        for att in ['label']:
            d.pop(att, None)
    for n1, n2, d in graph.edges(data=True):
        for att in ['traces', 'weight', 'key']:
            d.pop(att, None)

    for n1, n2, d in graph.edges(data=True):
        d['label'] = "\"" + ",".join([st for st in d['label']]) + "\""

            # str(d['label']).replace('(', '').replace(')', '') \
            # .replace('[', '').replace(']', '').replace(',', ', ')


def experiment_2(log_set):

    # LOGS SET 1
    if log_set == 1:
        BASE_DIR = '../../data/logs/example/mktails/paper_example/'
        LOG_PATH = BASE_DIR + 'mktails_log.log'
        traces = SimpleLogParser.read_log(LOG_PATH)
        OUTPUT_DIR = '../../data/logs/example/mktails/paper_example/'

        ks = [1, 2]
        k2s = {2: set(['lf'])}
        # k2s = {}
        ks_dict =[(ks, k2s)]

    # LOGS SET 2
    if log_set == 2:
        BASE_DIR = "../../data/bear/"
        LOG_PATH = BASE_DIR + 'desktop.log'
        traces = SimpleLogParser.read_log(LOG_PATH)
        OUTPUT_DIR = '../../data/logs/example/mktails/bear/'

    # LOGS SET 3
    if log_set == 3:
        model_path = "C:/Users/USER/PycharmProjects/StatsLogDiffProject/models/stamina/cvs.net.dot"
        model = ProtocolModel(model_path, id=0)
        log = LogGenerator.produce_log_from_model(model.graph, traces2produce=10000)
        LogWriter.write_log(log, "../../data/logs/example/mktails/csv/csv.log")
        traces = SimpleLogParser.read_log('../../data/logs/example/mktails/csv/csv.log')
        OUTPUT_DIR = '../../data/logs/example/mktails/csv/'
        ks = [3, 1]
        k2s = {3: set(['login', 'appendfile'])}
        ks_dict = [(ks, k2s)]

    # LOGS SET 4
    if log_set == 4:
        BASE_DIR = '../../data/logs/example/mktails/left_inv/'
        LOG_PATH = BASE_DIR + 'left_inv.log'
        traces = SimpleLogParser.read_log(LOG_PATH)
        OUTPUT_DIR = '../../data/logs/example/mktails/left_inv/'
        ks = [2]
        k2s = {2 : ['a', 'b', 'c', 'd', 'e']}
        ks_dict = [(ks, k2s)]


    # ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    # ks = [1]
    for (ks, k2s) in ks_dict:
        print('processing k -- ', ks, '--')
        print('running without past minimization')
        s_time_build = time()
        G = simple_runner(traces, ks, k2s, OUTPUT_DIR, "mktails", False)
        time_build = (time() - s_time_build) / 60
        for n in G.nodes(data=True):
            n[1].pop('label')
        print('original -   nodes, edges:', len(G.nodes()), len(G.edges))
        print('running with past minimization')
        time_paige = 1
        time_kanella = 1
        # s_time1 = time()
        # G_min1 = minimize_nfa(G.copy(), OUTPUT_DIR, PaigeMethod=True)
        # time_paige = ((time()-s_time1) / 60) + time_build
        s_time2 = time()
        G_min2 = minimize_nfa(G.copy(), OUTPUT_DIR, PaigeMethod=False)
        time_kanella = ((time()-s_time2) / 60) + time_build
        s_time3 = time()
        G_min3 = simple_runner(traces, ks, k2s, OUTPUT_DIR, "mktails", True)
        G_min1 = G_min2
        time_past_minimization = (time() - s_time3) / 60
        print('paige -      nodes, edges', len(G_min1.nodes()), len(G_min1.edges()))
        print('kanellakis - nodes, edges', len(G_min2.nodes()), len(G_min2.edges()))
        print('past -       nodes, edges', len(G_min3.nodes()), len(G_min3.edges()))
        try:
            print('time paige/kanella/ratio:', time_paige / time_past_minimization, time_kanella/ time_past_minimization)
        except:
            pass
        if True:
            remove_attributes(G)
            remove_attributes(G_min3)
            nx.drawing.nx_pydot.write_dot(G, OUTPUT_DIR + 'model.dot')
            nx.drawing.nx_pydot.write_dot(G_min1, OUTPUT_DIR + 'model_minimized_paige.dot')
            nx.drawing.nx_pydot.write_dot(G_min2, OUTPUT_DIR + 'model_minimized_kanellakis.dot')
            nx.drawing.nx_pydot.write_dot(G_min3, OUTPUT_DIR + 'model_minimized_past.dot')
            if False:
                nfa1 = automata_IO.nfa_dot_importer(OUTPUT_DIR + 'model.dot')
                nfa2 = automata_IO.nfa_dot_importer(OUTPUT_DIR + 'model_minimized_paige.dot')
                nfa3 = automata_IO.nfa_dot_importer(OUTPUT_DIR + 'model_minimized_past.dot')
                nfa4 = automata_IO.nfa_dot_importer(OUTPUT_DIR + 'model_minimized_kanellakis.dot')
                for min_nfa, name in [(nfa2, 'paige'), (nfa3, 'past'), (nfa4, 'kanellakis')]:
                    print(name, 'is equiv:', check_relation(nfa1, min_nfa) == 0)
                    if check_relation(nfa1, min_nfa) != 0:
                        raise ValueError('not equiv!!!!!')

        # simple_runner(traces, k, OUTPUT_DIR, "mktails_past", True)
        print('-----------------------')
        # exit()

if __name__ == '__main__':
    import random, numpy as np
    random.seed(a=1234567)
    np.random.seed(seed=1234567)
    experiment_2(log_set=1)
    # bear_based_experiments()
    # MODEL_NAME = 'csv'
    # LOG_PATH = '../../data/logs/example/cvs/l0.log'
    # GRAPH_OUTPUT = "../../data/logs/example/cvs/ktails_models/"
    # ks = [10]
    # log_based_experiments(LOG_PATH, ks, GRAPH_OUTPUT, MODEL_NAME)
    # experiment_1()
    # experiment_2()
