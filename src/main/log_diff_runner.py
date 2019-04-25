#!/usr/bin/python
import sys
import getopt, json, sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/../../')
import pandas as pd

from src.logs_parsers.simple_log_parser import SimpleLogParser
from src.statistical_diffs.statistical_log_diff_analyzer import MultipleSLPDAnalyzer
from src.graphs.diff_graph_overlay import *
from src.ktails.ktails import kTailsRunner
from src.utils.project_constants import *


def process_input_files(argv):
    '''
    process the input parameters
    :return:
    dict of logs ids to path; output_dir; alpha parameter; delta parameter
    '''
    inputfile, alpha, delta = '', '', ''
    try:
        opts, args = getopt.getopt(argv, "ha:d:c:k:r:", ["input_config=", "alpha=", "delta=", "k_param=", "r_param"])
    except getopt.GetoptError:
        print('error parsing input parameters, call: test.py -c <input_config> -a <alpha> -d <delta> -k <k_param>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -c <input_config> -a <alpha> -d <delta> -r <alg_to_run: 1:s2kdiff;2:snkdiff>')
            sys.exit()
        elif opt in ("-c", "--config"):
            inputfile = arg
        elif opt in ("-a", "--alpha"):
            try:
                alpha = float(arg)
            except:
                raise ValueError('alpha must be a float')
        elif opt in ("-d", "--delta"):
            try:
                delta = float(arg)
            except:
                raise ValueError('delta must be a float')
        elif opt in ("-k", "--k_param"):
            try:
                k = int(arg)
            except:
                raise ValueError('k must be a integer')
        elif opt in ("-r", "--r_param"):
            try:
                alg2run = int(arg)
                if alg2run not in [1, 2, 3]:
                    raise ValueError('wrong -alg_to_run parameter choose 1 for s2kdiff; 2 for snkdiff; 3 for k_tails')
            except:
                raise ValueError('wrong -alg_to_run parameter choose 1 for s2kdiff; and 2 for snkdiff; 3 for k_tails')

    with open(inputfile) as f:
        data = json.load(f)

    log_paths = {}
    for log in data["logs"]:

        if 'id' not in log:
            raise ValueError('missing id value in', log)
        if 'path' not in log:
            raise ValueError('missing path value in', log)
        if not os.path.exists(log['path']):
            raise ValueError('log file does not exists', log['id'], log['path'])
        if log['id'] in log_paths:
            raise ValueError('log id must be uniuqe, the following value appears more than once', log)

        log_paths[log['id']] = log['path']

    if 'output_dir' not in data:
        raise ValueError('output_dir missing form config_file')

    output_dir = data['output_dir']
    if not os.path.exists(output_dir):
        raise ValueError('output_dir does missing', output_dir)
    if not os.path.isdir(output_dir):
        raise ValueError('output_dir does map to a dir', output_dir)
    return log_paths, output_dir, alpha, delta, k, alg2run


def run(log_paths, output_dir, alpha, delta, k, alg2run):

    ### GET LOGS ###
    logs_dict = {}
    for log in log_paths:
        logs_dict[log] = SimpleLogParser.read_log(log_paths[log]) #TODO: remove this [:1000]
        print(log , len(logs_dict[log]), sum([1 for tr in logs_dict[log] for ev in tr]))

    if alg2run == 3:
        run_kTails(list(log_paths.values())[0], output_dir, k)
        return

    if len(logs_dict) < 2:
        raise ValueError("Please include at least two logs")

    ### COMPUTE DIFFS ###
    alg = MultipleSLPDAnalyzer(logs_dict)
    diffs = alg.find_statistical_diffs(k, delta, alpha)
    vals = "_".join(['k_' + str(k), 'd_' + str(delta), 'al_' + str(alpha)])
    keys = list(diffs[list(diffs.keys())[0]].keys())
    keys.extend([SOURCE_ATTR_NAME, TARGET_ATTR_NAME])
    df = pd.DataFrame(columns=keys)
    sig_diffs = []
    for diff in diffs:
        item = diffs[diff].copy()
        item[SOURCE_ATTR_NAME] = diff[0]
        item[TARGET_ATTR_NAME] = diff[1]
        if item[DIFFERENT_IDS_ATTR_NAME]:
            sig_diffs.append(item)
        df = df.append(item, ignore_index=True)


    ### Prepare all diffs file
    df[df.mutiple_proportion_test_significant == True]
    df = df.sort_values(by=STATISTICS_ATTR_NAME, ascending=False)
    df.to_csv(output_dir + 'results.csv', index=False)

    ### Prepare significant diffs file
    df = df[df[MUTIPLE_PROPORTION_TEST_SIGNIFICANT_ATTR_NAME] == True]
    df[PAIRWISE_COMPARISON_ATTR_NAME] = df[PAIRWISE_COMPARISON_ATTR_NAME].apply(
        func=lambda x: dict([diff for diff in x.items() if diff[1]['pvalue'] < alpha]))
    df[MAX_DIFF_ATTR_NAME] = df[PAIRWISE_COMPARISON_ATTR_NAME].apply(
        func=lambda x: max([diff['diff'] for diff in x.values()], default=0))

    df = df.sort_values(by=MAX_DIFF_ATTR_NAME, ascending=False)
    df = df[[MAX_DIFF_ATTR_NAME, SOURCE_ATTR_NAME, TARGET_ATTR_NAME, DIFFERENT_IDS_ATTR_NAME, PVALUE_ATTR_NAME, PAIRWISE_COMPARISON_ATTR_NAME]]
    df.to_csv(output_dir + 'significant_results.csv', index=False)

    ### Overlay significant diffs
    if len(sig_diffs) == 0:
        print('No differences found')
        return
    if alg2run == 1: s2kdiff_overlay(alg, k, logs_dict, output_dir, sig_diffs)
    if alg2run == 2: snkdiff_overlay(delta, k, logs_dict, output_dir, sig_diffs)


def find_most_interesting_diff_in_log(sig_diffs, first_log=True, by_pvalue=True):
    sig_diffs_in_log = {}
    proportion_attr = P1_ATTR_NAME if first_log else P2_ATTR_NAME
    for ind in range(len(sig_diffs)):
        pairwise_comparison = sig_diffs[ind][PAIRWISE_COMPARISON_ATTR_NAME]
        diff = next(iter(pairwise_comparison.values()))
        if diff[proportion_attr] > 0:
            sig_diffs_in_log[ind] = diff
    if by_pvalue:
        return min(sig_diffs_in_log.items(), key=lambda x: abs(x[1][PVALUE_ATTR_NAME]))[0]
    return max(sig_diffs_in_log.items(), key= lambda x: abs(x[1][P1_ATTR_NAME]-x[1][P2_ATTR_NAME]))[0]


def snkdiff_overlay(delta, k, logs_dict, output_dir, sig_diffs):
    single_log = []
    for log in logs_dict:
        single_log.extend(logs_dict[log])
    ktails_runner = kTailsRunner(single_log, k)
    ktails_runner.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
    g = ktails_runner.get_graph()
    g = overlay_differences_over_graph(g, sig_diffs, delta)
    write2file(g, output_dir + 'graph_nkdiff.dot')

def run_kTails(log_file, output_dir, k):

    log = SimpleLogParser.read_log(log_file)
    ktails_runner = kTailsRunner(log, k)
    ktails_runner.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
    ktails_runner.infer_transition_probabilities()
    ktails_runner.overlay_transition_probabilities_over_graph()
    ktails_runner.write2file(output_dir + 'graph_ktails.dot')

def s2kdiff_overlay(alg, k, logs_dict, output_dir, sig_diffs):
    k_tail_models = {}
    for log in logs_dict:
        ktails_runner = kTailsRunner(logs_dict[log], k)
        ktails_runner.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
        k_tail_models[log] = ktails_runner

    models_transitions_prob = [(log, k_tail_models[log].infer_transition_probabilities()) for log in k_tail_models]
    for log in logs_dict:

        diffs2covering_traces = alg.find_covering_traces(sig_diffs, models_transitions_prob[0], \
                                                         models_transitions_prob[1], logs_dict[log], k)
        model = k_tail_models[log]
        model.overlay_transition_probabilities_over_graph()
        diff_ind = find_most_interesting_diff_in_log(sig_diffs, first_log=True)
        print('overlying most significant diff over log :', log, '\n', sig_diffs[diff_ind])
        model.overlay_trace_over_graph(sig_diffs[diff_ind],
                                       diffs2covering_traces[diff_ind])  ## TODO: missing pvalue, bolden diff
        model.write2file(output_dir + 'graph_2kdiff_' + log + '.dot')
        covering_trace = logs_dict[log][diffs2covering_traces[diff_ind]]
        try:
            with open(output_dir + 'graph_2kdiff_' + log + '_covering_trace.txt', 'w') as fw:
                fw.write("\n".join(covering_trace))
        except:
            raise IOError('cannot write covering file')


def main(argv):

    log_paths, output_dir, alpha, delta, k, alg2run = process_input_files(argv)
    print('Input log files:')
    for log in log_paths:
        print(log, ':', log_paths[log])
    print('Output dir:', output_dir)
    print('Alpha =', alpha)
    print('Delta =', delta)
    print('k-parameter =', k)
    print('s2kdiff =', alg2run)

    run(log_paths, output_dir, alpha, delta, k, alg2run)

if __name__ == "__main__":
   main(sys.argv[1:])