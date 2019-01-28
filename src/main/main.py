#!/usr/bin/python
import sys, getopt
import json
import os

from simple_log_parser import SimpleLogParser
import src.statistical_diffs.statistical_log_diff_analyzer as snk
import src.statistical_diffs.statistical_log_diff_analyzer as s2k
import pandas as pd


def process_input_files(argv):
    '''
    process the input parameters
    :return:
    dict of logs ids to path; output_dir; alpha parameter; delta parameter
    '''
    inputfile = ''
    alpha = ''
    delta = ''
    try:
        opts, args = getopt.getopt(argv, "ha:d:c:k:", ["input_config=", "alpha=", "delta=", "k_param="])
    except getopt.GetoptError:
        print('error parsing input parameters, call: test.py -c <input_config> -a <alpha> -d <delta> -k <k_param>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -c <input_config> -a <alpha> -d <delta>')
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
    return log_paths, output_dir, alpha, delta, k

def run(log_paths, output_dir, alpha, delta, k):

    def handle_multiple_logs_mapping(diff, logs_ids):
        ## map diffs to log indices
        if 'experiments_per_log' in diff:
            item['experiments_per_log'] = dict(
                [(logs_ids[i], diff['experiments_per_log'][i]) for i in
                 range(len(diff['experiments_per_log']))])

        ## map difference pairs to log indices
        if 'different_ids' in item:
            diff_tuples = diff['different_ids']
            item['different_ids'] = [(logs_ids[diff_tuples[i][0]], logs_ids[diff_tuples[i][1]]) for i in
                                            range(len(diff_tuples))]
        if 'significant_diffs' in item:
            pairwise_diffs = item['significant_diffs']
            new_pairwise_diffs = {}
            for pair in pairwise_diffs:
                pair_with_log_ids = (logs_ids[pair[0]], logs_ids[pair[1]])
                new_pairwise_diffs[pair_with_log_ids] = \
                    pairwise_diffs[pair]
            item['significant_diffs'] = new_pairwise_diffs

    logs = {}
    for log in log_paths:
        logs[log] = SimpleLogParser.read_log(log_paths[log])
    logs_ids = list(logs.keys())

    logs_arr = list(logs.values())
    alg = snk.MultipleSLPDAnalyzer(logs_arr) if len(logs) > 2 else s2k.SLPDAnalyzer(logs_arr[0], logs_arr[1])
    diffs = alg.find_statistical_diffs(k, delta, alpha)
    vals = "_".join(['k_' + str(k), 'd_' + str(delta), 'al_' + str(alpha)])
    keys = list(diffs[list(diffs.keys())[0]].keys())
    keys.extend(['source', 'target'])
    df = pd.DataFrame(columns=keys)
    for diff in diffs:
        item = diffs[diff].copy()
        handle_multiple_logs_mapping(diffs[diff], logs_ids)
        item['source'] = diff[0]
        item['target'] = diff[1]
        df = df.append(item, ignore_index=True)
    df.to_csv(output_dir + 'results' + "_" + vals + '_diffs' + '.csv')

def main(argv):

    log_paths, output_dir, alpha, delta, k = process_input_files(argv)
    print('Input log files:')
    for log in log_paths:
        print(log, ':', log_paths[log])
    print('Output dir:', output_dir)
    print('Alpha =', alpha)
    print('Delta =', delta)
    print('k-parameter =', k)

    run(log_paths, output_dir, alpha, delta, k)

if __name__ == "__main__":
   main(sys.argv[1:])