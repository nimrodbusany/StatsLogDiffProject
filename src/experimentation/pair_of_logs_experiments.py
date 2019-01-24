import itertools

import pandas as pd

import src.statistical_diffs.statistical_log_diff_analyzer as sld
from bear_log_parser import *
from paired_experiment_results import Experiment_Result
from log_sampler import sample_traces
from log_based_mle import compute_mle_k_future_dict
from simple_log_parser import SimpleLogParser
from models.model_based_log_generator import LogGeneator


def get_logs(experiment_type, out_folder, bias=0.1, full_log_size=1000):
    if experiment_type == 0:
        LOG_PATH = '../../data/bear/findyourhouse_long.log'
        log_parser = BearLogParser(LOG_PATH)
        log_parser.process_log(True)
        mozilla4_traces = log_parser.get_traces_of_browser("Mozilla/4.0")
        mozilla4_traces = log_parser.get_traces_as_lists_of_event_labels(mozilla4_traces)
        mozilla5_traces = log_parser.get_traces_of_browser("Mozilla/5.0")
        mozilla5_traces = log_parser.get_traces_as_lists_of_event_labels(mozilla5_traces)
        experiment_name = "bear, mozilla"
        return mozilla4_traces, mozilla5_traces, experiment_name, out_folder + 'bear_pairwise/'

    if experiment_type == 1:
        LOG_PATH = '../../data/bear/filtered_logs/'
        mozilla4_traces = SimpleLogParser.read_log(LOG_PATH + 'desktop.log')
        mozilla5_traces = SimpleLogParser.read_log(LOG_PATH + 'mobile.log')
        experiment_name = "bear, desktop_mobile"
        return mozilla4_traces, mozilla5_traces, experiment_name, out_folder + 'bear_pairwise/'

    if experiment_type == 2:
        log1 = LogGeneator.produce_log_from_single_split_models(0.0, full_log_size)
        log2 = LogGeneator.produce_log_from_single_split_models(bias, full_log_size)
        return log1, log2, 'syn, single_split', out_folder + 'syn_pairwise/'

    if experiment_type == 3:
        log1, log2 = LogGeneator.produce_toy_logs(bias, full_log_size)
        return log1, log2, 'syn, toy', out_folder + 'syn_pairwise/'

    raise ValueError("experiment type: [0, 1]")


def print_diffs(diffs, outpath):
    with open(outpath, 'w') as fw:
        for d in diffs:
            fw.write(str(d) + ":" + str(diffs[d]) + "\n")

def log_based_experiments():

    ## statistical
    RESULT_FODLER = '../../results/statistical_experiments/'
    OUTPUT_ACTUAL_DIFFS = True
    EXPERIMENT_TYPE = 1
    ## Experiments main parameters
    ks = [2]
    min_diffs = [0.1]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    alphas = [0.05]  # [0.01, 0.05, 0.1, 0.2, 0.4]

    ## Repetition per configuration
    M = 5  # 10
    traces_to_sample = [500, 5000, 50000, 500000]  # [50, 500, 5000, 50000, 500000]

    experiment_results = Experiment_Result()
    log1, log2, experiment_name, outpath = get_logs(EXPERIMENT_TYPE, RESULT_FODLER)

    for (k, min_diff, alpha) in itertools.product(ks, min_diffs, alphas):
        dict1 = compute_mle_k_future_dict(log1, k)
        dict2 = compute_mle_k_future_dict(log2, k)
        ground_truth = [dict1, dict2]

        for sample in traces_to_sample:
            for trial in range(M):  ## repeat the experiment for m randomally selected logs
                sampled_log1 = sample_traces(log1, sample)
                sampled_log2 = sample_traces(log2, sample)
                alg = sld.SLPDAnalyzer(sampled_log1, sampled_log2)
                diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                if OUTPUT_ACTUAL_DIFFS:
                    vals = "_".join(['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 's_' + str(sample),
                                     't_' + str(trial)])
                    keys = list(diffs[list(diffs.keys())[0]].keys())
                    keys.extend(['source', 'target'])
                    df = pd.DataFrame(columns=keys)
                    for diff in diffs:
                        item = diffs[diff].copy()
                        item['source'] = diff[0]
                        item['target'] = diff[1]
                        df = df.append(item, ignore_index=True)
                    df.to_csv(outpath + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                    # print_diffs(diffs, RESULT_FODLER + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                ## filter_insignificant_diffs
                # significant_diffs = dict([(d, v) for d, v in diffs.items() if v['significant_diff'] is True])
                experiment_results.add_experiment_result(ground_truth, k, min_diff, -1, alpha, diffs, sample, sample)

    experiment_results.export_to_csv(
        outpath + 'ex_' + experiment_name + '_results' + '.csv')
    experiment_results.export_summary_to_csv(
        outpath + 'ex_' + experiment_name + '_results_summary' + '.csv')

def get_models(model_id, results_base_fodler, models2fetch, K):


    models = ['ctas.net.simplified.no_loops.dot', 'cvs.net.no_loops.dot', 'roomcontroller.simplified.net.dot',
              'ssh.net.simplified.dot', 'tcpip.simplified.net.dot', 'zip.simplified.dot']
    dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip', 'zip']
    MODELS_PATH = '../../models/stamina/'
    LOGS_OUTPUT_PATH = '../../data/logs/stamina/'
    results_base_fodler += dirs_[model_id] + "/"

    print('processing', models[model_id])
    models_transition_probs = []
    logs = []

    dir_ = LOGS_OUTPUT_PATH + dirs_[model_id] + "/"
    for j in range(models2fetch):
        ## read log
        log = SimpleLogParser.read_log(dir_+'l'+str(j)+'.log')
        logs.append(log)

        ## read model compute k_sequences transition probabilities
        transition_prob_path = dir_ + 'm'+str(j)+'_transitions_.csv'
        model_path = MODELS_PATH + models[model_id]
        from protocol_models_to_logs import ProtocolModel
        from ksequence_analytics import compute_k_sequence_transition_probabilities
        model = ProtocolModel(model_path, assign_transtion_probs=False)
        model.update_transition_probabilities(transition_prob_path)
        k_seqs2k_seqs = compute_k_sequence_transition_probabilities(model, K)
        models_transition_probs.append(k_seqs2k_seqs)

    return logs, models_transition_probs, models[model_id].split('.')[0], results_base_fodler

def model_based_experiments():

    ## statistical
    RESULT_FODLER = '../../results/statistical_experiments/model_based/pairwise/'
    OUTPUT_ACTUAL_DIFFS = True
    MODEL_ID = range(6)
    MODELS2FETCH = 2

    ## Experiments main parameters
    ks = [2]
    min_diffs = [0.01, 0.05, 0.1]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    alphas = [0.05]  # [0.01, 0.05, 0.1, 0.2, 0.4]

    ## Repetition per configuration
    M = 5  # 10
    traces_to_sample = [500, 5000, 50000, 100000]  # [50, 500, 5000, 50000, 500000]

    for model_id in MODEL_ID:
        experiment_results = Experiment_Result()
        for (k, min_diff, alpha) in itertools.product(ks, min_diffs, alphas):
            logs, models_transition_probs, experiment_name, outpath = get_models(model_id, RESULT_FODLER, MODELS2FETCH, k)
            import os
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            for sample_size in traces_to_sample:
                for trial in range(M):  ## repeat the experiment for m randomally selected logs
                    sampled_log = [sample_traces(log, sample_size) for log in logs]
                    alg = sld.SLPDAnalyzer(sampled_log[0], sampled_log[1])
                    diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                    if OUTPUT_ACTUAL_DIFFS:
                        vals = "_".join(['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 's_' + str(sample_size),
                                         't_' + str(trial)])
                        keys = list(diffs[list(diffs.keys())[0]].keys())
                        keys.extend(['source', 'target'])
                        df = pd.DataFrame(columns=keys)
                        for diff in diffs:
                            item = diffs[diff].copy()
                            item['source'] = diff[0]
                            item['target'] = diff[1]
                            item['true_m0'] = models_transition_probs[0].get(item['source'], 0)[item['target']]
                            item['true_m1'] = models_transition_probs[1].get(item['source'], 0)[item['target']]
                            item['true_diff'] = abs(item['true_m1'] - item['true_m0'])
                            item['statstical_success'] = 'ERR'
                            if not item['significant_diff']:
                                if item['pval'] == 'NA':
                                    item['statstical_success'] = 'NA'
                                elif item['true_diff'] > min_diff:
                                    item['statstical_success'] = False
                                elif item['true_diff'] < min_diff:
                                    item['statstical_success'] = True
                            else:
                                if item['true_diff'] > min_diff:
                                    item['statstical_success'] = True
                                if item['true_diff'] < min_diff:
                                    item['statstical_success'] = False

                            df = df.append(item, ignore_index=True)
                        df.to_csv(outpath + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                        # print_diffs(diffs, RESULT_FODLER + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                    ## filter_insignificant_diffs
                    # significant_diffs = dict([(d, v) for d, v in diffs.items() if v['significant_diff'] is True])
                    experiment_results.add_experiment_result(models_transition_probs, k, min_diff, -1, alpha, diffs, sample_size, sample_size)

        experiment_results.export_to_csv(
            outpath + 'ex_' + experiment_name + '_results' + '.csv')
        experiment_results.export_summary_to_csv(
            outpath + 'ex_' + experiment_name + '_results_summary' + '.csv')

if __name__ == '__main__':

    # log_based_experiments()
    model_based_experiments()