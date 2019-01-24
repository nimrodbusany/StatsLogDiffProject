import itertools

import pandas as pd

import src.statistical_diffs.statistical_log_diff_analyzer as sld
from bear_log_parser import *
from multi_log_experiment_results import Experiment_Result
from log_sampler import sample_traces
from log_based_mle import compute_mle_k_future_dict
from simple_log_parser import SimpleLogParser
from models.model_based_log_generator import LogGeneator


def get_logs(experiment_type, out_folder, full_log_size= 1000, biases = [0, 0.1, 0.1]):

    def duplicates_to_full_log_size(traces, target_size):
        while len(traces) < target_size:
            traces.extend(traces)

    logs = []
    if experiment_type == 0:
        LOG_PATH = '../../data/bear/findyourhouse_long.log'
        log_parser = BearLogParser(LOG_PATH)
        log_parser.process_log(True)
        mozilla4_traces = log_parser.get_traces_of_browser("Mozilla/4.0")
        mozilla4_traces = log_parser.get_traces_as_lists_of_event_labels(mozilla4_traces)
        duplicates_to_full_log_size(mozilla4_traces, full_log_size)
        mozilla5_traces = log_parser.get_traces_of_browser("Mozilla/5.0")
        mozilla5_traces = log_parser.get_traces_as_lists_of_event_labels(mozilla5_traces)
        duplicates_to_full_log_size(mozilla5_traces, full_log_size)
        # mozilla5_2_traces = log_parser.get_traces_of_browser("Mozilla/5.0")
        # mozilla5_2_traces = log_parser.get_traces_as_lists_of_event_labels(mozilla5_traces)
        experiment_name = "bear, mozilla"
        logs.extend([mozilla4_traces, mozilla4_traces, mozilla5_traces, mozilla5_traces, ])
        return logs , experiment_name, out_folder + 'bear_multiple_logs_0/'

    if experiment_type == 1:
        LOG_PATH = '../../data/bear/'
        mozilla_desktop = SimpleLogParser.read_log(LOG_PATH + 'desktop.log')
        mozilla_mobile = SimpleLogParser.read_log(LOG_PATH + 'mobile.log')
        duplicates_to_full_log_size(mozilla_desktop, full_log_size)
        duplicates_to_full_log_size(mozilla_mobile, full_log_size)
        experiment_name = "bear, desktop_mobile"
        logs.extend([mozilla_desktop, mozilla_desktop, mozilla_mobile, mozilla_mobile])
        return logs, experiment_name, out_folder + 'bear_multiple_logs/'

    if experiment_type == 2:
        for bias in biases:
            logs.append(LogGeneator.produce_log_from_single_split_models(bias, full_log_size))
        return logs, 'syn, single_split', out_folder + 'syn_multiple_logs/'

    if experiment_type == 3:
        for bias in biases:
            logs.append(LogGeneator.produce_toy_logs(bias, full_log_size)[1])
        return logs, 'syn, toy', out_folder + 'syn_multiple_logs2/'


    raise ValueError("experiment type: [0, 1, 2, 3]")

def print_diffs(diffs, outpath):
    with open(outpath, 'w') as fw:
        for d in diffs:
            fw.write(str(d) + ":" + str(diffs[d]) + "\n")

CHI_SQUARE_BASED = False


def log_based_experiments():
    ## statistical
    RESULT_FODLER = '../../results/statistical_experiments/'
    OUTPUT_ACTUAL_DIFFS = True
    EXPERIMENT_TYPE = 1
    CHI_SQUARE_BASED = False

    WRITE_SINGLE_EXPERIMENT_RESULTS = True
    ## Experiments main parameters
    ks = [2]
    min_diffs = [0.01, 0.05, 0.1, 0.2]  # , 0.05, 0.1, 0.2, 0.4] #[0.01, 0.05, 0.1, 0.2, 0.4]
    alphas = [0.05, 0.1, 0.15]  # [0.01, 0.05, 0.05, 0.1, 0.2, 0.4]

    ## Repetition per configuration

    M = 3  # 10
    full_log_size = 10000
    traces_to_sample = [10000]  # [50, 500, 5000, 50000, 100000, 500000]
    biases_array = [(0,)]  # (0, 0.1, -0.1)
    for k in ks:
        for biases in biases_array:
            experiment_results = Experiment_Result(biases)
            logs, experiment_name, outpath = get_logs(EXPERIMENT_TYPE, RESULT_FODLER, full_log_size=full_log_size,
                                                      biases=biases)
            ground_truth = []
            for log in logs:
                ground_truth.append(compute_mle_k_future_dict(log, k))

        for (min_diff, alpha) in itertools.product(min_diffs, alphas):

            for sample in traces_to_sample:
                for trial in range(M):  ## repeat the experiment for m randomally selected logs
                    sampled_logs = [sample_traces(log, sample) for log in logs]
                    alg = sld.MultipleSLPDAnalyzer(sampled_logs)
                    diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                    if OUTPUT_ACTUAL_DIFFS:
                        bs = str(biases).replace(',', '_')
                        vals = "_".join(
                            ['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 'bs_' + bs, 's_' + str(sample),
                             't_' + str(trial)])
                        keys = list(diffs[list(diffs.keys())[0]].keys())
                        keys.extend(['source', 'target'])
                        df = pd.DataFrame(columns=keys)
                        for diff in diffs:
                            item = diffs[diff].copy()
                            item['source'] = diff[0]
                            item['target'] = diff[1]
                            df = df.append(item, ignore_index=True)
                        if WRITE_SINGLE_EXPERIMENT_RESULTS:
                            df.to_csv(outpath + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                            # print_diffs(diffs, RESULT_FODLER + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                    ## filter_insignificant_diffs
                    # significant_diffs = dict([(d, v) for d, v in diffs.items() if v['significant_diff'] is True])
                    experiment_results.add_experiment_result(ground_truth, k, min_diff, biases, alpha, diffs, sample,
                                                             full_log_size)

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


def _enrich_diff_with_experiment_info(MODELS2FETCH, diff, item, min_diff, models_transition_probs):
    item['source'] = diff[0]
    item['target'] = diff[1]
    for ind in range(MODELS2FETCH):
        item['true_m' + str(ind)] = models_transition_probs[ind].get(item['source'], 0)[item['target']]
    for ind1, ind2 in itertools.product(range(MODELS2FETCH), range(MODELS2FETCH)):
        if ind1 >= ind2:
            continue
        real_diff = abs(models_transition_probs[ind1].get(item['source'], 0)[item['target']] -
                        models_transition_probs[ind2].get(item['source'], 0)[item['target']])
        item['true_diff' + str(ind1) + '_' + str(ind2)] = real_diff
        if real_diff > min_diff and (ind1, ind2) in item['different_ids'] or \
                                real_diff < min_diff and (ind1, ind2) not in item['different_ids']:
            item[str(ind1) + '_' + str(ind2) + '_ok'] = True
        else:
            item[str(ind1) + '_' + str(ind2) + '_ok'] = False


def model_based_experiments():

    ## statistical
    RESULT_FODLER = '../../results/statistical_experiments/model_based/multiple_logs/'
    OUTPUT_ACTUAL_DIFFS = True
    MODELS2FETCH = 4
    CHI_SQUARE_BASED = False

    WRITE_SINGLE_EXPERIMENT_RESULTS = True
    ## Experiments main parameters
    ks = [2]
    min_diffs = [0.01, 0.05, 0.1, 0.2]  # , 0.05, 0.1, 0.2, 0.4] #[0.01, 0.05, 0.1, 0.2, 0.4]
    alphas = [0.05, 0.1, 0.15]  # [0.01, 0.05, 0.05, 0.1, 0.2, 0.4]

    ## Repetition per configuration
    MODEL_IDS = range(5)
    M = 5  # 10
    full_log_size = 100000
    traces_to_sample = [50, 500, 1000, 5000, 10000, 50000, 100000]  # [50, 500, 5000, 50000, 100000, 500000]
    biases = (0,)  # (0, 0.1, -0.1)
    for model_id in MODEL_IDS:
        experiment_results = Experiment_Result(biases)
        for k in ks:
            logs, models_transition_probs, experiment_name, outpath = get_models(model_id, RESULT_FODLER, MODELS2FETCH, k)
            import os
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            for (min_diff, alpha) in itertools.product(min_diffs, alphas):

                for sample in traces_to_sample:
                    for trial in range(M):  ## repeat the experiment for m randomally selected logs
                        sampled_logs = [sample_traces(log, sample) for log in logs]
                        alg = sld.MultipleSLPDAnalyzer(sampled_logs)
                        diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                        if OUTPUT_ACTUAL_DIFFS:
                            vals = "_".join(
                                ['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 's_' + str(sample),
                                 't_' + str(trial)])
                            keys = list(diffs[list(diffs.keys())[0]].keys())
                            keys.extend(['source', 'target'])
                            df = pd.DataFrame(columns=keys)
                            for diff in diffs:
                                item = diffs[diff].copy()
                                _enrich_diff_with_experiment_info(MODELS2FETCH, diff, item, min_diff, models_transition_probs)
                                df = df.append(item, ignore_index=True)

                            if WRITE_SINGLE_EXPERIMENT_RESULTS:
                                df.to_csv(outpath + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                                # print_diffs(diffs, RESULT_FODLER + 'ex_' + experiment_name + "_" + vals + '_diffs' + '.csv')
                        ## filter_insignificant_diffs
                        # significant_diffs = dict([(d, v) for d, v in diffs.items() if v['significant_diff'] is True])
                        experiment_results.add_experiment_result(models_transition_probs, k, min_diff, (0,), alpha, diffs, sample,
                                                                 full_log_size)

        experiment_results.export_to_csv(
            outpath + 'ex_' + experiment_name + '_results' + '.csv')
        experiment_results.export_summary_to_csv(
            outpath + 'ex_' + experiment_name + '_results_summary' + '.csv')




if __name__ == '__main__':
    # log_based_experiments()
    model_based_experiments()
