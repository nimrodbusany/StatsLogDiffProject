from Paired_Experiment_Results import Experiment_Result, compute_mle_k_future_dict, sample_traces
from BearLogParser import *
from SimpleLogParser import SimpleLogParser
from src.logs.log_generator import LogGeneator
import src.statistical_diffs.statistical_log_diff_analyzer as sld
import itertools
import pandas as pd


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


if __name__ == '__main__':

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
