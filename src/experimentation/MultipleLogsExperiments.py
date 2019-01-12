from Multi_Log_Experiment_Results import Experiment_Result, compute_mle_k_future_dict, sample_traces
from BearLogParser import *
from SimpleLogParser import SimpleLogParser
from src.logs.log_generator import LogGeneator
import src.statistical_diffs.statistical_log_diff_analyzer as sld
import itertools
import pandas as pd


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
        LOG_PATH = '../../data/bear/filtered_logs/'
        mozilla_desktop = SimpleLogParser.read_log(LOG_PATH + 'desktop.log')
        mozilla_mobile = SimpleLogParser.read_log(LOG_PATH + 'mobile.log')
        duplicates_to_full_log_size(mozilla_desktop, full_log_size)
        duplicates_to_full_log_size(mozilla_mobile, full_log_size)
        experiment_name = "bear, desktop_mobile"
        logs.extend([mozilla_desktop, mozilla_desktop, mozilla_mobile, mozilla_mobile])
        return logs, experiment_name, out_folder + 'bear_multiple_logs_1/'

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

if __name__ == '__main__':

      ## statistical
    RESULT_FODLER = '../../results/statistical_experiments/'
    OUTPUT_ACTUAL_DIFFS = True
    EXPERIMENT_TYPE = 1
    CHI_SQUARE_BASED = False


    WRITE_SINGLE_EXPERIMENT_RESULTS = True
    ## Experiments main parameters
    ks = [2]
    min_diffs = [0.01, 0.05, 0.1, 0.2] #, 0.05, 0.1, 0.2, 0.4] #[0.01, 0.05, 0.1, 0.2, 0.4]
    alphas = [0.05, 0.15, 0.25, 0.35] # [0.01, 0.05, 0.05, 0.1, 0.2, 0.4]

    ## Repetition per configuration

    M = 3 # 10
    full_log_size = 10000
    traces_to_sample = [10000] # [50, 500, 5000, 50000, 100000, 500000]
    biases_array = [(0, )] #(0, 0.1, -0.1)
    for k in ks:
        for biases in biases_array:
            experiment_results = Experiment_Result(biases)
            logs, experiment_name, outpath = get_logs(EXPERIMENT_TYPE, RESULT_FODLER, full_log_size=full_log_size, biases = biases)
            ground_truth = []
            for log in logs:
                ground_truth.append(compute_mle_k_future_dict(log, k))

        for (min_diff, alpha) in itertools.product(min_diffs, alphas):

                for sample in traces_to_sample:
                    for trial in range(M): ## repeat the experiment for m randomally selected logs
                        sampled_logs = []
                        log1 = sample_traces(logs[0], sample)
                        log2 = sample_traces(logs[1], sample)
                        log3 = sample_traces(logs[2], sample)
                        log4 = sample_traces(logs[3], sample)
                        sampled_logs.append(log1)
                        sampled_logs.append(log2)
                        sampled_logs.append(log3)
                        sampled_logs.append(log4)
                        # for log in logs:
                        #     sampled_logs.append(sample_traces(log, sample))
                        alg = sld.MultipleSLPDAnalyzer(sampled_logs)
                        diffs = alg.find_statistical_diffs(k, min_diff, alpha)
                        if OUTPUT_ACTUAL_DIFFS:
                            bs = str(biases).replace(',','_')
                            vals = "_".join(['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 'bs_' + bs, 's_' + str(sample), 't_' + str(trial)])
                            keys = list(diffs[list(diffs.keys())[0]].keys())
                            keys.extend(['source', 'target'])
                            df = pd.DataFrame(columns= keys)
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
                        experiment_results.add_experiment_result(ground_truth, k, min_diff, biases, alpha, diffs, sample, full_log_size)

    experiment_results.export_to_csv(
        outpath + 'ex_' + experiment_name + '_results' + '.csv')
    experiment_results.export_summary_to_csv(
        outpath + 'ex_' + experiment_name + '_results_summary' + '.csv')
