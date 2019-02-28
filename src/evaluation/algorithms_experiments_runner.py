import pandas as pd
import numpy as np
from datetime import datetime
import random, string, itertools

from src.utils.project_constants import *
from src.utils.disk_operations import create_folder_if_missing
from src.statistical_diffs.statistical_log_diff_analyzer import MultipleSLPDAnalyzer
from src.ktails.ktails import kTailsRunner
from src.graphs.diff_graph_overlay import overlay_differences_over_graph


OUTPUT_ACTUAL_DIFFS = True

def analyze_z_test_results(pair_key, min_diff, p1, p2, item):

    pair_name = '_'.join([str(ind) for ind in pair_key])
    real_diff = abs(p1 - p2)
    item['GT_diff_' + pair_name] = real_diff

    if real_diff > min_diff and pair_key in item[DIFFERENT_IDS_ATTR_NAME] or \
                            real_diff < min_diff and pair_key not in item['different_ids']:
        item[pair_name + "_correct_classifcation"] = True
        item['correct_classifcation'] += 1
    else:
        item[pair_name + "_correct_classifcation"] = False
        item['incorrect_classifcation'] += 1

    test_res = item['pairwise_comparisons'].get(pair_key)
    if test_res:
        if not test_res['significant_diff']:
            if test_res[PVALUE_ATTR_NAME] == np.nan:
                item[pair_name + '_statstical_success'] = np.nan
            elif real_diff > min_diff:
                item['incorrect_classifcation_in_z_tests'] += 1
                item[pair_name + '_statstical_success'] = False
            elif real_diff < min_diff:
                item['correct_classifcation_in_z_tests'] += 1
                item[pair_name + '_statstical_success'] = True
        else:
            if real_diff > min_diff:
                item['correct_classifcation_in_z_tests'] += 1
                item[pair_name + '_statstical_success'] = True
            if real_diff < min_diff:
                item[pair_name + '_statstical_success'] = False
                item['incorrect_classifcation_in_z_tests'] += 1


def _enrich_item_with_experiment_info(models2fetch, diff, item, min_diff, logs_batch, \
                                      k, alpha, log_size, trial):

    models_transition_probs = logs_batch.true_kseq_transtion_probs
    model_name = logs_batch.batch_name
    logs_ids = list(logs_batch.logs.keys())
    item[SOURCE_ATTR_NAME], item[TARGET_ATTR_NAME] = diff[0], diff[1]
    item['correct_classifcation'], item['incorrect_classifcation'] = 0, 0
    item['correct_classifcation_in_z_tests'], item['incorrect_classifcation_in_z_tests'] = 0, 0

    for ind in range(models2fetch):
        item['true_m' + str(ind)] = models_transition_probs[ind].get(item[SOURCE_ATTR_NAME], {}).get(item[TARGET_ATTR_NAME], np.nan)

    for ind1, ind2 in itertools.product(range(models2fetch), range(models2fetch)):
        if ind1 >= ind2:
            continue
        pair_key = (logs_ids[ind1], logs_ids[ind2])
        p1 = models_transition_probs[ind1].get(item[SOURCE_ATTR_NAME], {}).get(item[TARGET_ATTR_NAME], np.nan)
        p2 = models_transition_probs[ind2].get(item[SOURCE_ATTR_NAME], {}).get(item[TARGET_ATTR_NAME], np.nan)
        analyze_z_test_results(pair_key, min_diff, p1, p2, item)


    item.update({K_ATTR_NAME: k, MIN_DIFF_ATTR_NAME: min_diff, ALPHA_ATTR_NAME:alpha,
                 LOG_SIZE_ATTR_NAME:log_size, TRIAL_ATTR_NAME: trial, MODEL_ATTR_NAME : model_name})

def output_summary_as_csv(df, outpath, fname):
    try:
        df.to_csv(outpath + fname, index=False)
    except:
        default_fname = 'diffs.csv'
        if fname != default_fname:
            print('cannot write results of test:', outpath+fname, "trying with default file name")
            output_summary_as_csv(df, outpath, default_fname) ## try again diffent_fname
        else:
            print('cannot write results of test, skipped')


def run_skdiffs(logs_manager, algorithm, ks, min_diffs, alphas, traces_to_sample, experiment_results,
                results_folder, models2fetch_arr = [2], repetitions=10, experiment_set_id=None):

    if algorithm not in [1, 2]:
        raise ValueError('1 - S2KDiff, 2 - SNKDnkdiff')

    if not experiment_set_id:
        experiment_set_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        results_folder += experiment_set_id + '/'

    configurations = itertools.product(models2fetch_arr, ks, min_diffs, alphas, traces_to_sample, range(repetitions))
    for (models2fetch, k, min_diff, alpha, log_size, trial) in configurations:
        vals = "_".join(['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 's_' + str(log_size),
                         't_' + str(trial)])
        print('Experiment Configuration:', vals)
        logs_manager.reset()
        configuration_id = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        logs_batch = logs_manager.get_next_logs_batch(k, logs2fetch=models2fetch, traces2produce=log_size, \
                                                      batch_id=experiment_set_id)
        while logs_batch:

            output_folder = results_folder + "/" + logs_batch.batch_name + "/" + configuration_id + "/"
            create_folder_if_missing(output_folder)

            alg = MultipleSLPDAnalyzer(logs_batch.logs, algorithm == 1)
            diffs = alg.find_statistical_diffs(k, min_diff, alpha)
            if OUTPUT_ACTUAL_DIFFS:
                if len(diffs) > 0:
                    keys = list(diffs[list(diffs.keys())[0]].keys())
                    keys.extend([SOURCE_ATTR_NAME, TARGET_ATTR_NAME])
                    df = pd.DataFrame(columns=keys)
                    for diff in diffs:
                        item = diffs[diff].copy()
                        _enrich_item_with_experiment_info(models2fetch, diff, item, min_diff, logs_batch, \
                                                           k, alpha, log_size, trial)
                        df = df.append(item, ignore_index=True)

                    fname = (vals).replace("0.", "")+'.csv'
                    output_summary_as_csv(df, output_folder, fname)
                else:
                    print('No diffs found, exps params: ',configuration_id, 'params', vals)
            experiment_results.add_experiment_result(logs_batch, k, min_diff, alpha, diffs, log_size)
            logs_batch = logs_manager.get_next_logs_batch(k, logs2fetch=models2fetch, traces2produce=log_size, \
                                                          batch_id=experiment_set_id)


def run_s2kdiff_overlay_step(alg, log_a, k, ktails_a, ktails_b, sig_diffs):

    transition_prob1, transition_prob2 = ktails_a.infer_transition_probabilities(), ktails_b.infer_transition_probabilities()
    diffs2covering_traces = alg.find_covering_traces(sig_diffs, transition_prob1, transition_prob2,log_a, k)  ## TODO
    ktails_a.overlay_transition_probabilities_over_graph()
    ktails_b.overlay_transition_probabilities_over_graph()
    if len(sig_diffs) > 0:
        ktails_a.overlay_trace_over_graph(sig_diffs[0], diffs2covering_traces[0])  ## TODO: missing pvalue, bolden diff


def measure_algorithms(alpha, k, logs_batch, min_diff, run_s2kdiff): ## TODO CALL CODE FROM MAIN!!!

    ## repeat the experiment for m randomally selected logs
    stat_alg_start_time = datetime.now()
    alg = MultipleSLPDAnalyzer(logs_batch.logs)
    diffs = alg.find_statistical_diffs(k, min_diff, alpha)
    sig_diffs = []
    sign_id_tag = DIFFERENT_IDS_ATTR_NAME
    for diff in diffs:
        item = diffs[diff].copy()
        # handle_multiple_logs_mapping(diffs[diff], [str(id) for id in range(len(logs_batch.logs))], item)
        item[SOURCE_ATTR_NAME] = diff[0]
        item[TARGET_ATTR_NAME] = diff[1]
        if item[sign_id_tag]:
            if not run_s2kdiff:
                if len(item[sign_id_tag]) > 0:
                    sig_diffs.append(item)
            else:
                sig_diffs.append(item)
    stat_alg_time = (datetime.now() - stat_alg_start_time).total_seconds()
    if run_s2kdiff:
        iter_ = iter(logs_batch.logs)
        first_log_id = next(iter_)
        first_log = logs_batch.logs[first_log_id]
        second_log_id = next(iter_)
        second_log = logs_batch.logs[second_log_id]
        print("processing logs:", first_log_id, second_log_id)
        ktails_start_time = datetime.now()
        ktails = kTailsRunner(first_log, k)
        ktails2 = kTailsRunner(second_log, k)
        ktails.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
        ktails2.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
        ktails_time = (datetime.now() - ktails_start_time).total_seconds()
        overlay_start_time = datetime.now()
        missing_from_both_logs=True
        exception_ = None
        try:
            run_s2kdiff_overlay_step(alg, first_log, k, ktails, ktails2, sig_diffs)
            missing_from_both_logs=False
        except ValueError as error:
            exception_ = error
        try:
            run_s2kdiff_overlay_step(alg, second_log, k, ktails, ktails2, sig_diffs)
            missing_from_both_logs = False
        except ValueError as error:
            exception_ = error
        if missing_from_both_logs:
            print ('smthing went wrong, no coverring trace in both logs for diff!')
            raise exception_
        overlay_time = -1 if sig_diffs == 0 else (datetime.now() - overlay_start_time).total_seconds()
    else:
        single_log = []
        for log_id in logs_batch.logs:
            single_log.extend(logs_batch.logs[log_id])
        ktails_start_time = datetime.now()
        ktails = kTailsRunner(single_log, k)
        g = ktails.run_ktails(add_dummy_init=False,
                              add_dummy_terminal=False)
        ktails_time = (datetime.now() - ktails_start_time).total_seconds()
        overlay_start_time = datetime.now()
        overlay_differences_over_graph(g, sig_diffs)
        overlay_time = (datetime.now() - overlay_start_time).total_seconds()
    return ktails_time, overlay_time, stat_alg_time