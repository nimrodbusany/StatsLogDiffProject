import itertools, random, string
from datetime import datetime
import pandas as pd
import numpy as np

from src.evaluation.logs_manager import ModelBasedLogsManager, RealWorldLogsManager
from src.utils.disk_operations import create_folder_if_missing
from src.utils.project_constants import *
from src.evaluation.algorithms_experiments_runner import measure_algorithms

### TIMING EXPERIMENTS
DFLT_NUM_OF_MODELS = 4
DFLT_LOG_SIZE = 1000
DFLT_K = 2
DFLT_MIN_DIFF = 0.01
DFLT_ALPHA = 0.05
DFLT_REPETITIONS = 2

def run_time_measurements(logs_manager_, output_dir, \
                          k_exp=False, diff_exp=False, alpha=False, sample_exp=False, logs_exp=False):

    experiment_set_id = 'exp_id_' + ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    print('Experiment ID', experiment_set_id)
    output_dir = output_dir + '/' + experiment_set_id + "/"
    print('Experiment ID', experiment_set_id)
    if k_exp:  # snkdiff experiment varying k
        print("==== RUNNING k EXPERIMENTS ====")
        run_and_measure_algorithms(output_dir,  's2kdiff_k.csv', logs_manager_, 1, models_2_fetch=[2], ks=[1, 2, 3, 4], experiment_set_id=experiment_set_id)
        run_and_measure_algorithms(output_dir  + '/snkdiff.csv', logs_manager_, 2, models_2_fetch=[4], ks=[1, 2, 3, 4], experiment_set_id=experiment_set_id)

    if alpha:  # snkdiff experiment varying alpha
        print("==== RUNNING ALPHA EXPERIMENTS ====")
        run_and_measure_algorithms(output_dir,  's2kdiff_alpha.csv', logs_manager_, 1, models_2_fetch=[2], alphas=[0.1, 0.15])
        run_and_measure_algorithms(output_dir,  'snkdiff_alpha.csv', logs_manager_, 2, models_2_fetch=[4], alphas=[0.1, 0.15])

    if diff_exp:  # snkdiff experiment varying min_diff
        print("==== RUNNING MIN_DIFF EXPERIMENTS ====")
        run_and_measure_algorithms(output_dir,  's2kdiff_min_diff.csv', logs_manager_, 1, min_diffs=[0.01, 0.05, 0.1, 0.2, 0.4])
        run_and_measure_algorithms(output_dir,  'snkdiff__min_diff.csv', logs_manager_, 2, models_2_fetch=[4], min_diffs=[0.01, 0.05, 0.1, 0.2, 0.4])

    if sample_exp:  # snkdiff experiment varying samples
        print("==== RUNNING LOG_SIZE EXPERIMENTS ====")
        run_and_measure_algorithms(output_dir,  's2kdiff_sample_size.csv', logs_manager_, 1, models_2_fetch=[2],
                                     traces_to_sample=[100, 1000, 2000, 5000])  # 100,  500, 2500, 12500, 60000
        run_and_measure_algorithms(output_dir,  'snkdiff_sample_size.csv', logs_manager_, 2, models_2_fetch=[4],
                                   traces_to_sample=[100, 1000, 2000, 5000])  # , 500, 1000, 5000, 10000, 50000]) ## 100, 1000, 10000, 50000

    if logs_exp:  # Experiment_setup VARYING LOGS!
        print("==== RUNNING NUMBER_OF_LOGS EXPERIMENTS ====")
        run_and_measure_algorithms(output_dir,  'snkdiff_num_logs.csv', logs_manager_, 2, models_2_fetch=[2, 4, 6, 8])

    ## run and time s2kdiff
    # df = run_and_measure_algorithms(logs_manager, run_s2kdiff=1, ks=[2, 3, 4], min_diffs=[0.05], alphas=[0.05], \
    #                                 traces_to_sample=[100, 1000, 10000, 50000], models2fetch_arr = [2],  repetitions = 10, experiment_set_id = experiment_set_id)
    # df = run_and_measure_algorithms(logs_manager, run_s2kdiff=1, ks=[2], min_diffs=[0.05], alphas=[0.05], \
    #                                 traces_to_sample=[100], models2fetch_arr=[2], repetitions=2, experiment_set_id=experiment_set_id)
    # output_folder = output_folder + "/" + experiment_set_id
    # create_folder_if_missing(output_folder)

    # df.to_csv(output_folder + '/s2kdiff.csv', index=False)
    # print("DONE RUNNING 2KDIFF")
    ## run and time snkdiff
    # df = run_and_measure_algorithms(logs_manager, run_s2kdiff=2, ks=[2, 3, 4], min_diffs=[0.05], alphas=[0.05], \
    #                                 traces_to_sample=[100, 1000, 10000, 50000],
    #                                 models2fetch_arr=[2, 4, 6, 8], repetitions=10, experiment_set_id=experiment_set_id)
    # df = run_and_measure_algorithms(logs_manager, run_s2kdiff=2, ks=[2], min_diffs=[0.05], alphas=[0.05], \
    #                                 traces_to_sample=[100], models2fetch_arr=[4], repetitions=2,
    #                                 experiment_set_id=experiment_set_id)
    # df.to_csv(output_folder + '/snkdiff.csv', index=False)
    # print("DONE RUNNING 2NKDIFF")

def run_and_measure_algorithms(output_path, fname, logs_manager, run_s2kdiff,  ks=[DFLT_K], min_diffs=[DFLT_MIN_DIFF], \
                                           alphas=[DFLT_ALPHA], traces_to_sample=[DFLT_LOG_SIZE],
                               models_2_fetch = [DFLT_NUM_OF_MODELS], repetitions=DFLT_REPETITIONS):
    if run_s2kdiff not in [1, 2]:
        raise ValueError('1 - S2KDiff, 2 - SNKDnkdiff')

    measurements = []
    configurations = itertools.product(models_2_fetch, ks, min_diffs, alphas, traces_to_sample, range(repetitions))
    for (models2fetch, k, min_diff, alpha, log_size, trial) in configurations:
        vals = "_".join(['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 's_' + str(log_size),
                         't_' + str(trial)])
        print('Experiment Configuration:', vals)
        logs_manager.reset()
        read_start_time = datetime.now()
        logs_batch = logs_manager.get_next_logs_batch(k, logs2fetch=models2fetch, traces2produce=log_size)
        read_time = (datetime.now() - read_start_time).total_seconds()
        while logs_batch:
            ktails_time, overlay_time, stat_alg_time = measure_algorithms(alpha, k, logs_batch, min_diff,
                                                                      run_s2kdiff)
            measurements.append([logs_batch.batch_name, models2fetch, k, min_diff, alpha, traces_to_sample, trial, \
                                 read_time, stat_alg_time, ktails_time, overlay_time])

            read_start_time = datetime.now()
            logs_batch = logs_manager.get_next_logs_batch(k, logs2fetch=models2fetch, traces2produce=log_size)
            read_time = (datetime.now() - read_start_time).total_seconds()

    df = pd.DataFrame(columns=[MODEL_ATTR_NAME, NUM_LOGS_ATTR_NAME, K_ATTR_NAME, MIN_DIFF_ATTR_NAME, ALPHA_ATTR_NAME, LOG_SIZE_ATTR_NAME,
                               TRIAL_ATTR_NAME, READ_TIME_ATTRIBUTE, FIND_DIFF_ALG_TIME_ATTRIBUTE, KTAILS_TIME_ATTRIBUTE, GRAPH_OVERLAY_TIME_ATTRIBUTE], data=measurements)

    create_folder_if_missing(output_path)
    df.to_csv(output_path+fname, index=False)


if __name__ == '__main__':

    random.seed(a=1234567)
    np.random.seed(seed=1234567)

    output_dir = '../../evaluation/results/time/models/'
    configuration_file = "../../evaluation/configuration_files/models_input_configuration.json"
    logs_manager = ModelBasedLogsManager(configuration_file)
    run_time_measurements(logs_manager, output_dir, k_exp=True, diff_exp=True, sample_exp=True, logs_exp=True, alpha=True)

    # run_time_measurements(logs_manager, k_exp=True, diff_exp=True, sample_exp=True, logs_exp=True, alpha=True)

    # output_dir = '../../evaluation/results/time/bear/'
    # configuration_file = "../../evaluation/configuration_files/logs_input_configuration.json"
    # logs_manager = RealWorldLogsManager(configuration_file)
    # run_time_measurements(logs_manager, k_exp=True, diff_exp=True, sample_exp=True, logs_exp=True, alpha=True)


# run_time_measurements(configuration_file="models_input_configuration.json")

#
# def run_and_measure_algorithms(logs_manager, run_s2kdiff,  ks, min_diffs, alphas, traces_to_sample,
#                                 models2fetch_arr = [2], repetitions=10, experiment_set_id=None):
#     if run_s2kdiff not in [1, 2]:
#         raise ValueError('1 - S2KDiff, 2 - SNKDnkdiff')
#
#     measurements = []
#     configurations = itertools.product(models2fetch_arr, ks, min_diffs, alphas, traces_to_sample, range(repetitions))
#     for (models2fetch, k, min_diff, alpha, log_size, trial) in configurations:
#         vals = "_".join(['k_' + str(k), 'd_' + str(min_diff), 'al_' + str(alpha), 's_' + str(log_size),
#                          't_' + str(trial)])
#         print('Experiment Configuration:', vals)
#         logs_manager.reset()
#         read_start_time = datetime.now()
#         logs_batch = logs_manager.get_next_logs_batch(k, logs2fetch=models2fetch, traces2produce=log_size, \
#                                                       experiment_set_id=experiment_set_id)
#         read_time = (datetime.now() - read_start_time).total_seconds()
#         while logs_batch:
#             ktails_time, overlay_time, stat_alg_time = run_algorithms(alpha, k, log_size, logs_batch, min_diff,
#                                                                       run_s2kdiff)
#             measurements.append([logs_batch.batch_name, models2fetch, k, min_diff, alpha, log_size, trial, \
#                                  read_time, stat_alg_time, ktails_time, overlay_time])
#
#             read_start_time = datetime.now()
#             logs_batch = logs_manager.get_next_logs_batch(k, logs2fetch=models2fetch, traces2produce=log_size, \
#                                                           experiment_set_id=experiment_set_id)
#             read_time = (datetime.now() - read_start_time).total_seconds()
#
#     return pd.DataFrame(columns=["model_name", "number_of_logs", "k", "min_diff", "alpha", "log_size", "trial", \
#                                  "read_time", "stat_alg_time", "ktails_time", "overlay_time"], data=measurements)
#
#
# def run_s2kdiff_overlay_step(alg, log_a, k, ktails_a, ktails_b, sig_diffs):
#
#     transition_prob1, transition_prob2 = ktails_a.infer_transition_probabilities(), ktails_b.infer_transition_probabilities()
#     diffs2covering_traces = alg.find_covering_traces(sig_diffs, transition_prob1, transition_prob2,log_a, k)  ## TODO
#     ktails_a.overlay_transition_probabilities_over_graph()
#     ktails_b.overlay_transition_probabilities_over_graph()
#     if len(sig_diffs) > 0:
#         ktails_a.overlay_trace_over_graph(sig_diffs[0], diffs2covering_traces[0])  ## TODO: missing pvalue, bolden diff
#
#
# def run_algorithms(alpha, k, log_size, logs_batch, min_diff, run_s2kdiff):
#
#     ## repeat the experiment for m randomally selected logs
#     stat_alg_start_time = datetime.now()
#     alg = MultipleSLPDAnalyzer(logs_batch.logs)
#     diffs = alg.find_statistical_diffs(k, min_diff, alpha)
#     sig_diffs = []
#     sign_id_tag = DIFFERENT_IDS_ATTR_NAME
#     for diff in diffs:
#         item = diffs[diff].copy()
#         handle_multiple_logs_mapping(diffs[diff], [str(id) for id in range(len(logs_batch.logs))], item)
#         item[SOURCE_ATTR_NAME] = diff[0]
#         item[TARGET_ATTR_NAME] = diff[1]
#         if item[sign_id_tag]:
#             if not run_s2kdiff:
#                 if len(item[sign_id_tag]) > 0:
#                     sig_diffs.append(item)
#             else:
#                 sig_diffs.append(item)
#     stat_alg_time = (datetime.now() - stat_alg_start_time).total_seconds()
#     if run_s2kdiff:
#         ktails_start_time = datetime.now()
#         ktails = kTailsRunner(logs_batch.logs[0], k)
#         ktails2 = kTailsRunner(logs_batch.logs[1], k)
#         ktails.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
#         ktails2.run_ktails(add_dummy_init=False, add_dummy_terminal=False)
#         ktails_time = (datetime.now() - ktails_start_time).total_seconds()
#         overlay_start_time = datetime.now()
#         run_s2kdiff_overlay_step(alg, logs_batch.logs[0], k, ktails, ktails2, sig_diffs)
#         run_s2kdiff_overlay_step(alg, logs_batch.logs[1], k, ktails, ktails2, k, sig_diffs)
#         overlay_time = -1 if sig_diffs == 0 else (datetime.now() - overlay_start_time).total_seconds()
#     else:
#         single_log = []
#         for log in logs_batch.logs:
#             single_log.extend(log)
#         ktails_start_time = datetime.now()
#         ktails = kTailsRunner(single_log, k)
#         g = ktails.run_ktails(add_dummy_init=False,
#                               add_dummy_terminal=False)
#         ktails_time = (datetime.now() - ktails_start_time).total_seconds()
#         overlay_start_time = datetime.now()
#         overlay_differences_over_graph(g, sig_diffs)
#         overlay_time = (datetime.now() - overlay_start_time).total_seconds()
#     return ktails_time, overlay_time, stat_alg_time
#
#
# if __name__ == '__main__':
#
#     import pandas as pd
#     BASE_FOLDER = '../../results/statistical_experiments/paper_experiments/model_based/time/'
#
#     ## run and time s2kdiff
#     df = run_and_measure_algorithms([2], 0, [-1, 100], [2, 3], [0.05],
#                                     [0.05], 1, run_s2kdiff=True)
#     # df.to_csv(BASE_FOLDER + 's2kdiff.csv')
#     ## run and time snkdiff
#     # df = run_and_measure_algorithms([2, 4, 6, 8], 0, [100], [2], [0.05],
#     #                                 [0.05], 1, run_s2kdiff=False)
#     #
#     df.to_csv(BASE_FOLDER + 'snkdiff.csv', index=False)
