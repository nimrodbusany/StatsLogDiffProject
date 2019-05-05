import random, string
import numpy as np

from src.evaluation.experiment_results import ExperimentResult
from src.evaluation.algorithms_experiments_runner import run_skdiffs
from src.evaluation.logs_manager import ModelBasedLogsManager, RealWorldLogsManager

DFLT_NUM_OF_MODELS = 4
DFLT_LOG_SIZE = -1
DFLT_K = 2
DFLT_MIN_DIFF = 0.01
DFLT_ALPHA = 0.05
DFLT_REPETITIONS = 20


### SOUNDNESS EXPERIMENTS
def run_varying_param_experiment(output_dir, experiment_set_id, algorithm, logs_manager_, varying_parameter, ks=[DFLT_K], min_diffs=[DFLT_MIN_DIFF], \
                                           alphas=[DFLT_ALPHA], traces_to_sample=[DFLT_LOG_SIZE], models_2_fetch = [DFLT_NUM_OF_MODELS], repetitions=DFLT_REPETITIONS):
    try:
        experiment_results = ExperimentResult()
        label = 'snkdiff' if algorithm == 2 else 's2kdiff'
        result_fodler = output_dir + label +'/'+experiment_set_id+'/varying_' + varying_parameter + "/"
        run_skdiffs(logs_manager_, algorithm, ks, min_diffs, alphas, traces_to_sample, experiment_results, result_fodler, \
                    models2fetch_arr=models_2_fetch, repetitions=repetitions, experiment_set_id=experiment_set_id)
        experiment_results.export_to_csv(result_fodler)
    except Exception as e:
        print("skipped experiment!", experiment_set_id, algorithm, logs_manager, varying_parameter, ks, min_diffs, \
                                           alphas, traces_to_sample, models_2_fetch, repetitions)
        raise e

def run_evaluation_experiments(output_dir, logs_manager_, k_exp=False, diff_exp=False, sample_exp=False, logs_exp=False, alpha=False):

    experiment_set_id = 'exp_id_' + ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    print('Experiment ID', experiment_set_id)
    if k_exp: # snkdiff experiment varying k
        print("==== RUNNING k EXPERIMENTS ====")
        run_varying_param_experiment(output_dir, experiment_set_id, 1, logs_manager_, "k", ks=[1, 2, 3, 4])
        run_varying_param_experiment(output_dir, experiment_set_id, 2, logs_manager_, "k", ks=[1, 2, 3, 4])

    if alpha: # snkdiff experiment varying alpha
        print("==== RUNNING ALPHA EXPERIMENTS ====")
        run_varying_param_experiment(output_dir, experiment_set_id, 1, logs_manager_, "alpha", alphas=[0.01, 0.05, 0.1, 0.15])
        run_varying_param_experiment(output_dir, experiment_set_id, 2, logs_manager_, "alpha", alphas=[0.01, 0.05, 0.1, 0.15])


    if diff_exp: # snkdiff experiment varying min_diff
        print("==== RUNNING MIN_DIFF EXPERIMENTS ====")
        run_varying_param_experiment(output_dir, experiment_set_id, 1, logs_manager_, "min_diff", min_diffs=[0.01, 0.05, 0.1, 0.2, 0.4])
        run_varying_param_experiment(output_dir, experiment_set_id, 2, logs_manager_, "min_diff", min_diffs=[0.01, 0.05, 0.1, 0.2, 0.4])

    if sample_exp: # snkdiff experiment varying samples
        print("==== RUNNING LOG_SIZE EXPERIMENTS ====")
        run_varying_param_experiment(output_dir, experiment_set_id, 1, logs_manager_, "sample_size",
                                     traces_to_sample=[128, 256, 512, 1024, 2048, 4096, 8192, 16384]) # 100,  500, 2500, 12500, 60000
        run_varying_param_experiment(output_dir, experiment_set_id, 2, logs_manager_, "sample_size",
                                     traces_to_sample=[128, 256, 512, 1024, 2048, 4096, 8192, 16384]) #, 500, 1000, 5000, 10000, 50000]) ## 100, 1000, 10000, 50000

    if logs_exp: # Experiment_setup VARYING LOGS!
        print("==== RUNNING NUMBER_OF_LOGS EXPERIMENTS ====")
        run_varying_param_experiment(output_dir, experiment_set_id, 2, logs_manager_, "num_of_logs", models_2_fetch=[2, 4, 6, 8])

if __name__ == '__main__':

    random.seed(a=1234567)
    np.random.seed(seed=1234567)

    output_dir = '../../evaluation/results/soundness/ktails_models/'
    configuration_file = "../../evaluation/configuration_files/models_input_configuration.json"
    logs_manager = ModelBasedLogsManager(configuration_file)
    run_evaluation_experiments(output_dir, logs_manager, k_exp=True, diff_exp=True, sample_exp=True, logs_exp=True, alpha=True)

    output_dir = '../../evaluation/results/soundness/bear/'
    configuration_file = "../../evaluation/configuration_files/logs_input_configuration_quart.json"
    logs_manager = RealWorldLogsManager(configuration_file)
    run_evaluation_experiments(output_dir, logs_manager, k_exp=True, diff_exp=True, sample_exp=True, logs_exp=True, alpha=True)
