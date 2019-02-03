from s2kdiff_experiments import S2KDiff_Experiment_Result
from s2kdiff_experiments import run_model_based_experiments as s2kdiff_model_based_experiments
from snkdiff_experiments import SNKDiff_Experiment_Result
from snkdiff_experiments import run_model_based_experiments as snkdiff_model_based_experiments


DFLT_NUM_OF_MODELS = 4
DFLT_SAMPLE_SIZE = 10000
DFLT_K = 2
DFLT_MIN_DIFF = 0.01
DFLT_ALPHA = 0.05
DFLT_REPETITIONS = 10

############################
### S2KDIFF experiments ####
############################

def s2kdiff_experiment_setup_varying_instance():

    def export_results(experiment_results, name):
        experiment_results.export_to_csv(
            RESULT_FODLER + 'results.csv')
        experiment_results.export_summary_to_csv(
            RESULT_FODLER + 'results_summary' + '.csv')

    ks, min_diffs, alphas, traces_to_sample, repetitions = [DFLT_K], [DFLT_MIN_DIFF], [DFLT_ALPHA], [DFLT_SAMPLE_SIZE], DFLT_REPETITIONS
    ## Experiments main parameters
    # ks = [2]
    # min_diffs = [0.01, 0.05, 0.1]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    # alphas = [0.05]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    # traces_to_sample = [500, 5000, 50000, 100000]  # [50, 500, 5000, 50000, 500000]
    experiment_results = S2KDiff_Experiment_Result()
    RESULT_FODLER = '../../results/statistical_experiments/paper_experiments/model_based/s2kdiff/varying_instance/single_instance/'
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, models_group=0, repetitions=repetitions)
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, models_group=1, repetitions=repetitions)
    export_results(experiment_results, 'single')
    experiment_results = S2KDiff_Experiment_Result ()
    RESULT_FODLER = '../../results/statistical_experiments/paper_experiments/model_based/s2kdiff/varying_instance/multiple_instances/'
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, models_group=0, repetitions=repetitions)
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, models_group=1, repetitions=repetitions)
    export_results(experiment_results, 'multiple')
    return experiment_results, experiment_results


def s2kdiff_varying_param_experiment_setup(varying_parameter, ks=[DFLT_K], min_diffs=[DFLT_MIN_DIFF], alphas=[DFLT_ALPHA], traces_to_sample=[DFLT_SAMPLE_SIZE], repetitions=DFLT_REPETITIONS):

    def export_results(experiment_results, base_dir):
        experiment_results.export_to_csv(
            base_dir + 'results.csv')
        experiment_results.export_summary_to_csv(
            base_dir + 'results_summary.csv')
        experiment_results.export_summary_to_csv(
            base_dir + 'results_grpby_summary' + '.csv', [varying_parameter])
    ## Experiments main parameters
    # ks = [2]
    # min_diffs = [0.01, 0.05, 0.1]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    # alphas = [0.05]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    # traces_to_sample = [500, 5000, 50000, 100000]  # [50, 500, 5000, 50000, 500000]
    experiment_results = S2KDiff_Experiment_Result()
    BASE_FOLDER = '../../results/statistical_experiments/paper_experiments/model_based/s2kdiff/varying_' + varying_parameter + "/"

    RESULT_FODLER = BASE_FOLDER + 'single_instance/'
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, models_group=0, repetitions=repetitions)
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, models_group=1, repetitions=repetitions)
    RESULT_FODLER = BASE_FOLDER + 'multiple_instances/'
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, models_group=0, repetitions=repetitions)
    s2kdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, models_group=1, repetitions=repetitions)
    export_results(experiment_results, BASE_FOLDER)
    return experiment_results, experiment_results

def run_s2kdiff_experiments():

    # s2kdiff experiment varying instance type
    s2kdiff_experiment_setup_varying_instance()
    # s2kdiff experiment varying k
    s2kdiff_varying_param_experiment_setup("k", ks=[1, 2, 3, 4])
    # s2kdiff experiment varying min_diff
    s2kdiff_varying_param_experiment_setup("min_diff", min_diffs=[0.01, 0.05, 0.1, 0.2, 0.5])
    # s2kdiff experiment varying samples
    s2kdiff_varying_param_experiment_setup("l1_traces", traces_to_sample=[100, 1000, 10000, 50000])
    # Experiment_setup VARYING LOGS!
    # Experiment_setup real world logs!


############################
### SNKDIFF experiments ####
############################

def snkdiff_experiment_setup_varying_instance():

    def export_results(experiment_results):
        experiment_results.export_to_csv(
            RESULT_FODLER + 'results.csv')
        experiment_results.export_summary_to_csv(
            RESULT_FODLER + 'results_summary.csv')
    # ks, min_diffs, alphas, traces_to_sample, repetitions = [2], [0.01], [0.05], [100], 10
    ## Experiments main parameters
    # ks = [2]
    # min_diffs = [0.01, 0.05, 0.1]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    # alphas = [0.05]  # [0.01, 0.05, 0.1, 0.2, 0.4]
    # traces_to_sample = [500, 5000, 50000, 100000]  # [50, 500, 5000, 50000, 500000]
    ks, min_diffs, alphas, traces_to_sample, repetitions = [DFLT_K], [DFLT_MIN_DIFF], [DFLT_ALPHA], \
                        [DFLT_SAMPLE_SIZE], DFLT_REPETITIONS
    experiment_results = SNKDiff_Experiment_Result()
    RESULT_FODLER = '../../results/statistical_experiments/paper_experiments/model_based/snkdiff/varying_instance/single_instance/'
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, \
                                    models_group=0, repetitions=repetitions)
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, \
                                    models_group=1, repetitions=repetitions)
    export_results(experiment_results)
    experiment_results = SNKDiff_Experiment_Result()
    RESULT_FODLER = '../../results/statistical_experiments/paper_experiments/model_based/snkdiff/varying_instance/multiple_instances/'
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, \
                                    models_group=0, repetitions=repetitions)
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, \
                                    models_group=1, repetitions=repetitions)
    export_results(experiment_results)
    return experiment_results, experiment_results

def snkdiff_varying_param_experiment_setup(varying_parameter, ks=[DFLT_K], min_diffs=[DFLT_MIN_DIFF], \
                                           alphas=[DFLT_ALPHA], traces_to_sample=[DFLT_SAMPLE_SIZE], models_2_fetch = [DFLT_NUM_OF_MODELS],  repetitions=DFLT_REPETITIONS):

    def export_results(experiment_results, base_dir):
        experiment_results.export_to_csv(
            base_dir + 'results.csv')
        experiment_results.export_summary_to_csv(
            base_dir + 'results_summary' + '.csv')
        experiment_results.export_summary_to_csv(
            base_dir + 'results_grpby_summary' + '.csv', [varying_parameter])

    experiment_results = SNKDiff_Experiment_Result()
    BASE_FOLDER = '../../results/statistical_experiments/paper_experiments/model_based/snkidff/varying_' + varying_parameter + "/"
    RESULT_FODLER = BASE_FOLDER + 'single_instance/'
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, models_group=0, models_to_fetch_arr = models_2_fetch, repetitions=repetitions)
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=True, models_group=1, models_to_fetch_arr = models_2_fetch, repetitions=repetitions)

    RESULT_FODLER = BASE_FOLDER + 'multiple_instances/'
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, models_group=0, models_to_fetch_arr = models_2_fetch, repetitions=repetitions)
    snkdiff_model_based_experiments(ks, min_diffs, alphas, traces_to_sample, experiment_results, RESULT_FODLER, single_instance=False, models_group=1, models_to_fetch_arr = models_2_fetch, repetitions=repetitions)
    export_results(experiment_results, BASE_FOLDER)
    return experiment_results, experiment_results


def run_snkdiff_experiments():

    # snkdiff experiment varying instance type
    snkdiff_experiment_setup_varying_instance()
    # snkdiff experiment varying k
    snkdiff_varying_param_experiment_setup("k", ks=[1, 2, 3, 4])
    # snkdiff experiment varying min_diff
    snkdiff_varying_param_experiment_setup("min_diff", min_diffs=[0.01, 0.05, 0.1, 0.2, 0.5])
    # snkdiff experiment varying samples
    snkdiff_varying_param_experiment_setup("sample_size", traces_to_sample=[100, 1000, 10000, 50000]) ## 100, 1000, 10000, 50000
    # Experiment_setup VARYING LOGS!
    # snkdiff_varying_param_experiment_setup("num_of_logs", models_2_fetch=[2, 4, 6, 8])
    # Experiment_setup real world logs!

if __name__ == '__main__':

    # run_s2kdiff_experiments()
    run_snkdiff_experiments()