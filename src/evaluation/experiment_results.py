import numpy as np
import pandas as pd

from src.utils.project_constants import *


class ExperimentResult:

    def __init__(self):
        self.pairwise_results = None
        self.chi_square_results = None
        self.columns = [MODEL_ATTR_NAME, LOG_SIZE_ATTR_NAME, K_ATTR_NAME, MIN_DIFF_ATTR_NAME, ALPHA_ATTR_NAME, NUM_LOGS_ATTR_NAME]
        self.columns.extend([TP_ATTR_NAME, FP_ATTR_NAME, TN_ATT_NAME, FN_ATTR_NAME, \
                             PRECISION_ATTR_NAME, RECALL_ATTR_NAME, ACCURACY__ATTR_NAME, EST_ALPHA_ATTR_NAME, EST_BETA_ATTR_NAME])
        self.grp_by_columns = [MODEL_ATTR_NAME, LOG_SIZE_ATTR_NAME, K_ATTR_NAME, MIN_DIFF_ATTR_NAME, ALPHA_ATTR_NAME, NUM_LOGS_ATTR_NAME]

    def compare_two_logs(self, transition_probabilities, sampled_diffs, i, j, min_diff):
        p1 = transition_probabilities.get(i, 0)
        p2 = transition_probabilities.get(j, 0)
        real_diff = abs(p1 - p2)

        if i > j:
            raise ValueError("expecting paired comparisons to be performed from lower to upper index")
        pair = sampled_diffs[PAIRWISE_COMPARISON_ATTR_NAME].get((i, j))
        if real_diff > min_diff: ## handle positive instances
            if pair and pair[SIGNIFICANT_DIFF_ATTR_NAME]:
                return TP_ATTR_NAME ## predicted diff that exists
            else:
                return FN_ATTR_NAME ## did not predict diff that exists
        else:
            if pair and pair[SIGNIFICANT_DIFF_ATTR_NAME]: ## handle negative instances
                return FP_ATTR_NAME  ## predicted diff that did not exists
            else:
                return TN_ATT_NAME  ## did not predict diff that does not exists


    def find_diff_between_all_logs(self, ground_truth_log_tuples, number_of_traces_array, alpha):

        true_proportions = list(ground_truth_log_tuples.items())
        for i in range(len(true_proportions)-1):
            if true_proportions[i][1] != true_proportions[i+1][1]:
                return True, {}
        return False, {}


    def add_experiment_result(self, logs_batch, k, min_diff, alpha, statistical_diffs, number_of_trace):
        '''
            :param transitions_to_probabilities_per_log: list of dictionaries of mapping k_futures -> k_futures for each of the logs
        '''
        T2Ps = logs_batch.true_kseq_transtion_probs
        new_chi_square_row = [logs_batch.batch_name, number_of_trace, k, min_diff, alpha, len(T2Ps)]
        new_pairwise_row = [logs_batch.batch_name, number_of_trace, k, min_diff, alpha, len(T2Ps)]
        ## create a mapping of transitions to probabilities (in each of the logs)
        all_transitions = {}
        log_ids = list(logs_batch.logs.keys())
        for j in range(len(T2Ps)):
            T2P = T2Ps[j]
            id_j = log_ids[j]
            ### go over the dict of each log
            for trans_pre in T2P:
                ### go over each equiv class
                trans_futures = T2P[trans_pre]
                for future_trans in trans_futures:
                    ### go over each future and add it to the mapping of transitions to probabilities
                    transition = (trans_pre, future_trans)
                    pr = trans_futures[future_trans]
                    transition_probabilities = all_transitions.get(transition, {})
                    transition_probabilities[id_j] = pr
                    all_transitions[transition] = transition_probabilities

        tp, tn, fp, fn = 0, 0, 0, 0
        m_tp, m_tn, m_fp, m_fn = 0, 0, 0, 0

        for transition in all_transitions:

                    ## actual diffs found
                    actual_transition_diffs_found = statistical_diffs.get(transition)
                    if not actual_transition_diffs_found: ## if transition was missed in all sampled logs, skip it!
                        continue
                    ## check first chi_square test
                    experiments_per_log = actual_transition_diffs_found[EXPERIMENTS_PER_LOG_ATTR_NAME]
                    ground_truth_chi_square_was_significant = self.find_diff_between_all_logs( \
                        all_transitions[transition], experiments_per_log, alpha)[0]
                    actual_chi_square_was_significant = actual_transition_diffs_found[MUTIPLE_PROPORTION_TEST_SIGNIFICANT_ATTR_NAME]
                    if not ground_truth_chi_square_was_significant and actual_chi_square_was_significant:
                        m_fp += 1
                    if not ground_truth_chi_square_was_significant and not actual_chi_square_was_significant:
                        m_tn += 1
                        # continue ## we got it right on the chi_square, no diffs, not further checks!
                    if ground_truth_chi_square_was_significant and actual_chi_square_was_significant:
                        m_tp += 1
                    if ground_truth_chi_square_was_significant and not actual_chi_square_was_significant:
                        m_fn += 1

                    ## check second multiple logs test
                    for i in range(len(T2Ps)):
                        for j in range(i+1, len(T2Ps)):
                            id_i = log_ids[i]
                            id_j = log_ids[j]
                            res = self.compare_two_logs(all_transitions[transition], \
                                                        actual_transition_diffs_found, id_i, id_j, min_diff)
                            if res == 'tp':
                                tp+=1
                            if res == 'tn':
                                tn += 1
                            if res == 'fp':
                                fp += 1
                            if res == 'fn':
                                fn += 1

        acc, power, precision, recall, statistical_error = self.compute_precision_metric(fn, fp, tn, tp)
        a_acc, a_power, a_precision, a_recall, a_statistical_error = self.compute_precision_metric(m_fn, m_fp, m_tn, m_tp)
        new_pairwise_row.extend([tp, fp, tn, fn, precision, recall, acc, statistical_error, power])
        new_chi_square_row.extend([m_tp, m_fp, m_tn, m_fn, a_precision, a_recall, a_acc, a_statistical_error, a_power])
        self.pairwise_results = np.vstack([self.pairwise_results, new_pairwise_row]) if self.pairwise_results is not None else np.array(new_pairwise_row)
        self.chi_square_results = np.vstack(
            [self.chi_square_results, new_chi_square_row]) if self.chi_square_results is not None else np.array(new_chi_square_row)

    @staticmethod
    def compute_precision_metric(fn, fp, tn, tp):

        precision = -1 if (tp + fp) == 0 else tp / (tp + fp)
        recall = -1 if (tp + fn) == 0 else tp / (tp + fn)
        statistical_error = -1 if (tn + fp) == 0 else fp / (tn + fp)
        power = -1 if (tp + fn) == 0 else tp / (tp + fn)
        acc = -1 if (tp + tn + fp + fn) == 0 else (tp + tn) / (tp + tn + fp + fn)
        return acc, power, precision, recall, statistical_error

    def export_to_csv(self, results_folder, round_floats=False):

        def export_df(df, output_path):
            df = pd.DataFrame(data=df, columns=self.columns)
            types = dict([(c, 'str') if c in [MODEL_ATTR_NAME] else (c, 'float') for c in self.columns])
            for t in types:
                df[t] = df[t].astype(types[t])
            df[TOTAL_TRANSITION_ATTR_NAME] = df[TP_ATTR_NAME] + df[TP_ATTR_NAME] + df[FP_ATTR_NAME] + df[FN_ATTR_NAME]
            if round_floats:
                df.to_csv(output_path, index=False, float_format='%.4f')
            else:
                df.to_csv(output_path, index=False)

        export_df(self.pairwise_results, results_folder + '/results_z_tests.csv')
        export_df(self.chi_square_results, results_folder + '/results_chi_square.csv')


