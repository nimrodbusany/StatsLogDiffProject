import numpy as np
import pandas as pd

import src.statistical_diffs.statistical_log_diff_analyzer as sld
# import src.statistical_diffs.hypothesis_testing
from hypothesis_testing import is_significant_by_multiple_proportions_test


def create_sample_realization(number_of_trace, p1):
    a = np.ones(int(p1 * number_of_trace))
    b = np.zeros(number_of_trace)
    b[:len(a)] += a
    return b

class SNKDiff_Experiment_Result():

    def __init__(self, biases= (0,)):
        self.pairwise_results = None
        self.chi_square_results = None
        self.columns = ['model', 'single_instance', 'sample_size', 'k', 'min_diff', 'alpha', 'num_of_logs']
        self.bs = ['bias_'+str(i) for i in range(len(biases))]
        self.columns.extend(self.bs)
        self.columns.extend(['tp', 'fp', 'tn', 'fn', 'precision',
                   'recall', 'acc', 'true_error (tp/tn+fp)', 'power (tp/tp+fn)'])
        self.grp_by_columns = ['model', 'single_instance', 'sample_size', 'k', 'min_diff', 'alpha', 'num_of_logs']
        self.grp_by_columns.extend(self.bs)

    def compare_two_logs(self, transition_probabilities, sampled_diffs, i, j, min_diff):
        p1 = transition_probabilities.get(i, 0)
        p2 = transition_probabilities.get(j, 0)
        real_diff = abs(p1 - p2)
        # real_pooled_variance = (1 / (number_of_trace ** 2)) * (p1 * (1 - p1) + p2 * (1 - p2))
        # real_se = np.sqrt(real_pooled_variance)
        # real_cohens_d = real_diff / real_se
        # real_diff = real_cohens_d > min_diff
        # log1 = create_sample_realization(number_of_trace, p1)
        # log2 = create_sample_realization(number_of_trace, p2)

        if i > j:
            raise ValueError("expecting paired comparisons to be performed from lower to upper index")
        pair = sampled_diffs['significant_diffs'].get((i, j))
        if real_diff > min_diff: ## handle positive instances
            if pair and pair[ 'significant_diff']:
                return 'tp' ## predicted diff that exists
            else:
                return 'fn'  ## did not predict diff that exists
        else:
            if pair and pair['significant_diff']: ## handle negative instances
                return 'fp'  ## predicted diff that did not exists
            else:
                return 'tn'  ## did not predict diff that does not exists


    def find_diff_between_all_logs(self, ground_truth_log_tuples, number_of_traces_array, alpha):

        logs = []
        for log_ind in range(len(number_of_traces_array)): ## TODO: find better solution
            log_pr = ground_truth_log_tuples.get(log_ind, 0)
            if number_of_traces_array[log_ind] < 8:
                number_of_traces_array[log_ind] = 8

            logs.append(create_sample_realization(number_of_traces_array[log_ind], log_pr))


        return is_significant_by_multiple_proportions_test(logs, alpha, sld.MultipleSLPDAnalyzer.CHI_SQUARE_BASED)[0]
        # p1 = sum(logs[0]) / len(logs[0])
        # if len([sum(log) / len(log) for log in logs if p1 == sum(log) / len(log)]) == len(logs):
        #     return False
        #
        # res = stats.f_oneway(*logs)
        # if math.isnan(res.pvalue): ## check HACK: try a simple correction!
        #     for log in logs:
        #         if sum(log) == 0:
        #             log[0] = 1
        #     res = stats.f_oneway(*logs)
        #
        # if math.isnan(res.pvalue):
        #     raise AssertionError('unexpected error when testing with chi_square over real logs')
        #
        # if res.pvalue < alpha:
        #     return True
        #
        # return False

    def add_experiment_result(self, T2Ps, k, min_diff, biases, alpha, statistical_diffs, number_of_trace, model_name, single_instance):
        '''
            :param transitions_to_probabilities_per_log: list of dictionaries of mapping k_futures -> k_futures for each of the logs
        '''
        new_chi_square_row = [model_name, single_instance, number_of_trace, k, min_diff, alpha, len(T2Ps)]
        new_chi_square_row.extend(biases)
        new_pairwise_row = [model_name, single_instance, number_of_trace, k, min_diff, alpha, len(T2Ps)]
        new_pairwise_row.extend(biases)
        ## create a mapping of transitions to probabilities (in each of the logs)
        all_transitions = {}

        for j in range(len(T2Ps)):
            T2P = T2Ps[j]
            ### go over the dict of each log
            for trans_pre in T2P:
                ### go over each equiv class
                trans_futures = T2P[trans_pre]
                for future_trans in trans_futures:
                    ### go over each future and add it to the mapping of transitions to probabilities
                    transition = (trans_pre, future_trans)
                    pr = trans_futures[future_trans]
                    transition_probabilities = all_transitions.get(transition, {})
                    transition_probabilities[j] = pr
                    all_transitions[transition] = transition_probabilities

        tp, tn, fp, fn = 0, 0, 0, 0
        m_tp, m_tn, m_fp, m_fn = 0, 0, 0, 0

        for transition in all_transitions:

                    ## actual diffs found
                    actual_transition_diffs_found = statistical_diffs.get(transition)
                    if not actual_transition_diffs_found: ## if transition was missed in all sampled logs, skip it!
                        continue
                    ## check first chi_square test
                    experiments_per_log = actual_transition_diffs_found['experiments_per_log']
                    ground_truth_chi_square_was_significant = self.find_diff_between_all_logs( \
                        all_transitions[transition], experiments_per_log, alpha)
                    actual_chi_square_was_significant = actual_transition_diffs_found['mutiple_proportion_test_significant']
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
                            res = self.compare_two_logs(all_transitions[transition], \
                                                        actual_transition_diffs_found, i, j, min_diff)
                            if res == 'tp':
                                tp+=1
                            if res == 'tn':
                                tn += 1
                            if res == 'fp':
                                fp += 1
                            if res == 'fn':
                                fn += 1

        acc, power, precision, recall, statistical_error = self.compute_percision_metric(fn, fp, tn, tp)
        a_acc, a_power, a_precision, a_recall, a_statistical_error = self.compute_percision_metric(m_fn, m_fp, m_tn, m_tp)
        new_pairwise_row.extend([tp, fp, tn, fn, precision, recall, acc, statistical_error, power])
        new_chi_square_row.extend([m_tp, m_fp, m_tn, m_fn, a_precision, a_recall, a_acc, a_statistical_error, a_power])
        self.pairwise_results = np.vstack([self.pairwise_results, new_pairwise_row]) if self.pairwise_results is not None else np.array(new_pairwise_row)
        self.chi_square_results = np.vstack(
            [self.chi_square_results, new_chi_square_row]) if self.chi_square_results is not None else np.array(new_chi_square_row)

    def compute_percision_metric(self, fn, fp, tn, tp):
        precision = -1 if (tp + fp) == 0 else tp / (tp + fp)
        recall = -1 if (tp + fn) == 0 else tp / (tp + fn)
        statistical_error = -1 if (tn + fp) == 0 else fp / (tn + fp)
        power = -1 if (tp + fn) == 0 else tp / (tp + fn)
        acc = -1 if (tp + tn + fp + fn) == 0 else (tp + tn) / (tp + tn + fp + fn)
        return acc, power, precision, recall, statistical_error

    def export_to_csv(self, path):

        import pandas as pd
        df = pd.DataFrame(data=self.pairwise_results, columns=self.columns)
        types = dict([(c, 'str') if c in ['model', 'single_instance'] else (c, 'float') for c in self.columns])
        for t in types:
            df[t] = df[t].astype(types[t])
        df['total_transition'] = df['tp'] + df['tn'] + df['fp'] + df['fn']
        path_z_test = path.replace('.csv', '_z_tests.csv')
        df.to_csv(path_z_test, index=False, float_format='%.4f')


        # path_pairwise = path.replace('.csv', '_s2kdiff.csv')
        # with open(path_pairwise, 'wb') as f:
        #     f.write(bytes(",".join(self.columns)+'\n','utf-8'))
        #     np.savetxt(f, self.pairwise_results, delimiter=",", fmt='%.4f')

        df = pd.DataFrame(data=self.chi_square_results, columns=self.columns)
        types = dict([(c, 'str') if c in ['model', 'single_instance'] else (c, 'float') for c in self.columns])
        for t in types:
            df[t] = df[t].astype(types[t])
        df['total_transition'] = df['tp'] + df['tn'] + df['fp'] + df['fn']
        path_chi_square = path.replace('.csv', '_chi_square.csv')
        df.to_csv(path_chi_square, index=False, float_format='%.4f')

        # chi_square = path.replace('.csv', '_multiple_props.csv')
        # with open(chi_square, 'wb') as f:
        #     f.write(bytes(",".join(self.columns)+'\n','utf-8'))
        #     np.savetxt(f, self.chi_square_results, delimiter=",", fmt='%.4f')


    def summarize_this(self, path, results, grp_by_columns=None):

        if not grp_by_columns:
            grp_by_columns = self.grp_by_columns
        else:
            grp_by_columns.extend(self.grp_by_columns)

        df = pd.DataFrame(data=results, columns=self.columns)
        types = dict([(c, 'str') if c in ['model', 'single_instance'] else (c, 'float') for c in self.columns])
        for t in types:
            df[t] = df[t].astype(types[t])

        # bias = df['bias']
        df = df.replace(-1, np.NaN)
        # df['bias'] = bias  ## TODO: resolve this ugly hack; need to add bias, since group by with NaN values break the group by
        df['total_transition'] = df['tp'] + df['tn'] + df['fp'] + df['fn']
        grps = df.groupby(grp_by_columns)
        df_means = grps.mean()
        df_means['parameter'] = 'mean'
        df_std = grps.std()
        df_std['parameter'] = 'std'
        df_median = grps.median()
        df_median['parameter'] = 'statistics'
        summary = pd.concat([df_means, df_std, df_median], axis=0)
        summary.to_csv(path)

    def export_summary_to_csv(self, path, grp_by_columns= None):

        path_pairwise = path.replace('.csv', '_z_tests.csv')
        self.summarize_this(path_pairwise, self.pairwise_results, grp_by_columns)
        path_chi_square = path.replace('.csv', '_chi_square.csv')
        self.summarize_this(path_chi_square, self.chi_square_results, grp_by_columns)

        # path_pairwise = path.replace('.csv', '_z_tests.csv')
        # res = self.summarize_results(self.pairwise_results)
        # res.to_csv(path_pairwise)
        #
        # path_chi_square = path.replace('.csv', '_chi_square.csv')
        # res = self.summarize_results(self.chi_square_results)
        # res.to_csv(path_chi_square)

    def summarize_results(self, results):
        df = pd.DataFrame(data=results, columns=self.columns)
        df = df.replace(-1, np.NaN)
        # df['bias'] = bias  ## TODO: resolve this ugly hack; need to add bias, since group by with NaN values break the group by
        grps = df.groupby(self.grp_by_columns)
        res = grps.apply(np.mean)
        return res

    def __repr__(self):
        return str(self.pairwise_results)

