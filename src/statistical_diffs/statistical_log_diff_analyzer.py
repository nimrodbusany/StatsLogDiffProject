import numpy as np
from src.logs.log_feature_extractor import *
from src.statistical_modules.hypothesis_testing import perform_z_test,  preform_chi_square_test
from src.utils.project_constants import *

__VERBOSE__ = False


class MultipleSLPDAnalyzer:

    def __init__(self, logs, skip_chi_square=False):
        self.logs = logs.copy()
        self.skip_chi_square = skip_chi_square

    def perfrom_multiple_proportions_test(self, transition_vals_per_log, min_diff, alpha):

        experiments_per_log = [transition_vals_per_log[log][0] for log in transition_vals_per_log]
        logs = []
        logs_to_analyze = {}
        pairwise_comparisons = {}
        different_ids = []

        for log in transition_vals_per_log: ## TODO find better solution
            vals = transition_vals_per_log[log]
            if vals[0] == -1:
                logs.append([])
                logs_to_analyze[log] = []
                continue
            a = np.ones(vals[1])
            b = np.zeros(vals[0])
            b[:len(a)] += a
            logs.append(b)
            logs_to_analyze[log] = b


        if len(logs) <= 1: ## TODO replace results dictrionaty with dedicated object
            return False, {STATISTICS_ATTR_NAME: -1, PVALUE_ATTR_NAME: -1, PAIRWISE_COMPARISON_ATTR_NAME: {}, MUTIPLE_PROPORTION_TEST_SIGNIFICANT_ATTR_NAME:False
                , DIFFERENT_IDS_ATTR_NAME: different_ids, COUNT_SIGNIFICANT_DIFFS_ATTR_NAME: 0, EXPERIMENTS_PER_LOG_ATTR_NAME: experiments_per_log}

        # Perform the mutiple proportions test
        if not self.skip_chi_square:
            is_significant, multiple_test_results = preform_chi_square_test(logs, alpha)
            multiple_proportion_test_results = {STATISTICS_ATTR_NAME: multiple_test_results[STATISTICS_ATTR_NAME], PVALUE_ATTR_NAME: multiple_test_results[PVALUE_ATTR_NAME],
                        PAIRWISE_COMPARISON_ATTR_NAME: pairwise_comparisons, MUTIPLE_PROPORTION_TEST_SIGNIFICANT_ATTR_NAME: is_significant, COUNT_SIGNIFICANT_DIFFS_ATTR_NAME: 0,
                      DIFFERENT_IDS_ATTR_NAME: different_ids, EXPERIMENTS_PER_LOG_ATTR_NAME: experiments_per_log}

            if is_significant is False:
                return is_significant, multiple_proportion_test_results
        else:
            multiple_proportion_test_results = {STATISTICS_ATTR_NAME: None,
                                                PVALUE_ATTR_NAME: None,
                                                PAIRWISE_COMPARISON_ATTR_NAME: pairwise_comparisons,
                                                MUTIPLE_PROPORTION_TEST_SIGNIFICANT_ATTR_NAME: None,
                                                COUNT_SIGNIFICANT_DIFFS_ATTR_NAME: 0,
                                                DIFFERENT_IDS_ATTR_NAME: different_ids,
                                                EXPERIMENTS_PER_LOG_ATTR_NAME: experiments_per_log}


        # if multiple proportion tests is_significant, perform pairwise comaprison
        log_ids = list(logs_to_analyze.keys())
        for i in range(len(log_ids)):
            log_id = log_ids[i]
            vals1 = transition_vals_per_log[log_id]
            if vals1[0] == -1:
                continue
            for j in range(i+1, len(log_ids)):
                log_id2 = log_ids[j]
                vals2 = transition_vals_per_log[log_id2]
                if vals2[0] == -1:
                    continue
                statistical_diff, pairwise_test_results = perform_z_test(vals1[1], vals1[0], vals2[1], vals2[0], min_diff, alpha)
                pairwise_comparisons[(log_id, log_id2)] = pairwise_test_results
                if pairwise_test_results[SIGNIFICANT_DIFF_ATTR_NAME]:
                    different_ids.append((log_id, log_id2))
                    multiple_proportion_test_results[COUNT_SIGNIFICANT_DIFFS_ATTR_NAME] += 1

        if len(pairwise_comparisons) > (len(self.logs)*(len(self.logs)-1)) / 2: ## sainity check: assertion for debug
            raise ValueError('something went wrong, too many diffs!')
        return multiple_proportion_test_results[COUNT_SIGNIFICANT_DIFFS_ATTR_NAME] == 0, \
               multiple_proportion_test_results ## if not pairwise diffs were found, do not report a diff

    def find_statistical_diffs(self, k = 2, min_diff = 0.0, alpha=0.05):
        '''
        :param k: compare k-sequences of length
        :param min_diff: minimum difference to consider
        :param alpha: statistical bounds to consider
        :return: a set of k_sequences that statistically differ
        '''
        dicts_rep = {}
        for log in self.logs:
            dict_rep = extract_k_sequences_to_future_dicts(self.logs[log], k)
            dicts_rep[log] = dict_rep

        k_seqs = set()
        for log in dicts_rep:
            k_seqs = k_seqs.union(set(dicts_rep[log].keys()))

        count, statistical_diffs = 0, {}
        for k_seq in k_seqs: ## TODO: COMPARE THE DIFFs OUT OF INTERSECTION
            futures = {}
            for log in dicts_rep:
                futures[log] = dicts_rep[log].get(k_seq)

            future_k_seqs = set()
            for log in futures:
                if futures[log]:
                    future_k_seqs = future_k_seqs.union(set(futures[log].keys()))

            for future_seq in future_k_seqs:
                log_transitions = {}
                count += 1
                transition_missing_from_all_logs = True
                for log in self.logs:
                    n = get_trails(dicts_rep[log], k_seq) ## DOUBLE CHECK
                    f_dict = dicts_rep[log]
                    state_dict = f_dict.get(k_seq)
                    if state_dict:
                        m = state_dict.get(future_seq, 0)
                        if m:
                            transition_missing_from_all_logs = False
                    else:
                        n, m = -1, -1 ##TODO find better solution!
                    log_transitions[log] = (n, m)

                if transition_missing_from_all_logs:
                    raise ValueError("smting went wrong, no log indices for transition!")
                statistical_diff, params = self.perfrom_multiple_proportions_test(log_transitions, min_diff, alpha)
                if __VERBOSE__:
                    st_ = ""
                    for i in range(len(log_transitions)):
                        t = log_transitions[i]
                        v = t[1] / t[0] if t[0] else 0
                        st_ += '-> p'+str(i)+', n'+str(i)+':'+str(v)+','+str(t[0]) + '\n'
                    if not statistical_diff:
                        print('No Diff:', k_seq, future_seq, '\n' + st_)
                    else:
                        print('FOUND STATISTICAL DIFF:', k_seq, future_seq, '\n' + st_)
                # if statistical_diff:
                statistical_diffs[(k_seq, future_seq)] = params
        print ('processed a total of ', count, 'transitions')
        return statistical_diffs

    def _compute_trace_probability(self, states2trasitions_probabilities, trace, k):

        ind, prob = 0, 1
        while ind < len(trace) - k:
            trace_ftr = trace[ind:ind+k]
            next_trace_ftr = trace[ind+1:ind+k+1]
            found_next = False
            for state in states2trasitions_probabilities:
                if trace_ftr == state:
                    next_ftrs = states2trasitions_probabilities.get(trace_ftr, {})
                    if next_trace_ftr in next_ftrs:
                        prob *= next_ftrs[next_trace_ftr]
                        found_next = True
                        break
                    else:
                        return 0
            if not found_next:
                return 0
            ind += 1
        return prob

    def find_covering_traces(self, diffs, states2trasitions_probabilities_a, states2trasitions_probabilities_b, traces, k):

        def from_diff_to_trantision(diff):
            transition = list(diff[SOURCE_ATTR_NAME])
            if len(diff[TARGET_ATTR_NAME]) == k:
                transition.append(diff[TARGET_ATTR_NAME][-1])
            return tuple(ev for ev in transition)

        def is_diff_in_trace(diff, trace, k):
            transition = from_diff_to_trantision(diff)
            for i in range(len(trace)):
                if transition == trace[i: i + k + 1]:
                    return True
            return False

        traces2prob_diff = {}
        diffs2traces = {}
        processed_traces = set() ## used to avoid processing the same trace multiple times
        for trace_ind_ in range(len(traces)):
            trace = traces[trace_ind_]
            if trace in processed_traces:
                continue
            processed_traces.add(trace)
            pa = self._compute_trace_probability(states2trasitions_probabilities_a, trace, k)
            pb = self._compute_trace_probability(states2trasitions_probabilities_b, trace, k)
            prob_diff = pa - pb
            traces2prob_diff[trace_ind_] = (prob_diff, pa, pb)

            for diff_ind in range(len(diffs)):
                if is_diff_in_trace(diffs[diff_ind], trace, k):
                    covering_traces = diffs2traces.get(diff_ind, [])
                    covering_traces.append(trace_ind_)
                    diffs2traces[diff_ind] = covering_traces

        for diff_ind in range(len(diffs)):
            if diff_ind not in diffs2traces:
                pairwise_comparison = diffs[diff_ind][PAIRWISE_COMPARISON_ATTR_NAME]
                test_result = next(iter(pairwise_comparison.values()))
                if test_result[P1_ATTR_NAME] > 0 and test_result[P2_ATTR_NAME] > 0:
                    raise ValueError("ERROR! No trace covering diff that appear in both logs: " + str(diffs[diff_ind]))
        selected_traces = {}
        for diff_ind in diffs2traces:
            selected_traces[diff_ind] = max(diffs2traces[diff_ind], key=lambda x: traces2prob_diff[x][0])
        return selected_traces