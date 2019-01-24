import numpy as np

from src.logs.log_feature_extractor import *
from src.statistical_modules.hypothesis_testing import proportions_comparison, \
    is_significant_by_multiple_proportions_test

__VERBOSE__ = False


def _extract_k_sequences_to_future_dicts(self, log, k):
    ksDict = {}
    for trace in log:
        for i in range(len(trace)):
            k_seq = trace[i: i + k]
            future_k_seq = trace[i + 1: i + k + 1]
            futures = ksDict.get(k_seq, {})
            futures[future_k_seq] = futures.get(future_k_seq, 0) + 1
            ksDict[k_seq] = futures

    for k_seq in ksDict:
        futures = ksDict[k_seq]
        total_transitions = sum([x[1] for x in futures.items()])
        for future_k_seq in futures:
            futures[future_k_seq] = futures[future_k_seq] / float(total_transitions)
        futures[TOTAL_TRANSITIONS_ATTRIBUTE] = total_transitions
    return ksDict


class SLPDAnalyzer:


    def __init__(self, logA, logB):
        self.logA = logA
        self.logB = logB


    def find_statistical_diffs(self, k = 2, min_diff = 0.0, alpha=0.05):
        '''
        :param k: compare k-sequences of length
        :param min_diff: minimum difference to consider
        :param alpha: statistical bounds to consider
        :return: a set of k_sequences that statistically differ
        '''
        dictA = extract_k_sequences_to_future_dicts(self.logA, k)
        dictB = extract_k_sequences_to_future_dicts(self.logB, k)

        k_seqs = set(dictA.keys())
        k_seqs = k_seqs.union(set(dictB.keys())) ## TODO: COMPARE THE DIFFs OUT OF INTERSECTION
        statistical_diffs = {}

        count = 0
        for k_seq in k_seqs: ## TODO: COMPARE THE DIFFs OUT OF INTERSECTION
            future_a = dictA.get(k_seq)
            future_b = dictB.get(k_seq)
            future_k_seqs = set()
            if future_a:
                future_k_seqs = future_k_seqs.union(set(future_a.keys()))
            if future_b:
                future_k_seqs = future_k_seqs.union(set(future_b.keys()))

            for future_seq in future_k_seqs:
                count+=1

                n1 = get_trails(dictA, k_seq)
                m1 = 0
                if future_a:
                    m1 = future_a.get(future_seq, 0)

                n2 = get_trails(dictB, k_seq)
                m2 = 0
                if future_b:
                    m2 = future_b.get(future_seq, 0)

                statistical_diff, params = proportions_comparison(m1, n1, m2, n2, min_diff, alpha)
                if __VERBOSE__:
                    p1 = m1 / n1 if n1 else 0
                    p2 = m2 / n2 if n2 else 0
                    if not statistical_diff:
                        print('No Diff:', k_seq, future_seq, '\np1, n1, p2, n2:', p1, n1, p2, n2)
                    else:
                        print('FOUND STATISTICAL DIFF:', k_seq, future_seq, '\np1, n1, p2, n2:', p1, n1, p2, n2)

                statistical_diffs[(k_seq, future_seq)] = params
        print ('processed a total of ', count, 'transitions')
        return statistical_diffs


class MultipleSLPDAnalyzer:

    CHI_SQUARE_BASED = True

    def __init__(self, logs):
        self.logs = logs.copy()

    def perfrom_scott_knott(self, transition_vals_per_log, min_diff, alpha):
        return None, None

    def compute_se_pooled(self, transition_vals_per_log):

        denom_ = 0
        nomi_ = 0
        for v in transition_vals_per_log:
            denom_+=v[0]
            p = (v[1] / v[0])
            if v[0] > 1:
                nomi_ += (v[0] - 1)((v[0] - v[1])*(p**2) + (v[1]*(1 - p)**2))
            denom_ += v[0]
        return np.sqrt(nomi_ / denom_)

    def perfrom_multiple_proportions_test(self, transition_vals_per_log, min_diff, alpha):

        experiments_per_log = [t[0] for t in transition_vals_per_log]
        logs = []
        logs_to_analyze = {}
        pair_wise_diffs = {}
        different_ids = []

        for i in range(len(transition_vals_per_log)): ## TODO find better solution
            vals = transition_vals_per_log[i]
            a = np.ones(vals[1])
            b = np.zeros(vals[0])
            b[:len(a)] += a
            logs.append(b)
            logs_to_analyze[i] = b

        if len(logs) <= 1: ## TODO replace results dictrionaty with dedicated object
            return False, {'statistic': -1, 'pvalue': -1,'significant_diffs': {}, 'mutiple_proportion_test_significant':False
                , 'different_ids': different_ids, 'count_significant_diffs': 0, 'experiments_per_log': experiments_per_log}

        # Perform the mutiple proportions test
        is_significant, multiple_test_results = is_significant_by_multiple_proportions_test(logs, alpha, MultipleSLPDAnalyzer.CHI_SQUARE_BASED)
        multiple_proportion_test_results = {'statistic': multiple_test_results.statistic, 'pvalue': multiple_test_results.pvalue, 'significant_diffs': pair_wise_diffs,
         'mutiple_proportion_test_significant': is_significant, 'count_significant_diffs': 0,
                  'different_ids': different_ids, 'experiments_per_log': experiments_per_log}

        if is_significant is False:
            return is_significant, multiple_proportion_test_results

        # if multiple proportion tests is_significant, perform pairwise comaprison
        log_ids = list(logs_to_analyze.keys())
        for i in range(len(log_ids)):
            log_id = log_ids[i]
            vals1 = transition_vals_per_log[i]
            for j in range(i+1, len(log_ids)):
                log_id2 = log_ids[j]
                vals2 = transition_vals_per_log[j]
                statistical_diff, pairwise_test_results = proportions_comparison(vals1[1], vals1[0], vals2[1], vals2[0], min_diff, alpha)
                pair_wise_diffs[(log_id, log_id2)] = pairwise_test_results
                if pairwise_test_results['significant_diff']:
                    different_ids.append((log_id, log_id2))
                    multiple_proportion_test_results['count_significant_diffs'] += 1

        if len(pair_wise_diffs) > (len(self.logs)*(len(self.logs)-1)) / 2: ## sainity check: assertion for debug
            raise ValueError('something went wrong, too many diffs!')
        return multiple_proportion_test_results['count_significant_diffs'] == 0, \
               multiple_proportion_test_results ## if not pairwise diffs were found, do not report a diff


    def find_statistical_diffs(self, k = 2, min_diff = 0.0, alpha=0.05):
        '''
        :param k: compare k-sequences of length
        :param min_diff: minimum difference to consider
        :param alpha: statistical bounds to consider
        :return: a set of k_sequences that statistically differ
        '''
        dicts_rep = []
        for l in self.logs:
            dict_rep = extract_k_sequences_to_future_dicts(l, k)
            dicts_rep.append(dict_rep)

        k_seqs = set()
        for d in dicts_rep:
            k_seqs = k_seqs.union(set(d.keys()))

        count, statistical_diffs = 0, {}
        for k_seq in k_seqs: ## TODO: COMPARE THE DIFFs OUT OF INTERSECTION
            futures = []
            for futures_dict in dicts_rep:
                futures.append(futures_dict.get(k_seq))

            future_k_seqs = set()
            for futures_dict in futures:
                if futures_dict:
                    future_k_seqs = future_k_seqs.union(set(futures_dict.keys()))

            for future_seq in future_k_seqs:
                tuples = []
                count += 1
                transition_missing_from_all_logs = True
                for i in range(len(self.logs)):
                    n = get_trails(dicts_rep[i], k_seq) ## DOUBLE CHECK
                    m = 0
                    f_dict = dicts_rep[i]
                    state_dict = f_dict.get(k_seq)
                    if state_dict:
                        m = state_dict.get(future_seq, 0)
                        if m:
                            transition_missing_from_all_logs = False
                    else:
                        n, m = -1, -1 ##TODO find better solution!
                        print('missing state')
                    tuples.append((n, m))

                if transition_missing_from_all_logs:
                    raise ValueError("smting went wrong, no log indices for transition!")

                statistical_diff, params = self.perfrom_multiple_proportions_test(tuples, min_diff, alpha)
                if __VERBOSE__:
                    st_ = ""
                    for i in range(len(tuples)):
                        t = tuples[i]
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

