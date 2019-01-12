import src.statistical_diffs.statistical_log_diff_analyzer as sld
from src.logs.log_generator import LogGeneator
from src.statistical_modules.mle_for_log import *
from src.sampling.LogSampler import sample_traces
import itertools
import numpy as np
import pandas as pd


class Experiment_Result():

    def __init__(self):
        self.results = None
        self.columns = ['l1_traces', 'l2_traces', 'k', 'min_diff', 'bias', 'alpha', 'tp', 'fp', 'tn', 'fn', 'precision',
                   'recall', 'acc', 'true_error (tp/tn+fp)', 'power (tp/tp+fn)', 'total_transitions', 'not_enought_data']
        self.grp_by_columns = ['l1_traces', 'l2_traces', 'k', 'min_diff', 'bias', 'alpha']

    def add_experiment_result(self, T2Ps, k, min_diff, bias, alpha, statistical_diffs, M1, M2):
        '''
            :param transitions_to_probabilities_per_log: list of dictionaries of mapping k_futures -> k_futures for each of the logs
        '''
        new_row = [M1, M2, k, min_diff, bias, alpha]
        found_diffs = statistical_diffs
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
                    pr = trans_futures[future_trans]
                    transition = (trans_pre, future_trans)
                    transition_probabilities = all_transitions.get(transition, {})
                    transition_probabilities[j] = pr
                    all_transitions[transition] = transition_probabilities

        tp, tn, fp, fn = 0, 0, 0, 0
        not_enough_data_counter = 0
        for transition in all_transitions:
            transition_probabilities = all_transitions[transition]
            real_diff = abs(transition_probabilities.get(0, 0) - transition_probabilities.get(1, 0))
            significant_diff = False
            act_diff = found_diffs.get(transition)
            if act_diff is not None:
                significant_diff = act_diff['significant_diff']
                if act_diff['pval']=='NND':
                    not_enough_data_counter += 1
                    continue
            else:
                not_enough_data_counter += 1
                continue

            if real_diff > min_diff:
                if significant_diff:
                    tp += 1 ## predicted diff that exists
                else:
                    fn += 1 ## did not predict diff that exists
            else:
                if significant_diff:
                    fp += 1 ## predicted diff that did not exists
                else:
                    tn += 1 ## did not predict diff that does not exists

        precision = -1 if (tp+fp) == 0 else tp/(tp+fp)
        recall = -1 if (tp+fn) == 0 else tp / (tp+fn)
        statistical_error = -1 if (tn + fp) == 0 else fp / (tn + fp)
        power = -1 if (tp + fn) == 0 else tp / (tp + fn)
        acc = (tp+tn)/(tp+tn+fp+fn)
        new_row.extend([tp, fp, tn, fn, precision, recall, acc, statistical_error, power, len(all_transitions), not_enough_data_counter])
        self.results = np.vstack([self.results, new_row]) if self.results is not None else np.array(new_row)

    def export_to_csv(self, path):
        with open(path, 'wb') as f:
            f.write(bytes(",".join(self.columns)+'\n','utf-8'))
            np.savetxt(f, self.results, delimiter=",", fmt='%.4f')

    def export_summary_to_csv(self, path):


        df = pd.DataFrame(data=self.results, columns=self.columns)
        bias = df['bias']
        df = df.replace(-1, np.NaN)
        df['bias'] = bias ## TODO: resolve this ugly hack; need to add bias, since group by with NaN values break the group by
        grps = df.groupby(self.grp_by_columns)
        res = grps.apply(np.mean)
        res.to_csv(path)

    def __repr__(self):
        return str(self.results)
