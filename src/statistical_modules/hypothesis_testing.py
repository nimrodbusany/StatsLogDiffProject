import math
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from src.utils.project_constants import *
__VERBOSE__ = False

### TWO LOGS
def perform_z_test(m1, n1, m2, n2, delta, alpha):

    ## null hypothesis |p1 - p2| < delta
    p1 = 0 if n1 == 0 else m1 / n1
    p2 = 0 if n2 == 0 else m2 / n2
    p_hat = (m1 + m2) / (n1 + n2)
    if __VERBOSE__:
        print('p1, p2, abs(p1 - p2), delta, se:', p1, p2, abs(p1 - p2), delta)

    try:
        if p1 > p2:
            stat_test = proportions_ztest([m1, m2], [n1, n2], value=delta, alternative='larger')
        else:
            stat_test = proportions_ztest([m2, m1], [n2, n1], value=delta, alternative='larger')

        return stat_test[1] < alpha, {M1_ATTR_NAME: m1, N1_ATTR_NAME: n1, M2_ATTR_NAME: m2, N2_ATTR_NAME: n2, P1_ATTR_NAME: p1,\
                                      P2_ATTR_NAME: p2, 'diff': abs(p1-p2),\
                                      'p_hat': p_hat, STATISTICS_ATTR_NAME: stat_test[0], PVALUE_ATTR_NAME: stat_test[1],\
                                      'delta': delta, 'alpha': alpha, 'significant_diff': stat_test[1] < alpha}
    except:
        print('z_test assumption not met')
        return False, {M1_ATTR_NAME: m1, N1_ATTR_NAME: n1, M2_ATTR_NAME: m2, N2_ATTR_NAME: n2, P1_ATTR_NAME: p1,
                       P2_ATTR_NAME: p2, 'diff': abs(p1 - p2),
                       'p_hat': np.nan, 'se': np.nan, STATISTICS_ATTR_NAME: np.nan, PVALUE_ATTR_NAME: np.nan,
                       'delta': delta, 'alpha': alpha, 'significant_diff': False}


### MUTIPLE LOGS
def preform_chi_square_test(logs, alpha):

    valid_logs = []
    for log in logs:
        if len(log) > 5:
            valid_logs.append(log)
    try:
        counts = [np.sum(x) for x in valid_logs]
        log_sizes = [x.shape[0] for x in valid_logs]
        avg_proportion = np.sum(counts) / np.sum(log_sizes)
        expected_counts = [math.ceil(avg_proportion * log_size) for log_size in log_sizes]

        res = stats.chisquare(counts, expected_counts)
        is_significant = not math.isnan(res.pvalue) and res.pvalue < alpha
        return is_significant, {STATISTICS_ATTR_NAME: res.statistic, PVALUE_ATTR_NAME: res.pvalue}
    except:
        return False, {STATISTICS_ATTR_NAME: np.nan, PVALUE_ATTR_NAME: np.nan}