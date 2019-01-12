import numpy as np
from scipy import stats
import math

__VERBOSE__ = False


def means_comparison(N=1000):
    for i in range(N):
        count_success = 0
        n1 = 1000
        n2 = 1000
        hypo_diff = 0.00
        actual_diff = 0.1
        a = np.random.choice(2, p=[0.5 - actual_diff, 0.5 + actual_diff], size=n1)
        b = np.random.choice(2, p=[0.5, 0.5], size=n2)
        # res = sp.statistical_modules.ttest_ind(a, b, equal_var=False, )
        var_1 = a.var(ddof=1)
        var_2 = b.var(ddof=1)
        se = np.sqrt(var_1 / n1 + var_2 / n2)
        df = (var_1 / n1 + var_2 / n2) ** 2 / (((var_1 / n1) ** 2 / (n1 - 1)) + ((var_2 / n2) ** 2 / (n2 - 1)))
        t = (a.mean() - b.mean() - hypo_diff) / se
        p = 1 - stats.t.cdf(t, df=df)
        if p < 0.05:
            count_success += 1
        if __VERBOSE__:
            print('t, df:', t, df)
            print('p_value is:', p)

    print('success rate:', count_success / N)


def perform_hypothesis_testing(m1, n1, m2, n2, delta, alpha):
    ## null hypothesis |p1 - p2| = 0
    p1 = 0
    if n1 != 0:
        p1 = m1 / n1
    p2 = 0
    if n2 != 0:
        p2 = m2 / n2
    p_hat = (m1 + m2) / (n1 + n2)
    v1 = 1 / n1 if n1 else 0
    v2 = 1 / n2 if n2 else 0
    se = np.sqrt(p_hat * (1 - p_hat) * (v1 + v2))  ## DOUBLE CHECK!!!
    test_res = False

    if m1 < 5 or m2 < 5 or (n1 - m1) < 5 or (n1 - m1) < 5:
        params = {'m1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'p1': p1, 'p2': p2, 'diff': abs(p1 - p2),
                  'p_hat': p_hat, 'se': se, 'z_hat': 'NA', 'pval': 'NND', 'delta': delta, 'alpha': alpha,
                  'significant_diff': test_res}
        return (test_res, params)
    if se == 0:  ## TODO: double check if correctly handling the case of zero SE
        params = {'m1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'p1': p1, 'p2': p2, 'diff': abs(p1 - p2),
                  'p_hat': p_hat, 'se': se, 'z_hat': 'NA', 'pval': 'NA', 'delta': delta, 'alpha': alpha,
                  'significant_diff': test_res}

        return (test_res, params)
    if __VERBOSE__:
        print('p1, p2, abs(p1 - p2), delta, se:', p1, p2, abs(p1 - p2), delta, se)

    min_val = min(p1, p2)
    max_val = max(p1, p2)

    # z_hat = (min_val - max_val - delta) / se
    # z_hat2 = (max_val - min_val - delta) / se
    # p_v1 = stats.norm.cdf(z_hat) + 1 - stats.norm.cdf(z_hat2)
    actual_diff = max_val - min_val
    z_hat = (max_val - min_val - delta) / se
    # p_v1 = 2 * (1 - stats.norm.cdf(z_hat))
    p_v1 = (1 - stats.norm.cdf(z_hat))
    test_res = (not (actual_diff < delta)) and p_v1 < alpha
    params = {'m1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'p1': p1, 'p2': p2, 'diff': abs(p1 - p2),
              'p_hat': p_hat, 'se': se, 'z_hat': z_hat, 'effect_size': (abs(p1 - p2) / se), 'pval': p_v1,
              'delta': delta, 'alpha': alpha, 'significant_diff': test_res}
    return (test_res, params)


def perform_t_test(m1, n1, m2, n2, delta,
                   alpha):  ## null_hypo: μ1 - μ2 >= 7 https://stattrek.com/hypothesis-test/difference-in-means.aspx

    import math
    from scipy.stats import t

    avg_1 = m1 / n1
    avg_2 = m2 / n2
    p_hat = (m1 + m2) / (n1 + n2)
    test_res = False

    if n1 <= 1 or n2 <= 1:
        se = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
        params = {'m1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'p1': avg_1, 'p2': avg_2, 'diff': abs(avg_1 - avg_2),
                  'deg_fr': -1,
                  'p_hat': p_hat, 'se': se, 't_hat': 'NA', 'pval': 'NA', 'delta': delta, 'alpha': alpha,
                  'significant_diff': test_res}
        return (test_res, params)

    se_1 = math.sqrt((((1 - avg_1) ** 2) * m1) / (n1 - 1))
    se_2 = math.sqrt((((1 - avg_2) ** 2) * m2) / (n2 - 1))

    se_hat = ((se_1 ** 2 / n1) + (se_2 ** 2 / n2)) ** 0.5
    # se_hat = (( ((n1 - 1) * (se_1 ** 2)) + ( (n2 - 1) * (se_2 ** 2)) ) / (n1 + n2 - 2)) ** 0.5
    if se_1 == 0 and se_2 == 0:  ## TODO: decide on the right value to return
        params = {'m1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'p1': avg_1, 'p2': avg_2, 'diff': abs(avg_1 - avg_2),
                  'deg_fr': -1,
                  'p_hat': p_hat, 'se': se_hat, 't_hat': 'NA', 'pval': 'NA', 'delta': delta, 'alpha': alpha,
                  'significant_diff': test_res}
        return (test_res, params)

    deg_fr = se_hat ** 2 / ((((se_1 ** 2 / n1) ** 2) / (n1 - 1)) + (((se_2 ** 2 / n2) ** 2) / (n2 - 1)))
    deg_fr = round(deg_fr)
    # deg_fr = n1 + n2 - 2
    min_avg = min(avg_1, avg_2)
    max_avg = max(avg_1, avg_1)
    val = (max_avg - min_avg - delta) / (se_hat * (((1 / n1) + (1 / n2)) ** 0.5))
    val2 = (min_avg - max_avg - delta) / (se_hat * (((1 / n1) + (1 / n2)) ** 0.5))
    try:
        p_val1 = 1 - t.cdf(val, deg_fr)
        p_val2 = t.cdf(val2, deg_fr)
        p_val = p_val1 + p_val2
    except RuntimeWarning:
        print('overflow error')

    test_res = p_val < alpha

    params = {'m1': m1, 'n1': n1, 'm2': m2, 'n2': n2, 'p1': avg_1, 'p2': avg_2, 'diff': abs(avg_1 - avg_2),
              'deg_fr': deg_fr,
              'p_hat': p_hat, 'se': se_hat, 't_hat': val, 'pval': p_val, 'delta': delta, 'alpha': alpha,
              'significant_diff': test_res}

    if p_val < alpha:
        print(params)
        print('\n')

    return (test_res, params)


def proportions_comparison(m1, n1, m2, n2, delta, alpha, hypothesis_testing=True):
    if hypothesis_testing:
        return perform_hypothesis_testing(m1, n1, m2, n2, delta, alpha)
    else:
        return perform_t_test(m1, n1, m2, n2, delta, alpha)


def proportions_comparison_test(N=1000):
    def compute_cohens_h(a, n1, b, n2):
        return abs(2 * np.arcsin((sum(a) / n1) ** 0.5) - 2 * np.arcsin((sum(b) / n2) ** 0.5))

    count_success = 0
    cohens_h_arr = []
    for i in range(N):
        n1 = 1000
        n2 = 1000
        actual_diff = 0.45
        a = np.random.choice(2, p=[0.5 - actual_diff, 0.5 + actual_diff], size=n1)
        b = np.random.choice(2, p=[0.5, 0.5], size=n2)
        p_hat = (sum(a) + sum(b)) / (n1 + n2)
        se = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
        z_hat = (a.mean() - b.mean() - actual_diff) / se
        p = 1 - stats.norm.cdf(z_hat)
        if p < 0.05:
            count_success += 1
        cohens_h_arr.append(compute_cohens_h(a, n1, b, n2))
    print('avg. success rate:', count_success / N)
    print('avg. cohens\' rate:', np.mean(cohens_h_arr))


def is_significant_by_anova_test(logs, alpha):

    p1 = sum(logs[0]) / len(logs[0])  ## if all proportion are equal, then anova automatically pass;
    ## we add this since for certain values, e.g., proprtions of all logs equal 1
    ## the results has pvalue = nan
    if len([sum(log) / len(log) for log in logs if p1 == sum(log) / len(log)]) == len(logs):
        res = stats.f_oneway(*logs)
        return False, res

    res = stats.f_oneway(*logs)
    if math.isnan(res.pvalue):
        for log in logs:
            if sum(log) == 0:
                log[0] = 1
        res = stats.f_oneway(*logs)

    if math.isnan(res.pvalue):
        raise ValueError('cannot run ANOVA due to technical issues with test')

    return res.pvalue < alpha, res


def is_significant_by_chi_square_test(logs, alpha):


    valid_logs = []
    for log in logs:
        if np.sum(log) > 4 and len(log) - np.sum(log) > 4:
            valid_logs.append(log)

    counts = [np.sum(x) for x in valid_logs]
    log_sizes = [x.shape[0] for x in valid_logs]
    avg_proportion = np.sum(counts) / np.sum(log_sizes)
    expected_counts = [math.ceil(avg_proportion * log_size) for log_size in log_sizes]

    p1 = sum(logs[0]) / len(logs[0])  ## if all proportion are equal, then anova automatically pass;
    ## we add this since for certain values, e.g., proprtions of all logs equal 1
    ## the results has pvalue = nan
    if (len([sum(log) / len(log) for log in logs if p1 == sum(log) / len(log)]) == len(logs)) \
            or len(valid_logs) < 1:
        res = stats.chisquare(counts, expected_counts)
        return False, res


    res = stats.chisquare(counts, expected_counts)
    # if math.isnan(res.pvalue):
    #     for log in logs:
    #         if sum(log) == 0:
    #             log[0] = 1
    #     res = stats.f_oneway(*logs)

    # if math.isnan(res.pvalue):
    #     raise ValueError('cannot run ANOVA due to technical issues with test')
    is_significant = not math.isnan(res.pvalue) and res.pvalue < alpha
    return is_significant, res


def is_significant_by_multiple_proportions_test(logs, alpha, chi_square_based=True):
    if chi_square_based:
        return is_significant_by_chi_square_test(logs, alpha)
    else:
        return is_significant_by_anova_test(logs, alpha)
