import src.statistical_diffs.statistical_log_diff_analyzer as sld
from src.logs.log_generator import get_test_log


import unittest
from os.path import join
import networkx as nx

__VERBOSE__ = True

class Test_DGraph(unittest.TestCase):

    def test_logs_from_models_with_statistical_diff(self):

        if __VERBOSE__:
            sld.__VERBOSE__ = False

        min_diff = 0.0
        alpha = 0.05
        bias = 0.2
        k = 2
        log1 = get_test_log(0.0, 1000)
        log2 = get_test_log(bias, 1000)
        alg = sld.SLPDAnalyzer(log1, log2)
        diffs = alg.find_statistical_diffs(k, min_diff, alpha)
        diffs.sort()
        print(diffs)
        self.assertTrue(diffs == [[('I', 'a'), ('a', 'b')],[('I', 'a'), ('a', 'c')]], '2 statstical diffs are expected, (I, a), (a, b)')
        # self.assertTrue(len(diffs) == 2, '2 statstical diffs are expected, a->b and a->c')

    def test_logs_from_models_with_no_statistical_diff(self):

        if __VERBOSE__:
            sld.__VERBOSE__ = False

        min_diff = 0.0
        alpha = 0.05
        bias = 0.0
        k = 2
        log1 = get_test_log(0.0, 1000)
        log2 = get_test_log(bias, 1000)
        alg = sld.SLPDAnalyzer(log1, log2)
        diffs = alg.find_statistical_diffs(k, min_diff, alpha)
        self.assertTrue(len(diffs) == 0, 'no statstical diffs are expected')
