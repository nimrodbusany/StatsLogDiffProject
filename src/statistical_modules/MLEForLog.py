from src.logs.LogFeatureExtractor import *


def compute_mle_k_future_dict(log, k):

    l = LogFeatureExtractor(log)
    counts_dict = l.extract_k_sequences_to_future_dicts(k)
    mle_dict = l.compute_probabilities_from_k_sequences_to_future_dicts(counts_dict)
    return mle_dict
