from src.logs.log_feature_extractor import *


def compute_mle_k_future_dict(log, k):

    counts_dict = extract_k_sequences_to_future_dicts(log, k)
    mle_dict = compute_probabilities_from_k_sequences_to_future_dicts(counts_dict)
    return mle_dict
