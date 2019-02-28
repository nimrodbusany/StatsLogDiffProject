def extract_k_sequences_to_future_dicts(log, k): ### TODO: remove TOTAL_TRANSITIONS_ATTRIBUTE from dict, find
                                                    ### nicer solution
    ks_dict = {}
    for trace in log:
        for i in range(len(trace)):
            k_seq = trace[i: i + k]
            future_k_seq = trace[i + 1: i + k + 1]
            futures = ks_dict.get(k_seq, {})
            futures[future_k_seq] = futures.get(future_k_seq, 0) + 1
            ks_dict[k_seq] = futures

    return ks_dict


def get_trails(ks_dict_, k_seq):
    if k_seq not in ks_dict_:
        return 0
    return sum([v for k, v in ks_dict_[k_seq].items()])


def compute_probabilities_from_k_sequences_to_future_dicts(ks_dict):

    ks_dict = ks_dict.copy()
    for k_seq in ks_dict:
        futures = ks_dict[k_seq]
        total_transitions = sum([x[1] for x in futures.items()])
        for future_k_seq in futures:
            futures[future_k_seq] = futures[future_k_seq] / float(total_transitions)
    return ks_dict