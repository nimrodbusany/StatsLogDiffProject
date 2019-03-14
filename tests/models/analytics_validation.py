from ksequence_analytics import compute_k_sequence_transition_probabilities
from simple_log_parser import SimpleLogParser
from log_based_mle import compute_mle_k_future_dict
from protocol_models_to_logs import ProtocolModel
from protocol_models_to_logs import produce_logs_from_stamina

def validate_stamina():


    MODELS_PATH = '../../ktails_models/stamina/'
    LOGS_OUTPUT_PATH = '../../data/logs/stamina/'

    models = ['ctas.net.simplified.no_loops.dot', 'cvs.net.no_loops.dot', 'roomcontroller.simplified.net.dot',
              'ssh.net.simplified.dot', 'tcpip.simplified.net.dot', 'zip.simplified.dot']
    dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip', 'zip']
    MODEL_TO_PRODUCE = 2
    K = 2
    TRACES2PRODUCE, EXPECTED_DIFF = 10000, 0.05
    # produce_logs_from_stamina(TRACES2PRODUCE)
    for model_id in range(len(models)):
        print('processing', models[model_id])
        dir_ = LOGS_OUTPUT_PATH + dirs_[model_id] + "/"
        for j in range(MODEL_TO_PRODUCE):
            log = SimpleLogParser.read_log(dir_+'l'+str(j)+'.log')
            transition_prob_path = dir_ + 'm'+str(j)+'_transitions_.csv'
            model_path = MODELS_PATH + models[model_id]
            model = ProtocolModel(model_path)
            model.update_transition_probabilities(transition_prob_path)
            k_seqs2k_seqs = compute_k_sequence_transition_probabilities(model, K)
            dict_ = compute_mle_k_future_dict(log, K)
            for k_seq in k_seqs2k_seqs:
                est_transitions = dict_.get(k_seq, {})
                true_transitions = k_seqs2k_seqs[k_seq]
                for tran in true_transitions:
                    diff = true_transitions.get(tran)-est_transitions.get(tran, 0)
                    print(k_seq, '->', tran, true_transitions.get(tran), est_transitions.get(tran, 0))
                    if diff > EXPECTED_DIFF and est_transitions != {}:
                        raise AssertionError("MLE estimator with " + str(TRACES2PRODUCE) + ' is expected to coverge with '
                                               'a distance below ' + str(EXPECTED_DIFF) + ' but diff equals ' + str(diff))


if __name__ == '__main__':

    validate_stamina()