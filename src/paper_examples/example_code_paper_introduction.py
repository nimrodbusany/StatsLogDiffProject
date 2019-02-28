import os

from model_based_log_generator import LogGenerator
from log_writer import LogWriter
from protocol_models_to_logs import ProtocolModel
from main.log_diff_runner import run
from src.utils.project_constants import *

MODELS_PATH = '../../models/stamina/cvs.net.dot'  ## ordSet.net.dot
LOGS_OUTPUT_PATH = '../../data/logs/example/cvs/'  ## ordset
RESULTS_OUTPUT_PATH = LOGS_OUTPUT_PATH + '/output/'

def produce_logs():

    MODEL_TO_PRODUCE = 6
    TRACE2PRODUCE = 100000

    first_model = ProtocolModel(MODELS_PATH, 0, assign_transtion_probs=True)
    second_model = ProtocolModel(MODELS_PATH, 0, assign_transtion_probs=True)

    for instance_id in range(MODEL_TO_PRODUCE):
        print('processing instance:', instance_id, MODELS_PATH)
        ## read model & add transition probabilities
        # model_generator = ProtocolModel(MODELS_PATH, instance_id, assign_transtion_probs=True)
        model_generator = first_model if instance_id < 3 else second_model
        log = LogGenerator.produce_log_from_model(model_generator.graph,
                                                  transition_probability_attribute=TRANSITION_PROBABILITY_ATTRIBUTE,
                                                  traces2produce=TRACE2PRODUCE)
        ## generate transition probabilities
        model_generator.write_transitions_probabilities(LOGS_OUTPUT_PATH)
        ## produce k-Tail model
        LogWriter.write_log(log, LOGS_OUTPUT_PATH + 'l' + str(instance_id) + ".log")


def produce_snkdiff_model():

    files_ = os.listdir(LOGS_OUTPUT_PATH)
    logs2paths = {}
    for f_ in files_:
        if f_.endswith('.log'):
            logs2paths[f_] = LOGS_OUTPUT_PATH + f_

    # run(logs2paths, RESULTS_OUTPUT_PATH, alpha=0.05, delta=0.15, k=1, alg2run=2)
    logs2paths2 = {}
    logs2paths2['l0.log'] = logs2paths['l0.log']
    logs2paths2['l4.log'] = logs2paths['l4.log']
    run(logs2paths2, RESULTS_OUTPUT_PATH, alpha=0.05, delta=0.15, k=1, alg2run=1)

if __name__ == '__main__':

    # produce_logs()
    produce_snkdiff_model()

