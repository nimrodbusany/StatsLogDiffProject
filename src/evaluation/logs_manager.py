import json

from src.models.protocol_models_to_logs import ProtocolModel, TRANSITION_PROBABILITY_ATTRIBUTE
from src.logs.log_writer import LogWriter
from src.ktails.ksequence_analytics import compute_k_sequence_transition_probabilities
from src.logs_parsers.simple_log_parser import SimpleLogParser
from src.models.model_based_log_generator import LogGenerator
from src.statistical_modules.log_based_mle import compute_mle_k_future_dict
from src.sampling.log_sampler import sample_traces
from utils.disk_operations import create_folder_if_missing


class ExperimentBatch:

    def __init__(self, logs, true_kseq_transtion_probs , batch_name):
        self.logs = logs
        self.true_kseq_transtion_probs = true_kseq_transtion_probs
        self.batch_name = batch_name


class LogsManager:

    def __init__(self, write_logs=False):
        self.current_group_id = 0
        self.write_logs = False

    def get_next_logs_batch(self):
        raise NotImplemented('This method should be implemented by sub-classes')

    def reset(self):
        self.current_group_id = 0

class RealWorldLogsManager(LogsManager): ## TODO: simliar to the models manager, consider adding support of multiple log groups

    def __init__(self, logs_json_path, write_logs=False):

        super(RealWorldLogsManager, self).__init__(write_logs)
        content = json.load(open(logs_json_path))
        logs_path_dict = content["logs"]

        self.log_set_name = ""
        if "log_set_name" in content:
            self.log_set_name = content["log_set_name"]

        self.output_path = None
        if 'logs_output_path' in content:
            self.output_path = content["logs_output_path"]

        self.true_ksequence_transtion_probabilities = {}
        self.logs = {}
        for log_tuple in logs_path_dict:
            log = SimpleLogParser.read_log(log_tuple['path'])  ## dir_ + 'l' + str(j) + '.log'
            self.logs[log_tuple['id']] = log

    def get_next_logs_batch(self, k, logs2fetch=2, traces2produce=-1, batch_id = ""):

        if self.current_group_id:
            return

        logs, true_ksequence_transtion_probabilities = {}, []
        dir_ = self.output_path + "/" + batch_id + "/"
        if self.write_logs:
            create_folder_if_missing(dir_)

        logs_fetched = 0
        for log_id in self.logs:
            if logs_fetched >= logs2fetch:
                break
            traces = self.logs[log_id]
            if traces2produce != -1: ## if sampling, sample, run_mle, write_logs
                traces = sample_traces(traces, traces2produce)
                true_ksequence_transtion_probabilities.append(compute_mle_k_future_dict(traces, k))
                if self.write_logs:
                    LogWriter.write_log(traces, dir_ + 'l' + str(log_id) + ".log")
            else: ## if not sampling, run_mle if run for first time
                if len(self.true_ksequence_transtion_probabilities) < len(self.logs): ## only run mle once over complete log
                    self.true_ksequence_transtion_probabilities[log_id] = compute_mle_k_future_dict(traces, k)
                true_ksequence_transtion_probabilities.append(self.true_ksequence_transtion_probabilities[log_id])
            logs[log_id] = traces
            logs_fetched += 1

        self.current_group_id = 1
        return ExperimentBatch(logs, true_ksequence_transtion_probabilities, self.log_set_name)


class ModelBasedLogsManager(LogsManager):

    def __init__(self, models_json_path, write_logs=False):
        '''

        :param models_json_path:
        :param write_logs:
        '''
        super(ModelBasedLogsManager, self).__init__(write_logs)
        self.models = json.load(open(models_json_path))["models"]

    def get_next_logs_batch(self, k, logs2fetch=2, traces2produce=100, batch_id=""):

        if self.current_group_id == len(self.models):
            return

        models_info = self.models[self.current_group_id]
        print('processing', models_info["model"])
        logs, true_ksequence_transtion_probabilities = {}, []

        dir_ = models_info["logs_output_path"] + "/" + batch_id + "/"
        if self.write_logs:
            create_folder_if_missing(dir_ )
        logs_fetched = 0
        for j in range(logs2fetch):
            if logs_fetched >= logs2fetch:
                break
            ## use existing model to produce log
            model_path = models_info["model_folder"] + "/" + models_info["fname"]
            retries = 0
            while True:
                try:
                    model = ProtocolModel(model_path, id=j, assign_transtion_probs=True)
                    retries+=1
                    break
                except:
                    print('error when assigning weights to model, retrying')
                    if retries > 3:
                        raise AssertionError('cannot assign probabilities to model '+  model_path )
            log = LogGenerator.produce_log_from_model(model.graph,
                                                      transition_probability_attribute=TRANSITION_PROBABILITY_ATTRIBUTE,
                                                      traces2produce=traces2produce)
            logs[models_info["model"] + '_log_'+str(j)] = log
            if self.write_logs:
                model.write_transitions_probabilities(dir_)
                LogWriter.write_log(log, dir_ + 'l' + str(j) + ".log")
            ## compute k_sequences transition probabilities
            k_seqs2k_seqs = compute_k_sequence_transition_probabilities(model, k)
            true_ksequence_transtion_probabilities.append(k_seqs2k_seqs)
            logs_fetched+=1

        self.current_group_id += 1
        return ExperimentBatch(logs, true_ksequence_transtion_probabilities, models_info["model"])
