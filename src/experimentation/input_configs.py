class Models:
    def __init__(self, models_folder, logs_folder, models, dirs_):
        self.models_folder = models_folder
        self.logs_folder = logs_folder
        self.models = models
        self.dirs_ = dirs_

def get_models_location(models_group=0):
    if models_group:
        models = ['ctas.net.simplified.no_loops.dot', 'cvs.net.no_loops.dot', 'roomcontroller.simplified.net.dot',
                  'ssh.net.simplified.dot', 'tcpip.simplified.net.dot']
        dirs_ = ['ctas.net', 'cvs.net', 'roomcontroller', 'ssh', 'tcpip']
        MODELS_PATH = '../../models/stamina/'
        LOGS_OUTPUT_PATH = '../../data/logs/stamina/'
    else:
        MODELS_PATH = '../../models/david/'
        LOGS_OUTPUT_PATH = '../../data/logs/david/'
        models = ['Columba.simplified.dot', 'Heretix.simplified.dot', 'JArgs.simplified.dot',
                  'Jeti.Simplified.dot', 'jfreechart.Simplified.dot', 'OpenHospital.Simplified.dot',
                  'RapidMiner.Simplified.dot', 'tagsoup.Simplified.dot']
        dirs_ = ['Columba', 'Heretix', 'JArgs', 'Jeti', 'jfreechart', 'OpenHospital', 'RapidMiner', 'tagsoup']
    return Models(MODELS_PATH, LOGS_OUTPUT_PATH, models, dirs_)
