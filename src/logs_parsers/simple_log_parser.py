
class SimpleLogParser:

    TRACE_SEPARATOR = '--'

    @staticmethod
    def read_log(log_path):

        traces = []
        with open(log_path) as fr:
            lines = [l.strip() for l in fr.readlines()]
            tr = []
            for l in lines:
                if l == SimpleLogParser.TRACE_SEPARATOR:
                    if len(tr) > 0:
                        traces.append(tr)
                        tr = []
                    continue
                tr.append(l)
        return traces