
class SimpleLogParser:

    TRACE_SEPARATOR = '--'

    @staticmethod
    def read_log(log_path):

        traces = []
        with open(log_path) as fr:
            lines = [l.strip() for l in fr.readlines()]
            tr = []
            i = 0
            for l in lines:
                if l.strip() == SimpleLogParser.TRACE_SEPARATOR:
                    if len(tr) > 0:
                        traces.append(tr)
                        tr = []
                    continue
                tr.append(l)
                i+=1
        print('-> Done parsing ', log_path, 'read', len(traces), 'traces')
        return traces