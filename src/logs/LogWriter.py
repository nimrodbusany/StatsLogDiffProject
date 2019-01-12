
class LogWriter:

    TRACE_SEPARATOR = '--'

    @classmethod
    def write_log(cls, traces, outpath):
        with open(outpath, 'w') as fw:
            for tr in traces:
                for ev in tr:
                    fw.write(ev + "\n")
                fw.write(cls.TRACE_SEPARATOR + "\n")
