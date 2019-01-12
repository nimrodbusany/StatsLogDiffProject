from datetime import datetime

import URLFilter
import UserClassMapper
from BearEvent import BEAREvent

from bear.URLMapper import *


class BearLogParser:

    USER_WINDOW = 3600

    def __init__(self, path):
        self.path = path

    def _print_set(self, set_, label, n=10):
        print("PRINTING", label, ", total values", len(set_), ", printing top", n, ":")
        i = 0
        for v in set_:
            print(v)
            i += 1
            if i > n:
                break

    def _break_events_to_traces(self):
        ## uses an IP and a user window of 60 minutes to break events to traces
        ips2events = {}
        for ev in self.events:
            ## user are characterized by ip
            ip_events = ips2events.get(ev.ip, [])
            ip_events.append(ev)
            ips2events[ev.ip] = ip_events

        self.traces = []
        for ip in ips2events:
            ## if two request from the same ip are separated by more than USER_WINDOW,
            ## coniser them as different traces
            tr = []
            ip_events = ips2events[ip]
            ip_events.sort(key=lambda x: x.time)
            for i in range(len(ip_events)-1):
                tr.append(ip_events[i])
                delta = ip_events[i+1].time - ip_events[i].time
                if delta.seconds > self.USER_WINDOW:
                    self.traces.append(tr)
                    tr = []
            tr.append(ip_events[-1])
            self.traces.append(tr)

        return self.traces

    def process_log(self, verbose= False):

        urls = set()
        user_classes = set()
        req_types = set()
        ips = set()
        labels = set()

        with open(self.path) as fr:

            # filter lines with urls that should not be included
            lines = fr.readlines()
            lines2keep = [l for l in lines if not URLFilter.is_filtered_url(l)]
            print('only kept:', len(lines2keep), 'out of', len(lines), 'due to url filtering')

            self.events = []
            mapper = URLMapper()
            filter_events = 0

            for l in lines2keep:
                parts = l.strip().split()
                ip = parts[1]
                time = parts[4].strip('[') + " " + parts[5].strip(']')
                time = datetime.strptime(time, "%d/%b/%Y:%H:%M:%S %z")
                req_type = parts[6].strip("\"")
                url = parts[7]
                label = mapper.get_url_label(url, req_type, l)  # map urls to labels, used as event name

                if not label:  ## only keep events with labels
                    filter_events += 1
                    continue
                user_class = UserClassMapper.extract_user_class(l)
                # for p in parts:
                if verbose:
                    urls.add(url)
                    user_classes.add(user_class)
                    req_types.add(req_type)
                    ips.add(ip)
                    labels.add(label)
                self.events.append(BEAREvent(ip, time, req_type, url, label, user_class, l))
            print('extracted', len(self.events), 'events, filtered', filter_events, 'due to a missing labels')

            # break events according to time window of 60 minutes (according to the paper)
            self._break_events_to_traces()
            if verbose:
                self._print_set(urls, "urls")
                self._print_set(user_classes, "user_classes")
                self._print_set(req_types, "req_types")
                self._print_set(ips, "ips")
                self._print_set(labels, "labels")
        return self.traces

    def get_traces_of_browser(self, traces, browser):
        return [tr for tr in traces if browser in tr[0].user_class.split()[0]]


    def get_desktop_traces(self, traces):
        DESKTOP_OS = ['Windows NT', 'Windows 98', 'Macintosh']
        return [tr for tr in traces for os in DESKTOP_OS if os in tr[0].user_class]

    def get_mobile_traces(self, traces):
        MOBILE_OS = ['iPhone', 'iPad', 'Symbian', 'iPod', 'Android', 'SymbianOS', 'BlackBerry', 'Windows Phone', 'Mobile Safari']
        return [tr for tr in traces for os in MOBILE_OS if os in tr[0].user_class]




    def abstract_events(self, new_name_mapping, traces):
        for tr in traces:
            for i in range(len(tr)):
                ev = tr[i]
                if ev.label in new_name_mapping:
                    ev.label = new_name_mapping[ev.label]
        return traces

    def filter_events(self, events_names, traces, keep_provided_events= False):
        '''

        :param events_names: list of events name to either remove or only include
        :param traces:
        :param keep_provided_events: if true, only keeps the events in events_names;
        if false remove events in events_names
        :return: filtered traces
        '''

        empty_traces = []
        for j in range(len(traces)):
            tr = traces[j]
            ind2remove = []
            for i in range(len(tr)):
                ev = tr[i]
                filtering_condition = ev.label not in events_names if keep_provided_events else ev.label in events_names
                if filtering_condition:
                    ind2remove.append(i)
            for ind in sorted(ind2remove, reverse=True):
                del tr[ind]
            if len(tr) == 0:
                empty_traces.append(j)

        for ind in sorted(empty_traces, reverse=True):
            del traces[ind]
        return traces

    def get_traces_as_lists_of_event_labels(self, traces=None):

        if not traces:
            traces = self.traces
        traces_as_list = []
        for tr in traces:
            tr_as_list = []
            for ev in tr:
                tr_as_list.append(ev.label)
            traces_as_list.append(list(tr_as_list))
        return traces_as_list

if __name__ == '__main__':

    LOG_PATH = '../../data/bear/findyourhouse_long.log'
    log_parser = BearLogParser(LOG_PATH)
    log_parser.process_log()

    log_parser = BearLogParser(LOG_PATH)
    traces = log_parser.process_log(True)
    log1 = log_parser.get_traces_of_browser("Mozilla/4.0")
    log1 = log_parser.get_traces_as_lists_of_event_labels(log1)
    log2 = log_parser.get_traces_of_browser("Mozilla/5.0")
    log2 = log_parser.get_traces_as_lists_of_event_labels(log2)