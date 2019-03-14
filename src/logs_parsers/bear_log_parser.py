from datetime import datetime

import src.bear.url_filter as url_filter
import src.bear.user_class_mapper as user_class_mapper
from src.bear.bear_event import BEAREvent
from src.bear.url_mapper import *


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
                diffrent_user_class = ip_events[i+1].user_class != ip_events[i].user_class
                if diffrent_user_class or delta.seconds > self.USER_WINDOW:
                    self.traces.append(tr)
                    tr = []
            tr.append(ip_events[-1])
            self.traces.append(tuple(tr))

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
            lines2keep = [l for l in lines if not url_filter.is_filtered_url(l)]
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
                user_class = user_class_mapper.extract_user_class(l)
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

    def get_non_user_traces(self, traces):
        UsersOS = ['iPhone', 'iPad', 'Symbian', 'iPod', 'Android', 'SymbianOS', 'BlackBerry', 'Windows Phone', 'Mobile Safari', 'Windows NT', 'Windows 98', 'Macintosh']
        non_user_traces_ = []
        for tr in traces:
            found_os = False
            for os in UsersOS:
                if os in tr[0].user_class:
                    found_os = True
                    break
            if not found_os:
                non_user_traces_.append(tuple(tr))
        return non_user_traces_




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


def split_trace_in_half_by_time(traces, split_by_events=True):

    sorted_traces_by_date = sorted(traces, key=lambda tr: tr[0].time)
    if split_by_events:
        total_events = sum([1 for tr in traces for ev in tr])
        seen_events, mid_ = 0, 0
        while(seen_events) < total_events/2:
            seen_events += len(traces[mid_])
            mid_ += 1
        return sorted_traces_by_date[:mid_], sorted_traces_by_date[mid_:]
    else:
        mid_ = int(len(sorted_traces_by_date) / 2)
        return sorted_traces_by_date[:mid_], sorted_traces_by_date[mid_:]

def split_logs_to_two_and_write(traces):

    first_half, second_half = split_trace_in_half_by_time(traces)
    first_half_traces = []
    for tr in first_half:
        first_half_traces.append([ev.label for ev in tr])
    second_half_traces = []
    for tr in second_half:
        second_half_traces.append([ev.label for ev in tr])
    from log_writer import LogWriter
    LogWriter.write_log(first_half_traces, '../../data/bear/first_half_traces.log')
    LogWriter.write_log(second_half_traces, '../../data/bear/last_half_traces.log')


def split_traces_by_months_and_write(traces):

    months = {}
    for trace in traces:
        month_year = (trace[0].time.year, trace[0].time.month)
        if month_year not in months:
            months[month_year] = []
        trace_labels = [ev.label for ev in trace]
        months[month_year].append(trace_labels)
    from log_writer import LogWriter
    for m in months:
        print('month/year', m, 'had',len(months[m]), 'traces')
        LogWriter.write_log(months[m], '../../data/bear/'+str(m[0])+'_'+str(m[1])+'.log')


def split_traces_by_quarters_and_write(traces):

    quarters = {'Q1':[], 'Q2':[], 'Q3':[], 'Q4':[]}
    for trace in traces:
        if trace[0].time.year == 2011:
            continue
        trace_labels = [ev.label for ev in trace]
        if trace[0].time.month <= 3:
            quarters['Q1'].append(trace_labels)
        elif trace[0].time.month <= 6:
            quarters['Q2'].append(trace_labels)
        elif trace[0].time.month <= 9:
            quarters['Q3'].append(trace_labels)
        elif trace[0].time.month <= 12:
            quarters['Q4'].append(trace_labels)

    from log_writer import LogWriter
    for q in quarters:
        print('month/year', q, 'had',len(quarters[q]), 'traces')
        LogWriter.write_log(quarters[q], '../../data/bear/'+str(q)+'.log')

if __name__ == '__main__':

    LOG_PATH = '../../data/bear/findyourhouse_long.log'
    log_parser = BearLogParser(LOG_PATH)
    log_parser.process_log()

    log_parser = BearLogParser(LOG_PATH)
    traces = log_parser.process_log(True)
    mobile = log_parser.get_mobile_traces(traces)
    desktop = log_parser.get_desktop_traces(traces)
    non_user_traces = log_parser.get_non_user_traces(traces)
    # split_traces_by_months_and_write(traces)
    split_traces_by_quarters_and_write(traces)
    # split_logs_to_two_and_write(traces)
    # log1 = log_parser.get_traces_of_browser("Mozilla/4.0")
    # log1 = log_parser.get_traces_as_lists_of_event_labels(log1)
    # log2 = log_parser.get_traces_of_browser("Mozilla/5.0")
    # log2 = log_parser.get_traces_as_lists_of_event_labels(log2)