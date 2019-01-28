#!/usr/bin/python
import sys, getopt
import json
import os
from pprint import pprint

def process_input_files(argv):
    '''
    process the input parameters
    :return:
    dict of logs ids to path; output_dir; alpha parameter; delta parameter
    '''
    inputfile = ''
    alpha = ''
    delta = ''
    try:
        opts, args = getopt.getopt(argv, "ha:d:c:", ["input_config=", "alpha=", "delta="])
    except getopt.GetoptError:
        print('error parsing input parameters, call: test.py -c <input_config> -a <alpha> -d <delta>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -c <input_config> -a <alpha> -d <delta>')
            sys.exit()
        elif opt in ("-c", "--config"):
            inputfile = arg
        elif opt in ("-a", "--alpha"):
            alpha = arg
        elif opt in ("-d", "--delta"):
            delta = arg

    with open(inputfile) as f:
        data = json.load(f)

    log_paths = {}
    for log in data["logs"]:

        if 'id' not in log:
            raise ValueError('missing id value in', log)
        if 'path' not in log:
            raise ValueError('missing path value in', log)
        if not os.path.exists(log['path']):
            raise ValueError('log file does not exists', log['id'], log['path'])
        if log['id'] in log_paths:
            raise ValueError('log id must be uniuqe, the following value appears more than once', log)

        log_paths[log['id']] = log['path']

    if 'output_dir' not in data:
        raise ValueError('output_dir missing form config_file')

    output_dir = data['output_dir']
    if not os.path.exists(output_dir):
        raise ValueError('output_dir does missing', output_dir)
    if not os.path.isdir(output_dir):
        raise ValueError('output_dir does map to a dir', output_dir)
    return log_paths, output_dir, alpha, delta

def main(argv):

    log_paths, output_dir, alpha, delta = process_input_files(argv)
    print('Input log files:')
    for log in log_paths:
        print(log, ':', log_paths[log])
    print('Output dir:', output_dir)
    print('Alpha =', alpha)
    print('Delta =', delta)


if __name__ == "__main__":
   main(sys.argv[1:])