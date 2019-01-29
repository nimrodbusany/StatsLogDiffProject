# StatsLogDiffProject
Log Differencing with Statistical Guarantees Project

This project includes the source code of the s2KDiff and snKDiff algorithms.

The algorithms are used to compare between two (s2KDiff)/ n (snKDiff) logs using finite state machines. 
To run the algorithms, first create a json config file referncing logs file paths and output_dir.
Then, edit and run the statistical_log_differencing.bat file, after setting:
-c (config file), e.g., input_config.json 
-a (significance value), default, 0.05 
-d (the minimal difference that is used by the statistical test to consider a difference as interesting), default, 0.05 
-k k-Tails k parameter, default, 2

- Example logs can be found in StatsLogDiffProject/data/logs/logs.zip. 
- Example json files configs.json appear in the main folder (input_config_s2kdiff.json, input_config_snkdiff.json).

If the config file includes 2 logs then 2sKDiff is run, if more, snKDiff is run.

The paper describing the algorithms can be found:

Requirements:
Python 3+, python path is added to Path, modules: numpy, pandas, networkx, graphviz (https://pypi.org/project/graphviz/)
