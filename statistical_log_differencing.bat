echo %PYTHONPATH%
set PYTHONPATH=%~dp0
python "src/main/log_diff_runner.py" -c paper_examples/configurations/input_config_s2kdiff_case_study_2.json -a 0.05 -d 0.01 -k 1 -r 2
