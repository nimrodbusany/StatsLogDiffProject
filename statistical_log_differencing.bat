@echo off
echo %PYTHONPATH%
set PYTHONPATH=%~dp0

set foo=%1
ARGS=("$@")
::echo input: %foo%
::for /f "tokens=1,2 delims=:" %%a in ("%foo%") do set name=%%a & set val=%%b
::echo name:  %name%
::echo value: %val%

echo name:  %foo%
echo name:  ${ARGS[-1]

:: python "src/main/log_diff_runner.py" -c paper_examples/configurations/input_config_s2kdiff_case_study_2.json -a 0.05 -d 0.2 -k 1 -r 2
