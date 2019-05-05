@echo off
:main

    set argCount=0
    for %%x in (%*) do (
       set /A argCount+=1
       set "argVec[!argCount!]=%%~x"
    )

    (for %%x in (%*) do (
        set val1=%%x
        echo process %val1%
        call :process_arg %val1% val3 val4
        call :process_alg %val3% %val4%
        call :process_k %val3% %val4%
    ))
    echo -k %k%
    echo -a %alg%
exit /b 0

:process_arg
set tup=%~1
for /f "tokens=1,2 delims=:" %%a in ("%tup%") do (
set name=%%a
set val=%%b
)
:setvalue
set "%~2=%name%"
set "%~3=%val%"
exit /b 0

:process_k
if %~1==k (
    set k=%~2%
    echo found k %k%
)
exit /b 0

:process_alg
if %~1==a (
    set alg=%~2%
    echo found alg %alg%
)
exit /b 0
