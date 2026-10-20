@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
pushd "%~dp0"
if errorlevel 1 exit /b 1
if exist ".venv\Scripts\python.exe" goto venv
py -3.13 -c "import sys" >nul 2>&1
if not errorlevel 1 goto py313
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 goto py312
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 goto py311
python -c "import sys; sys.exit(not (3,11) <= sys.version_info[:2] <= (3,13))" >nul 2>&1
if not errorlevel 1 goto python
echo Python 3.11, 3.12 or 3.13 required. Open START_HERE.html.
popd
exit /b 1
:venv
".venv\Scripts\python.exe" %*
goto done
:py313
py -3.13 %*
goto done
:py312
py -3.12 %*
goto done
:py311
py -3.11 %*
goto done
:python
python %*
:done
set "COURSE_EXIT=%ERRORLEVEL%"
popd
exit /b %COURSE_EXIT%
