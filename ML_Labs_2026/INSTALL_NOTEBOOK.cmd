@echo off
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
echo Create the course environment: RUN.cmd -m venv .venv
pause
exit /b 1
:install
".venv\Scripts\python.exe" -m pip install --only-binary=:all: -r requirements-notebook.txt
popd
pause
