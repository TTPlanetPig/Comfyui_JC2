@echo off

REM 设置 Python 路径
set PYTHON_PATH=..\..\..\python_embeded\python.exe

echo Installing liger-kernel without dependencies...
%PYTHON_PATH% -m pip install liger-kernel==0.5.9 --no-deps

echo Installing remaining dependencies from requirements.txt...
%PYTHON_PATH% -m pip install -r requirements.txt

echo All done.
pause
