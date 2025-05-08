@echo off
SET PYTHONPATH=%~dp0
echo Starting backend with PYTHONPATH=%PYTHONPATH%
call venv\Scripts\activate
venv\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
echo.
pause >nul