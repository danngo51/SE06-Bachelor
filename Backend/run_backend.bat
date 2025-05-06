@echo off
SET PYTHONPATH=%~dp0ml_models\informer\informerModel
echo Starting backend with PYTHONPATH=%PYTHONPATH%
echo ----------------------------------------------

call venv\Scripts\activate
venv\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

echo ----------------------------------------------
echo.
echo Press any key to close...
pause >nul
