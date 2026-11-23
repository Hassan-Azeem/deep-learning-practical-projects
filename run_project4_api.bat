@echo off
cd /d "%~dp0project4"
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
