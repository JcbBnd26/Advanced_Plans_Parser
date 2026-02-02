@echo off
cd /d "C:\Projects\Advanced_Plan_Parser"
call .venv\Scripts\activate.bat
start "" pythonw scripts/gui.py
