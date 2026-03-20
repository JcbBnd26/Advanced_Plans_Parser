@echo off
cd /d "C:\Projects\Advanced_Plan_Parser"
call .venv\Scripts\activate.bat
python -m scripts.gui.gui
if errorlevel 1 (
    echo.
    echo GUI exited with an error. Check logs\gui_crash.txt for details.
    if exist logs\gui_crash.txt type logs\gui_crash.txt
    pause
)
