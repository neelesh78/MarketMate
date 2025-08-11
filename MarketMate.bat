@echo off
title MarketMate Launcher
cd /d "%~dp0"

echo.
echo ========================================
echo  MarketMate - Portfolio Tracker
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python not found! Please install Python 3.8+ and add it to PATH.
    pause
    exit /b 1
)

echo Python found
echo.

REM Check if required files exist
if not exist "marketmate.py" (
    echo  MarketMate files not found in current directory!
    pause
    exit /b 1
)
if not exist "launcher.py" (
    echo  Launcher file not found! Please ensure you are in the MarketMate directory.
    pause
    exit /b 1
)
echo  MarketMate files found
echo.

REM Launch options menu
:menu
echo Choose launch option:
echo.
echo 1.  Launch Launcher GUI (Recommended)
echo 2.  Launch Desktop GUI
echo 3.  Launch Web Interface
echo 4.  Run Command Line Analysis
echo 5.  Run Setup
echo 6.  Run Tests
echo 7.  Show Help
echo 8.  Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto launcher
if "%choice%"=="2" goto gui
if "%choice%"=="3" goto web
if "%choice%"=="4" goto cli
if "%choice%"=="5" goto setup
if "%choice%"=="6" goto test
if "%choice%"=="7" goto help
if "%choice%"=="8" goto exit
echo Invalid choice! Please enter 1-8.
echo.
goto menu

:launcher
echo.
echo Launching MarketMate Launcher...
python launcher.py
goto end

:gui
echo.
echo Launching Desktop GUI...
python marketmate_gui.py
goto end

:web
echo.
echo 🌐 Starting Web Interface...
echo Opening browser to http://localhost:8501
start "" "http://localhost:8501"
python -m streamlit run marketmate_ui.py
goto end

:cli
echo.
echo  Running Command Line Analysis...
python marketmate.py
echo.
echo Analysis complete! Check the results/ folder for output files.
pause
goto menu

:setup
echo.
echo  Running Setup...
python setup.py
pause
goto menu

:test
echo.
echo  Running Tests...
python test_marketmate.py
pause
goto menu

:help
echo.
echo MarketMate Help
echo.
echo MarketMate is a comprehensive portfolio analysis tool with multiple interfaces:
echo.
echo • Launcher GUI: Central hub for all MarketMate features
echo • Desktop GUI: Modern tkinter-based desktop application  
echo • Web Interface: Interactive Streamlit dashboard in browser
echo • Command Line: Quick analysis with default settings
echo.
echo For detailed documentation, see MarketMate_README.md
echo.
pause
goto menu

:exit
echo.
echo Thanks for using MarketMate! 👋
exit /b 0

:end
echo.
echo Press any key to return to menu...
pause >nul
goto menu
