@echo off
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_studio_server.ps1" %*
if errorlevel 1 (
    echo.
    echo [ERROR] Studio server exited with an error.
    pause
)
