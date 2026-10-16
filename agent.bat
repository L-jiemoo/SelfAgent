@echo off
setlocal
chcp 65001 >nul
"%~dp0agent.exe" %*
set ERR=%ERRORLEVEL%
endlocal & exit /b %ERR%
