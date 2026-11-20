@echo off
REM ILT 1.1.0 - instala ou reconfigura o ambiente (detecta a placa de video).
cd /d "%~dp0"
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 (
  echo.
  echo A instalacao nao foi concluida. Veja setup_windows.log ou rode DIAGNOSTICO.bat.
)
echo.
pause
