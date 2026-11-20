@echo off
REM ILT 1.1.0 - abre o programa (interface no navegador).
REM   ILT.bat            interface nova (navegador)
REM   ILT.bat --classic  interface classica (Tk)
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0src"
set "VPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo Primeira execucao: instalando o ambiente do ILT...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
  if errorlevel 1 (
    echo A instalacao falhou. Veja setup_windows.log.
    pause
    exit /b 1
  )
)
if /i "%~1"=="--classic" goto classic
"%VPY%" -c "import dash, plotly" >nul 2>&1
if errorlevel 1 goto classic
echo ILT abrindo no navegador. Mantenha esta janela aberta; para sair use o botao Encerrar.
"%VPY%" "%~dp0src\ilt_web.py"
goto end
:classic
"%VPY%" "%~dp0src\ilt_gui.py"
:end
endlocal
