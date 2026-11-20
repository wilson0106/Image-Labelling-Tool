@echo off
REM Gera diagnostics\windows_diagnostic.txt para suporte.
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"
if not exist diagnostics mkdir diagnostics
set "OUT=diagnostics\windows_diagnostic.txt"
echo ILT 1.1.0 - diagnostico Windows > "%OUT%"
echo Data: %DATE% %TIME% >> "%OUT%"
echo Pasta: %CD% >> "%OUT%"
ver >> "%OUT%"
echo. >> "%OUT%"
echo == Perfil == >> "%OUT%"
type .ilt_windows_profile >> "%OUT%" 2>nul
echo. >> "%OUT%"
echo == NVIDIA == >> "%OUT%"
nvidia-smi >> "%OUT%" 2>&1
echo. >> "%OUT%"
echo == Python / PyTorch == >> "%OUT%"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" src\gpu_profile.py >> "%OUT%" 2>&1
  ".venv\Scripts\python.exe" -c "import sys,torch;print(sys.version);print('torch',torch.__version__,'cuda',torch.version.cuda,'disponivel',torch.cuda.is_available())" >> "%OUT%" 2>&1
  ".venv\Scripts\python.exe" -m pip list >> "%OUT%" 2>&1
) else (
  echo .venv ausente - rode INSTALAR.bat >> "%OUT%"
)
echo. >> "%OUT%"
echo == Ultimas linhas do setup == >> "%OUT%"
powershell -NoProfile -Command "if (Test-Path setup_windows.log) { Get-Content setup_windows.log -Tail 60 }" >> "%OUT%" 2>&1
echo Diagnostico salvo em %OUT%
pause
