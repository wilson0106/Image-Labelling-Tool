#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

chmod +x setup_linux.sh run_gui.sh diagnose_linux.sh 2>/dev/null || true

echo "============================================================"
echo " ILT Data Extraction & IAT — Linux / Ubuntu 24.04 / 26.04"
echo " v1.1.0 (estável)"
echo " GPU detectada automaticamente (NVIDIA CUDA ou CPU)"
echo "============================================================"
echo

NEED_SETUP=0
if [ ! -x ".venv/bin/python" ]; then
  NEED_SETUP=1
elif ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  echo ".venv existente não possui pip funcional; ela será recriada."
  NEED_SETUP=1
elif ! .venv/bin/python -c 'import tkinter' >/dev/null 2>&1; then
  echo ".venv existente não possui Tkinter; ela será recriada."
  NEED_SETUP=1
elif ! .venv/bin/python -c 'import numpy' >/dev/null 2>&1; then
  echo ".venv existente está incompleta; o setup será executado novamente."
  NEED_SETUP=1
fi

if [ "$NEED_SETUP" = "1" ]; then
  echo "Configurando o ambiente local..."
  ./setup_linux.sh
else
  echo ".venv encontrada e Tkinter disponível."
  if [ -f ".ilt_linux_profile" ]; then
    echo "Perfil configurado:"
    sed -n 's/^profile=/  /p' .ilt_linux_profile
    sed -n 's/^python=/  Python: /p' .ilt_linux_profile
    sed -n 's/^env_kind=/  Ambiente: /p' .ilt_linux_profile
  fi
  echo
  echo "Trocou de placa de vídeo ou atualizou o driver? Rode de novo:"
  echo "  ./setup_linux.sh"
fi

echo
echo "Para abrir o ILT:"
echo "  ./run_gui.sh"
echo
echo "Se houver falha:"
echo "  ./diagnose_linux.sh"
