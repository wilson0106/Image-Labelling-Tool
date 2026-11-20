#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

VENV_PY="$PWD/.venv/bin/python"
DIST_EXE="$PWD/dist/ILT_Data_Extraction/ILT_Data_Extraction"

check_env() {
  [ -x "$VENV_PY" ] || return 1
  "$VENV_PY" -m pip --version >/dev/null 2>&1 || return 1
  "$VENV_PY" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
mods = (
    "numpy","pandas","cv2","matplotlib","scipy","sklearn","torch","torchvision",
    "openTSNE","PIL","psutil","tqdm","umap","pacmap","tkinter"
)
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
  "$VENV_PY" -c "import numpy" >/dev/null 2>&1
}

UI_MODE="web"
for arg in "$@"; do
  [ "$arg" = "--classic" ] && UI_MODE="classic"
done

run_source() {
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
  # Nova interface (navegador local). Sem Dash/Plotly, ou com --classic,
  # abre a interface clássica em Tk, que continua completa.
  if [ "$UI_MODE" = "web" ] && "$VENV_PY" -c "import dash, plotly" >/dev/null 2>&1; then
    "$VENV_PY" src/ilt_web.py
  else
    "$VENV_PY" src/ilt_gui.py
  fi
}

if [ -x "$VENV_PY" ]; then
  if ! check_env; then
    echo
    echo "============================================================"
    echo "O ambiente Python existe, mas está incompleto."
    echo "O ILT vai executar o setup Linux para reparar dependências."
    echo "============================================================"
    echo
    ./setup_linux.sh
    check_env || {
      echo "ERRO: o ambiente continua incompleto após o setup."
      echo "Execute ./diagnose_linux.sh e envie diagnostics/linux_diagnostic.txt."
      exit 1
    }
  fi
  run_source
  exit $?
fi

if [ -x "$DIST_EXE" ]; then
  exec "$DIST_EXE"
fi

echo
echo "============================================================"
echo "O ambiente Python ainda não foi configurado."
echo "O ILT vai iniciar o setup Linux agora."
echo "============================================================"
echo
./setup_linux.sh
check_env || {
  echo "ERRO: não foi possível preparar um ambiente Python completo."
  echo "Consulte setup_linux.log e execute ./diagnose_linux.sh se necessário."
  exit 1
}
run_source
