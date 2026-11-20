#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

VENV_PY="$PWD/.venv/bin/python"

test_env_ready() {
  [ -x "$VENV_PY" ] || return 1
  "$VENV_PY" -m pip --version >/dev/null 2>&1 || return 1
  "$VENV_PY" - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ("numpy", "tkinter")
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
}

if ! test_env_ready; then
  echo "Ambiente ILT ausente ou incompleto."
  echo "Executando setup_linux.sh antes da suíte de regressão..."
  ./setup_linux.sh
fi

test_env_ready || {
  echo "ERRO: o ambiente continua incompleto após o setup."
  echo "Execute ./diagnose_linux.sh e envie diagnostics/linux_diagnostic.txt."
  exit 1
}

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
echo "ILT regression suite — $("$VENV_PY" --version 2>&1)"
"$VENV_PY" run_regression_tests.py
