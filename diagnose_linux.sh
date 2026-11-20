#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
mkdir -p diagnostics
OUT="diagnostics/linux_diagnostic.txt"

{
  echo "ILT Data Extraction & IAT — Linux diagnostic"
  echo "Date: $(date -Is 2>/dev/null || date)"
  echo "PWD: $PWD"
  echo
  echo "== System =="
  uname -a || true
  if [ -f /etc/os-release ]; then cat /etc/os-release; fi
  echo
  echo "== ILT runtime profile =="
  cat .ilt_linux_profile 2>/dev/null || echo "not configured"
  echo
  echo "== NVIDIA =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  else
    echo "nvidia-smi: not found"
  fi
  echo
  echo "== Files =="
  cat VERSION.txt 2>/dev/null || true
  find src -maxdepth 1 -type f -name '*.py' -printf '%f\n' 2>/dev/null | sort || true
  echo
  echo "== Python =="
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="$(command -v python3 || true)"; fi
  echo "Python: ${PY:-not found}"
  if [ -n "${PY:-}" ]; then
    "$PY" --version || true
    "$PY" -m pip --version || echo "pip: FAIL / ausente"
    PYTHONPATH="$PWD/src" "$PY" - <<'PY' || true
import importlib, platform, sys
print("Executable:", sys.executable)
print("Platform:", platform.platform())
mods=["numpy","pandas","scipy","sklearn","torch","torchvision","cv2","matplotlib","PIL","openTSNE","umap","pacmap","hdbscan","tkinter"]
for m in mods:
    try:
        x=importlib.import_module(m)
        print(f"{m}: OK {getattr(x,'__version__','')}")
    except Exception as e:
        print(f"{m}: FAIL {type(e).__name__}: {e}")
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("PyTorch CUDA runtime:", torch.version.cuda or "CPU-only")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass
PY

    echo
    echo "== PyInstaller build readiness =="
    "$PY" - <<'PY' || true
import ctypes.util, pathlib, sys, sysconfig
ver=f"{sys.version_info.major}.{sys.version_info.minor}"
names=[sysconfig.get_config_var("INSTSONAME"), sysconfig.get_config_var("LDLIBRARY"),
       f"libpython{ver}.so.1.0", f"libpython{ver}.so"]
dirs=[sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LIBPL"),
      str(pathlib.Path(sys.base_prefix)/"lib"), "/usr/lib", "/usr/local/lib"]
multiarch=sysconfig.get_config_var("MULTIARCH")
if multiarch:
    dirs += [f"/usr/lib/{multiarch}", f"/usr/local/lib/{multiarch}"]
found=[]
for d in filter(None, dirs):
    for n in filter(None, names):
        p=pathlib.Path(d)/n
        if p.exists():
            found.append(str(p))
loader=ctypes.util.find_library(f"python{ver}")
print("Python minor:", ver)
print("Py_ENABLE_SHARED:", sysconfig.get_config_var("Py_ENABLE_SHARED"))
print("Shared lib:", found[0] if found else (loader or "NOT FOUND"))
if not found and not loader:
    print(f"Build PyInstaller: FAIL - instale libpython{ver}")
else:
    print("Build PyInstaller shared-library preflight: OK")
PY
    echo
    echo "== py_compile =="
    PYTHONPATH="$PWD/src" "$PY" -m compileall -q src && echo "compileall: OK" || echo "compileall: FAIL"
    echo
    echo "== regression suite =="
    PYTHONPATH="$PWD/src" "$PY" run_regression_tests.py || true
  fi
} >"$OUT" 2>&1

echo "Diagnóstico salvo em:"
echo "  $OUT"
