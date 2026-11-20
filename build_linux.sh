#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

PY="$PWD/.venv/bin/python"
BUILDLOG="$PWD/build_linux.log"

: > "$BUILDLOG"
{
  echo "============================================================"
  echo "ILT Data Extraction - Build Linux"
  echo "Date: $(date -Is 2>/dev/null || date)"
  echo "PWD: $PWD"
  echo "============================================================"
} >>"$BUILDLOG"

fail() {
  echo "ERRO: $*" | tee -a "$BUILDLOG" >&2
  exit 1
}

if [ ! -x "$PY" ]; then
  fail ".venv não encontrado. Execute ./setup_linux.sh primeiro."
fi

PY_MM="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
LIBPY_PKG="libpython${PY_MM}"

# PyInstaller on GNU/Linux embeds/collects the Python shared library.
# A Python interpreter can run normally even when that shared library
# package is absent (a common case with parallel Python installs on Ubuntu).
has_shared_libpython() {
  "$PY" - <<'PY' >/dev/null 2>&1
import ctypes.util
import pathlib
import sys
import sysconfig

ver = f"{sys.version_info.major}.{sys.version_info.minor}"
names = {
    sysconfig.get_config_var("INSTSONAME"),
    sysconfig.get_config_var("LDLIBRARY"),
    f"libpython{ver}.so.1.0",
    f"libpython{ver}.so",
}
names.discard(None)
names.discard("")

dirs = {
    sysconfig.get_config_var("LIBDIR"),
    sysconfig.get_config_var("LIBPL"),
    str(pathlib.Path(sys.base_prefix) / "lib"),
    "/usr/lib",
    "/usr/local/lib",
}
multiarch = sysconfig.get_config_var("MULTIARCH")
if multiarch:
    dirs.add(f"/usr/lib/{multiarch}")
    dirs.add(f"/usr/local/lib/{multiarch}")

for directory in filter(None, dirs):
    base = pathlib.Path(directory)
    for name in names:
        if (base / name).exists():
            raise SystemExit(0)

# ldconfig/loader-visible libraries are also acceptable.
if ctypes.util.find_library(f"python{ver}"):
    raise SystemExit(0)

raise SystemExit(1)
PY
}

show_shared_libpython() {
  "$PY" - <<'PY'
import ctypes.util
import pathlib
import sys
import sysconfig

ver = f"{sys.version_info.major}.{sys.version_info.minor}"
names = [
    sysconfig.get_config_var("INSTSONAME"),
    sysconfig.get_config_var("LDLIBRARY"),
    f"libpython{ver}.so.1.0",
    f"libpython{ver}.so",
]
dirs = [
    sysconfig.get_config_var("LIBDIR"),
    sysconfig.get_config_var("LIBPL"),
    str(pathlib.Path(sys.base_prefix) / "lib"),
    "/usr/lib",
    "/usr/local/lib",
]
multiarch = sysconfig.get_config_var("MULTIARCH")
if multiarch:
    dirs += [f"/usr/lib/{multiarch}", f"/usr/local/lib/{multiarch}"]

seen = set()
for directory in filter(None, dirs):
    for name in filter(None, names):
        p = pathlib.Path(directory) / name
        if p not in seen and p.exists():
            print(p)
            raise SystemExit(0)
        seen.add(p)

found = ctypes.util.find_library(f"python{ver}")
if found:
    print(found)
else:
    print("(não localizada)")
PY
}

install_libpython_if_possible() {
  command -v apt-get >/dev/null 2>&1 || return 1
  command -v apt-cache >/dev/null 2>&1 || return 1

  if ! apt-cache show "$LIBPY_PKG" >/dev/null 2>&1; then
    echo "Pacote $LIBPY_PKG não está disponível nos repositórios configurados." | tee -a "$BUILDLOG"
    return 1
  fi

  local sudo_cmd=()
  if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || return 1
    sudo_cmd=(sudo)
  fi

  echo | tee -a "$BUILDLOG"
  echo "O PyInstaller precisa da biblioteca compartilhada do Python $PY_MM." | tee -a "$BUILDLOG"
  echo "Pacote detectado no Ubuntu/Debian: $LIBPY_PKG" | tee -a "$BUILDLOG"
  echo | tee -a "$BUILDLOG"

  local answer="s"
  if [ -t 0 ]; then
    read -r -p "Instalar $LIBPY_PKG automaticamente? [S/n]: " answer
    answer="${answer:-s}"
  fi
  case "$answer" in
    s|S|sim|SIM|y|Y|yes|YES) ;;
    *)
      echo "Instalação automática cancelada." | tee -a "$BUILDLOG"
      return 1
      ;;
  esac

  echo "Instalando $LIBPY_PKG..." | tee -a "$BUILDLOG"
  "${sudo_cmd[@]}" apt-get update >>"$BUILDLOG" 2>&1
  "${sudo_cmd[@]}" apt-get install -y "$LIBPY_PKG" >>"$BUILDLOG" 2>&1
}

echo "[0/6] Verificando biblioteca compartilhada do Python..."
if ! has_shared_libpython; then
  {
    echo "Biblioteca compartilhada do Python $PY_MM não foi localizada."
    echo "Isso não impede o ILT de rodar por código-fonte, mas impede o PyInstaller de gerar a distribuição."
  } | tee -a "$BUILDLOG"

  if install_libpython_if_possible; then
    hash -r 2>/dev/null || true
  fi

  if ! has_shared_libpython; then
    {
      echo
      echo "Não foi possível disponibilizar libpython${PY_MM}.so."
      echo
      echo "No Ubuntu/Debian, instale o pacote correspondente e rode o build novamente:"
      echo "  sudo apt update"
      echo "  sudo apt install $LIBPY_PKG"
      echo
      echo "Se esse pacote não existir, o Python $PY_MM provavelmente veio de um repositório/instalação"
      echo "paralela que não fornece a shared library. Nesse caso, execute ./setup_linux.sh usando"
      echo "o runtime local Miniforge e tente o build novamente."
      echo
      echo "Nenhum arquivo do ILT foi alterado."
    } | tee -a "$BUILDLOG"
    exit 1
  fi
fi
echo "libpython: $(show_shared_libpython)" | tee -a "$BUILDLOG"

echo "[1/6] Verificando dependências essenciais..."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - <<'PY' >>"$BUILDLOG" 2>&1
import numpy,pandas,cv2,matplotlib,scipy,sklearn,torch,torchvision,openTSNE,PIL,psutil,tqdm,umap,pacmap
import iat_backend,iat_workspace,ilt_gui,features,projections,data,batch,ilt_contracts,ilt_clustering
from sklearn.cluster import HDBSCAN
print("Preflight OK")
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[2/6] Compilando e executando regressões..."
"$PY" -m compileall -q src
./run_regression_tests.sh

echo "[3/6] Instalando dependências de build..."
"$PY" -m pip install -r requirements-linux-build.txt >>"$BUILDLOG" 2>&1

# Double-check after installing PyInstaller so failures happen in preflight,
# not halfway through Analysis.
if ! has_shared_libpython; then
  fail "libpython${PY_MM}.so deixou de estar disponível após o preflight."
fi

rm -rf build dist/ILT_Data_Extraction
rm -f ILT_SELF_TEST_FAILED.txt

echo "[4/6] Gerando distribuição..."
"$PY" -m PyInstaller --clean --noconfirm ILT_Data_Extraction.spec >>"$BUILDLOG" 2>&1

DIST="$PWD/dist/ILT_Data_Extraction"
EXE="$DIST/ILT_Data_Extraction"
if [ ! -x "$EXE" ]; then
  fail "executável final não foi criado. Consulte $BUILDLOG."
fi

mkdir -p "$DIST/input" "$DIST/output" "$DIST/assets/mascots"
[ -f input/train_params.json ] && cp input/train_params.json "$DIST/input/"
if [ -d assets/mascots ]; then
  cp -a assets/mascots/. "$DIST/assets/mascots/" 2>/dev/null || true
fi

echo "[5/6] Executando autoteste do executável..."
if ! "$EXE" --ilt-self-test >>"$BUILDLOG" 2>&1; then
  fail "o executável foi criado, mas falhou no autoteste. Consulte $BUILDLOG."
fi

echo "[6/6] Finalizando diagnóstico de build..."
mkdir -p diagnostics
[ -f build/ILT_Data_Extraction/warn-ILT_Data_Extraction.txt ] && \
  cp build/ILT_Data_Extraction/warn-ILT_Data_Extraction.txt diagnostics/ || true
rm -rf build

echo
echo "============================================================"
echo "BUILD LINUX VALIDADO COM SUCESSO"
echo "============================================================"
echo "Execute:"
echo "  $EXE"
echo
echo "Para outro computador, copie a pasta inteira:"
echo "  $DIST"
