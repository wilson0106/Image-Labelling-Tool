#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

LOG="$PWD/setup_linux.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo " ILT Data Extraction & IAT 1.1.0 - Linux Ubuntu 24.04 / 26.04"
echo " Data/hora: $(date)"
echo " Pasta: $PWD"
echo "============================================================"
echo

if [ -f /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [ "${ID:-}" = "ubuntu" ]; then
        echo "Ubuntu detectado: ${VERSION_ID:-desconhecido}"
        case "${VERSION_ID:-}" in
            22.04|24.04|26.04) ;;
            *) echo "AVISO: esta versão foi validada para Ubuntu 24.04 e 26.04 (e 22.04)." ;;
        esac
        echo
    fi
fi

# O ILT foi estabilizado em Python 3.11/3.12 + PyTorch 2.5.1.
# Ubuntu 22.04/26.04 pode ter Python do sistema sem o conjunto completo
# necessário para criar um venv funcional (especialmente ensurepip/venv)
# ou sem Tkinter para a versão escolhida. O instalador valida a criação
# real de um venv com pip. Se ela falhar, cria Python 3.12 local via
# Miniforge, sem substituir o Python do sistema.

REQUESTED_PYTHON="${PYTHON:-}"
SYSTEM_PYTHON=""
ENV_KIND="venv"

py_minor() {
    "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

has_tk() {
    "$1" -c 'import tkinter; print(tkinter.TkVersion)' >/dev/null 2>&1
}

has_venv() {
    "$1" -c 'import venv' >/dev/null 2>&1
}

has_pip() {
    "$1" -m pip --version >/dev/null 2>&1
}

# Importar `venv` não garante que o Ubuntu tenha instalado o componente
# ensurepip/pythonX.Y-venv. Este teste cria um venv descartável e confirma
# que ele realmente nasce com pip funcional.
can_create_complete_venv() {
    local py="$1"
    local tmp_dir test_py rc=1
    tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/ilt-venv-check.XXXXXX")" || return 1
    if "$py" -m venv "$tmp_dir/venv" >/dev/null 2>&1; then
        test_py="$tmp_dir/venv/bin/python"
        if [ -x "$test_py" ] && has_pip "$test_py"; then
            rc=0
        fi
    fi
    rm -rf "$tmp_dir"
    return "$rc"
}

existing_env_bootstrap_ok() {
    local py="$PWD/.venv/bin/python"
    [ -x "$py" ] || return 1
    has_pip "$py" || return 1
    has_tk "$py" || return 1
}

detect_python() {
    if [ -n "$REQUESTED_PYTHON" ]; then
        if command -v "$REQUESTED_PYTHON" >/dev/null 2>&1 || [ -x "$REQUESTED_PYTHON" ]; then
            SYSTEM_PYTHON="$REQUESTED_PYTHON"
            return
        fi
        echo "ERRO: PYTHON='$REQUESTED_PYTHON' não foi encontrado."
        exit 1
    fi

    if command -v python3.11 >/dev/null 2>&1; then
        SYSTEM_PYTHON=python3.11
    elif command -v python3.12 >/dev/null 2>&1; then
        SYSTEM_PYTHON=python3.12
    elif command -v python3 >/dev/null 2>&1; then
        SYSTEM_PYTHON=python3
    else
        SYSTEM_PYTHON=""
    fi
}

try_apt_components() {
    local py="$1"
    local need_venv="$2"
    local need_tk="$3"

    command -v apt-get >/dev/null 2>&1 || return 1

    local SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
        command -v sudo >/dev/null 2>&1 || return 1
        SUDO="sudo"
    fi

    local ver default_ver=""
    ver="$(py_minor "$py")"
    if command -v python3 >/dev/null 2>&1; then
        default_ver="$(py_minor python3 2>/dev/null || true)"
    fi

    echo "Tentando completar o Python $ver com pacotes do sistema..."
    $SUDO apt-get update

    if [ "$need_venv" = "1" ]; then
        if apt-cache show "python${ver}-venv" >/dev/null 2>&1; then
            $SUDO apt-get install -y "python${ver}-venv" || true
        elif [ "$default_ver" = "$ver" ]; then
            $SUDO apt-get install -y python3-venv || true
        fi
    fi

    if [ "$need_tk" = "1" ]; then
        # Não instale python3-tk para um Python paralelo quando o pacote
        # corresponde apenas ao Python padrão da distribuição.
        if apt-cache show "python${ver}-tk" >/dev/null 2>&1; then
            $SUDO apt-get install -y "python${ver}-tk" || true
        elif [ "$default_ver" = "$ver" ]; then
            $SUDO apt-get install -y python3-tk || true
        else
            echo "Tkinter do sistema para Python $ver não está disponível neste repositório."
        fi
    fi
}

download_file() {
    local url="$1"
    local dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 20 "$url" -o "$dest"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest" "$url"
    else
        echo "ERRO: curl ou wget é necessário para baixar o runtime local."
        echo "Ubuntu/Debian: sudo apt install curl"
        return 1
    fi
}

create_local_py312_env() {
    local arch mf_arch mf_sha
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)
            mf_arch="x86_64"
            mf_sha="281b0ac7d550802efc81af633225a5e6116d29ae72f3ab4eae7168c3931a4c05"
            ;;
        aarch64|arm64)
            mf_arch="aarch64"
            mf_sha="89b786c8d2c8b0fda7553914c1314ae4ddaa094503802f279377b19ac4463cb2"
            ;;
        *)
            echo "ERRO: fallback automático não preparado para arquitetura '$arch'."
            return 1
            ;;
    esac

    local version="26.7.2-0"
    local mf_root="$PWD/.ilt_miniforge"
    local conda="$mf_root/bin/conda"
    local installer="/tmp/Miniforge3-${version}-Linux-${mf_arch}.sh"
    local url="https://github.com/conda-forge/miniforge/releases/download/${version}/Miniforge3-${version}-Linux-${mf_arch}.sh"

    echo
    echo "------------------------------------------------------------"
    echo " Python do sistema sem ambiente virtual completo/compatível"
    echo "------------------------------------------------------------"
    echo "Vou criar um Python 3.12 local com pip + Tkinter em .venv."
    echo "Isso preserva PyTorch 2.5.1 e não substitui o Python do Ubuntu."
    echo

    if [ ! -x "$conda" ]; then
        echo "Baixando Miniforge ${version} (~120 MB)..."
        download_file "$url" "$installer"

        if command -v sha256sum >/dev/null 2>&1; then
            local got
            got="$(sha256sum "$installer" | awk '{print $1}')"
            if [ "$got" != "$mf_sha" ]; then
                echo "ERRO: SHA-256 do Miniforge não confere."
                echo "Esperado: $mf_sha"
                echo "Obtido:   $got"
                rm -f "$installer"
                return 1
            fi
            echo "SHA-256 do Miniforge: OK"
        fi

        rm -rf "$mf_root"
        bash "$installer" -b -p "$mf_root"
        rm -f "$installer"
    fi

    if [ -d ".venv" ]; then
        echo "Removendo .venv incompleto/incompatível..."
        rm -rf .venv
    fi

    echo "Criando ambiente local Python 3.12 + Tkinter..."
    "$conda" create -y -p "$PWD/.venv" -c conda-forge python=3.12 tk pip

    if [ ! -x ".venv/bin/python" ]; then
        echo "ERRO: .venv/bin/python não foi criado."
        return 1
    fi
    if ! has_pip ".venv/bin/python"; then
        echo "ERRO: o runtime local foi criado, mas pip continua indisponível."
        return 1
    fi
    if ! has_tk ".venv/bin/python"; then
        echo "ERRO: o runtime local foi criado, mas Tkinter continua indisponível."
        return 1
    fi

    ENV_KIND="miniforge-py312"
    echo "Runtime local pronto: $(.venv/bin/python --version 2>&1)"
    echo "Tkinter: OK"
}

detect_python

if [ -n "$SYSTEM_PYTHON" ]; then
    VER="$(py_minor "$SYSTEM_PYTHON")"
    echo "Python encontrado: $("$SYSTEM_PYTHON" --version 2>&1) [$SYSTEM_PYTHON]"
    echo

    NEED_VENV=0
    NEED_TK=0
    has_venv "$SYSTEM_PYTHON" || NEED_VENV=1
    has_tk "$SYSTEM_PYTHON" || NEED_TK=1

    # O ponto crítico no Ubuntu 22.04: `import venv` pode funcionar mesmo
    # quando ensurepip não está instalado. Testamos a criação real.
    if [ "$NEED_VENV" = "0" ]; then
        if ! can_create_complete_venv "$SYSTEM_PYTHON"; then
            NEED_VENV=1
        fi
    fi

    if [ "$NEED_VENV" = "1" ] || [ "$NEED_TK" = "1" ]; then
        echo "Componentes ausentes/incompletos no Python $VER:"
        [ "$NEED_VENV" = "1" ] && echo "  - venv + ensurepip/pip funcional"
        [ "$NEED_TK" = "1" ] && echo "  - tkinter"
        echo
        try_apt_components "$SYSTEM_PYTHON" "$NEED_VENV" "$NEED_TK" || true
    fi

    # Revalidação após eventual tentativa via apt.
    SYSTEM_CAN_VENV=0
    if has_venv "$SYSTEM_PYTHON" && can_create_complete_venv "$SYSTEM_PYTHON"; then
        SYSTEM_CAN_VENV=1
    fi

    if [ "$SYSTEM_CAN_VENV" != "1" ] || ! has_tk "$SYSTEM_PYTHON"; then
        echo
        echo "O Python do sistema não consegue fornecer um venv completo para o ILT."
        echo "Usando fallback local e isolado (Miniforge/Python 3.12)."
        create_local_py312_env
    else
        case "$VER" in
            3.11|3.12)
                # Nunca reutilize uma .venv parcial: Python existir não basta.
                if [ -d ".venv" ] && ! existing_env_bootstrap_ok; then
                    echo "Removendo .venv incompleto (pip/Tkinter ausente)..."
                    rm -rf .venv
                fi

                if [ ! -x ".venv/bin/python" ]; then
                    echo "Criando ambiente virtual..."
                    if ! "$SYSTEM_PYTHON" -m venv .venv; then
                        echo "A criação do venv pelo Python do sistema falhou."
                        rm -rf .venv
                        create_local_py312_env
                    fi
                fi

                # Proteção extra para criações parcialmente bem-sucedidas.
                if ! existing_env_bootstrap_ok; then
                    echo "O venv criado pelo sistema ficou incompleto."
                    rm -rf .venv
                    create_local_py312_env
                fi
                ;;
            *)
                echo "Python $VER não pertence à pilha estável desta versão."
                echo "Usando runtime local Python 3.12..."
                create_local_py312_env
                ;;
        esac
    fi
else
    echo "Python 3 não foi encontrado no sistema."
    create_local_py312_env
fi

VENV_PY="$PWD/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "ERRO: ambiente .venv não foi criado."
    exit 1
fi
if ! has_pip "$VENV_PY"; then
    echo "ERRO: pip continua indisponível no ambiente final."
    echo "Consulte $LOG"
    exit 1
fi
if ! has_tk "$VENV_PY"; then
    echo "ERRO: tkinter continua indisponível no ambiente final."
    echo "Consulte $LOG"
    exit 1
fi

echo
echo "Ambiente Python selecionado:"
echo "  $("$VENV_PY" --version 2>&1)"
echo "  Executável: $VENV_PY"
echo "  Tipo: $ENV_KIND"
"$VENV_PY" -c 'import tkinter; print("  Tkinter:", tkinter.TkVersion)'

echo
echo "Atualizando pip/wheel/setuptools..."
"$VENV_PY" -m pip install --upgrade pip wheel setuptools

echo
echo "Detectando a placa de vídeo para escolher o PyTorch adequado..."
"$VENV_PY" src/gpu_profile.py || true
PROFILE_KEY="$("$VENV_PY" src/gpu_profile.py --key 2>/dev/null || echo cpu)"
case "$PROFILE_KEY" in cpu|cu118|cu124|cu128) ;; *) PROFILE_KEY=cpu ;; esac
REQ="requirements-${PROFILE_KEY}.txt"
[ -f "$REQ" ] || { echo "ERRO: $REQ não encontrado."; exit 1; }
echo
echo "Perfil escolhido automaticamente: $PROFILE_KEY  ($REQ)"
echo "(para forçar outro: ILT_PROFILE=cpu|cu118|cu124|cu128 ./setup_linux.sh)"

OLD_KEY=""
if [ -f ".ilt_linux_profile" ]; then
    OLD_KEY="$(sed -n 's/^profile_key=//p' .ilt_linux_profile | head -n 1)"
fi
if [ "$OLD_KEY" != "$PROFILE_KEY" ]; then
    echo "Garantindo o PyTorch correto para este perfil..."
    "$VENV_PY" -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true
fi

install_profile() {
    echo
    echo "Instalando dependências ($1). PyTorch/OpenCV podem demorar alguns minutos."
    "$VENV_PY" -m pip install -r "$1"
}
install_profile "$REQ"

echo
echo "Instalando a interface web (Dash/Plotly; versões fixadas em requirements-web.txt)."
"$VENV_PY" -m pip install -c requirements-web.txt -r requirements-web.txt \
  || echo "AVISO: interface web não instalada; a interface clássica continua disponível."

echo
echo "Opcional: pyswisseph (ciclo lunar da Análise Temporal)."
"$VENV_PY" -m pip install -r requirements-optional.txt \
  || echo "AVISO: pyswisseph não instalado (exige compilador C: sudo apt install build-essential python3-dev). O restante funciona normalmente."

mkdir -p input output assets/mascots
chmod +x START_HERE_LINUX.sh run_gui.sh diagnose_linux.sh setup_linux.sh 2>/dev/null || true

verify_install() {
    ILT_PROFILE_KEY="$PROFILE_KEY" "$VENV_PY" - <<'PY'
import os, sys, tkinter
import torch, cv2, numpy, pandas, sklearn, scipy, umap, pacmap, openTSNE, matplotlib, psutil, tqdm
from PIL import Image
key = os.environ.get("ILT_PROFILE_KEY", "cpu")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA runtime:", torch.version.cuda or "CPU-only")
print("OpenCV:", cv2.__version__, "| numpy:", numpy.__version__, "| pandas:", pandas.__version__)
try:
    import dash, plotly
    print("Interface web: Dash", dash.__version__, "/ Plotly", plotly.__version__)
except Exception:
    print("Interface web: indisponível (será usada a interface clássica)")
try:
    import swisseph  # noqa: F401
    print("Ciclo lunar (pyswisseph): OK")
except Exception:
    print("Ciclo lunar (pyswisseph): indisponível (opcional)")
if key != "cpu":
    if not torch.cuda.is_available():
        print("FALHA: PyTorch não acessa a GPU (driver/CUDA).")
        sys.exit(3)
    try:
        x = torch.randn(256, 256, device="cuda")
        y = (x @ x).sum().item()
        with torch.autocast("cuda", dtype=torch.float16):
            (x @ x).sum().item()
    except Exception as exc:
        print("FALHA: a GPU não executou operações com este PyTorch:", exc)
        sys.exit(3)
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} | compute capability {cap[0]}.{cap[1]} | teste: OK")
    print("Ajustes automáticos: CUDA + precisão mista (AMP);",
          "channels_last ativo" if torch.cuda.get_device_capability(0)[0] >= 7 else "channels_last desligado (sem Tensor Cores)")
else:
    print("Dispositivo: CPU (threads:", os.cpu_count(), ")")
PY
}

echo
echo "Verificando a instalação..."
if ! verify_install; then
    if [ "$PROFILE_KEY" != "cpu" ]; then
        echo
        echo "A GPU não pôde ser usada com o perfil $PROFILE_KEY. Instalando o perfil CPU para garantir o funcionamento."
        echo "(Depois de atualizar o driver NVIDIA, rode ./setup_linux.sh de novo.)"
        "$VENV_PY" -m pip uninstall -y torch torchvision >/dev/null 2>&1 || true
        PROFILE_KEY=cpu
        REQ="requirements-cpu.txt"
        install_profile "$REQ"
        verify_install
    else
        echo "ERRO: a verificação falhou. Rode ./diagnose_linux.sh e envie diagnostics/linux_diagnostic.txt."
        exit 1
    fi
fi

cat > .ilt_linux_profile <<EOF
profile_key=$PROFILE_KEY
profile=$("$VENV_PY" -c "import sys;sys.path.insert(0,'src');import gpu_profile as g;print(g.PROFILES['$PROFILE_KEY'])")
requirements=$REQ
python=$("$VENV_PY" --version 2>&1)
env_kind=$ENV_KIND
configured_at=$(date -Is 2>/dev/null || date)
EOF

echo
echo "============================================================"
echo " CONFIGURAÇÃO CONCLUÍDA"
echo "============================================================"
echo
echo "Para abrir o ILT:  ./run_gui.sh"
echo "(a interface abre no navegador; para encerrar use o botão Encerrar)"
echo
echo "Nas próximas vezes não é necessário rodar o setup novamente."
echo "Log: $LOG"
