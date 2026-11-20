# ILT Data Extraction & IAT 1.1.0 - instalador para Windows 11
# Cria um ambiente Python isolado (.venv) dentro desta pasta, detecta a placa de
# video e instala o PyTorch adequado (NVIDIA CUDA 11.8 / 12.4 / 12.8 ou CPU).
# Nada e instalado no sistema, exceto (se nao houver Python 3.11/3.12) um
# Python 3.12 privado na subpasta .python312 desta pasta.
# Para forcar um perfil:  $env:ILT_PROFILE="cpu"  (ou cu118, cu124, cu128)

# 'Continue': native programs (pip) write warnings to stderr; failures are
# detected through $LASTEXITCODE instead of aborting on the first warning.
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
Set-Location -LiteralPath $PSScriptRoot
$log = Join-Path $PSScriptRoot 'setup_windows.log'
try { Start-Transcript -Path $log -Append | Out-Null } catch {}

function Say($msg) { Write-Host $msg }
function Fail($msg) {
    Write-Host ''
    Write-Host "ERRO: $msg" -ForegroundColor Red
    Write-Host "Log: $log"
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
function Run($exe, [string[]]$argList) {
    # Out-Host: the program's output goes to the screen, not into the
    # function's return value (otherwise failures would look like success).
    & $exe @argList | Out-Host
    return ($LASTEXITCODE -eq 0)
}

Say '============================================================'
Say ' ILT Data Extraction & IAT 1.1.0 - Windows 11'
Say " Pasta: $PSScriptRoot"
Say '============================================================'
Say ''

if ($PSScriptRoot.Length -gt 80) {
    Say 'AVISO: o caminho desta pasta e longo. Se a instalacao do PyTorch falhar,'
    Say '       mova a pasta para algo curto, por exemplo C:\ILT, e rode de novo.'
    Say ''
}

# ---------------- 1. Python 3.11 / 3.12 com Tkinter ----------------
$probe = 'import sys, tkinter, venv; assert sys.version_info[:2] in ((3,11),(3,12)); print(sys.executable)'
function Find-Python {
    $local = Join-Path $PSScriptRoot '.python312\python.exe'
    $cands = @()
    if (Test-Path $local) { $cands += ,@($local) }
    if (Get-Command py -ErrorAction SilentlyContinue) { $cands += ,@('py', '-3.12'); $cands += ,@('py', '-3.11') }
    if (Get-Command python -ErrorAction SilentlyContinue) { $cands += ,@('python') }
    foreach ($c in $cands) {
        try {
            $exe = $c[0]; $pre = @()
            if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
            $out = & $exe @pre -c $probe 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return ($out | Select-Object -Last 1).Trim() }
        } catch {}
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Say 'Python 3.11/3.12 com Tkinter nao encontrado.'
    Say 'Baixando o Python 3.12.8 oficial (python.org) para uso exclusivo do ILT...'
    $url = 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe'
    $sha = '71bd44e6b0e91c17558963557e4cdb80b483de9b0a0a9717f06cf896f95ab598'
    $inst = Join-Path $env:TEMP 'python-3.12.8-amd64.exe'
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $inst -UseBasicParsing -ErrorAction Stop
    } catch { Fail "nao foi possivel baixar o Python ($url). Verifique a internet." }
    $got = (Get-FileHash -Algorithm SHA256 -LiteralPath $inst).Hash.ToLower()
    if ($got -ne $sha) { Fail "o instalador do Python baixado nao confere (SHA-256 $got)." }
    Say 'SHA-256 do instalador: OK'
    $target = Join-Path $PSScriptRoot '.python312'
    # TargetDir quoted: Start-Process joins the list with spaces.
    $instArgs = @('/quiet', 'InstallAllUsers=0', "TargetDir=`"$target`"", 'Include_tcltk=1', 'Include_pip=1',
              'Include_launcher=0', 'PrependPath=0', 'Shortcuts=0', 'AssociateFiles=0',
              'Include_test=0', 'Include_doc=0', 'CompileAll=0')
    $p = Start-Process -FilePath $inst -ArgumentList $instArgs -Wait -PassThru
    if ($p.ExitCode -ne 0) { Fail "a instalacao do Python terminou com codigo $($p.ExitCode)." }
    $py = Find-Python
    if (-not $py) { Fail 'o Python foi instalado, mas nao ficou utilizavel.' }
}
Say "Python: $py"

# ---------------- 2. Ambiente virtual ----------------
$vpy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path $vpy) {
    & $vpy -c 'import tkinter, pip' 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say '.venv incompleta encontrada; recriando...'
        Remove-Item -Recurse -Force (Join-Path $PSScriptRoot '.venv')
    }
}
if (-not (Test-Path $vpy)) {
    Say 'Criando ambiente virtual (.venv)...'
    if (-not (Run $py @('-m', 'venv', '.venv'))) { Fail 'nao foi possivel criar o ambiente virtual.' }
}
if (-not (Run $vpy @('-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'))) { Fail 'falha ao atualizar o pip.' }

# ---------------- 3. Placa de video -> perfil do PyTorch ----------------
Say ''
Say 'Detectando a placa de video...'
& $vpy 'src\gpu_profile.py'
$key = (& $vpy 'src\gpu_profile.py' '--key' | Select-Object -Last 1).Trim()
if (@('cpu', 'cu118', 'cu124', 'cu128') -notcontains $key) { $key = 'cpu' }
Say ''
Say "Perfil escolhido automaticamente: $key  (requirements-$key.txt)"

$profFile = Join-Path $PSScriptRoot '.ilt_windows_profile'
$oldKey = ''
if (Test-Path $profFile) {
    $line = Get-Content $profFile | Where-Object { $_ -like 'profile_key=*' } | Select-Object -First 1
    if ($line) { $oldKey = $line.Substring(12) }
}
if ($oldKey -ne $key) { & $vpy -m pip uninstall -y torch torchvision *> $null }

function Install-Profile($k) {
    Say ''
    Say "Instalando dependencias (requirements-$k.txt). PyTorch pode levar varios minutos..."
    return (Run $vpy @('-m', 'pip', 'install', '-r', "requirements-$k.txt"))
}
if (-not (Install-Profile $key)) {
    if ($key -ne 'cpu') {
        Say "Falha com o perfil $key; tentando o perfil CPU..."
        & $vpy -m pip uninstall -y torch torchvision *> $null
        $key = 'cpu'
        if (-not (Install-Profile $key)) { Fail 'falha ao instalar as dependencias.' }
    } else { Fail 'falha ao instalar as dependencias.' }
}

Say ''
Say 'Instalando a interface web (Dash/Plotly)...'
if (-not (Run $vpy @('-m', 'pip', 'install', '-c', 'requirements-web.txt', '-r', 'requirements-web.txt'))) {
    Say 'AVISO: interface web nao instalada; sera usada a interface classica.'
}
Say ''
Say 'Opcional: pyswisseph (ciclo lunar da Analise Temporal)...'
if (-not (Run $vpy @('-m', 'pip', 'install', '--only-binary=:all:', '-r', 'requirements-optional.txt'))) {
    Say 'AVISO: pyswisseph nao tem pacote pronto para Windows; o modo "Lunar (astronomico)" fica'
    Say '       indisponivel. Todo o resto funciona normalmente.'
}
New-Item -ItemType Directory -Force -Path input, output | Out-Null

# ---------------- 4. Verificacao (inclui teste real na GPU) ----------------
$verify = @'
import os, sys, tkinter
import torch, cv2, numpy, pandas, sklearn, scipy, umap, pacmap, openTSNE, matplotlib, psutil, tqdm
key = os.environ.get("ILT_PROFILE_KEY", "cpu")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA runtime:", torch.version.cuda or "CPU-only")
try:
    import dash, plotly
    print("Interface web: Dash", dash.__version__, "/ Plotly", plotly.__version__)
except Exception:
    print("Interface web: indisponivel (interface classica sera usada)")
if key != "cpu":
    if not torch.cuda.is_available():
        print("FALHA: PyTorch nao acessa a GPU (driver/CUDA)."); sys.exit(3)
    try:
        x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
        with torch.autocast("cuda", dtype=torch.float16):
            (x @ x).sum().item()
    except Exception as exc:
        print("FALHA: a GPU nao executou operacoes com este PyTorch:", exc); sys.exit(3)
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f"GPU {i}: {torch.cuda.get_device_name(i)} | compute capability {cap[0]}.{cap[1]} | teste: OK")
else:
    print("Dispositivo: CPU (threads:", os.cpu_count(), ")")
'@
$verifyFile = Join-Path $env:TEMP 'ilt_verify.py'
Set-Content -LiteralPath $verifyFile -Value $verify -Encoding ASCII
Say ''
Say 'Verificando a instalacao...'
$env:ILT_PROFILE_KEY = $key
& $vpy $verifyFile
if ($LASTEXITCODE -ne 0) {
    if ($key -ne 'cpu') {
        Say "A GPU nao pode ser usada com o perfil $key. Instalando o perfil CPU para garantir o funcionamento."
        Say '(Depois de atualizar o driver NVIDIA, rode INSTALAR.bat de novo.)'
        & $vpy -m pip uninstall -y torch torchvision *> $null
        $key = 'cpu'
        if (-not (Install-Profile $key)) { Fail 'falha ao instalar o perfil CPU.' }
        $env:ILT_PROFILE_KEY = $key
        & $vpy $verifyFile
        if ($LASTEXITCODE -ne 0) { Fail 'a verificacao falhou. Rode DIAGNOSTICO.bat.' }
    } else { Fail 'a verificacao falhou. Rode DIAGNOSTICO.bat.' }
}

@(
    "profile_key=$key",
    "python=$(& $vpy --version)",
    "configured_at=$(Get-Date -Format s)"
) | Set-Content -LiteralPath $profFile -Encoding ASCII

# Atalho na Area de Trabalho
try {
    $desk = [Environment]::GetFolderPath('Desktop')
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path $desk 'ILT Data Extraction.lnk'))
    $lnk.TargetPath = Join-Path $PSScriptRoot 'ILT.bat'
    $lnk.WorkingDirectory = $PSScriptRoot
    $ico = Join-Path $PSScriptRoot 'assets\branding\app_icon.ico'
    if (Test-Path $ico) { $lnk.IconLocation = $ico }
    $lnk.Save()
    Say 'Atalho "ILT Data Extraction" criado na Area de Trabalho.'
} catch { Say 'AVISO: nao foi possivel criar o atalho na Area de Trabalho.' }

Say ''
Say '============================================================'
Say ' INSTALACAO CONCLUIDA'
Say '============================================================'
Say 'Para abrir: clique duas vezes em ILT.bat (ou no atalho da Area de Trabalho).'
Say 'A interface abre no navegador; para encerrar use o botao Encerrar.'
try { Stop-Transcript | Out-Null } catch {}
exit 0
