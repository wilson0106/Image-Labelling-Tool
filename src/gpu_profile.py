#!/usr/bin/env python3
"""Pick the PyTorch build for this machine (stdlib only; runs before torch).

Used by setup_linux.sh and setup_windows.ps1.  Reads ``nvidia-smi`` and maps
the GPU architecture (compute capability) and driver version to one of:

  cpu    PyTorch 2.5.1 CPU            (no NVIDIA GPU, very old GPU or driver)
  cu118  PyTorch 2.5.1 + CUDA 11.8    (Maxwell, Pascal, Volta, Turing: GTX 9xx/10xx/16xx,
                                       RTX 20xx, Quadro P/T/RTX — e.g. Quadro P4000)
  cu124  PyTorch 2.5.1 + CUDA 12.4    (Ampere, Ada, Hopper: RTX 30xx/40xx, A-series, L4, H100)
  cu128  PyTorch 2.7.1 + CUDA 12.8    (Blackwell: RTX 50xx, B-series; needs PyTorch ≥ 2.7)

The runtime itself adapts to whatever GPU is present (``torch_runtime``):
CUDA + mixed precision when available, channels_last only on Tensor-Core
GPUs (compute capability ≥ 7.0), parallel data loading sized to the CPUs.

Override with the environment variable ILT_PROFILE=cpu|cu118|cu124|cu128.
Output: ``--key`` prints only the profile key; default prints a JSON report.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

PROFILES = {
    'cpu': 'CPU (PyTorch 2.5.1)',
    'cu118': 'NVIDIA CUDA 11.8 (PyTorch 2.5.1)',
    'cu124': 'NVIDIA CUDA 12.4 (PyTorch 2.5.1)',
    'cu128': 'NVIDIA CUDA 12.8 (PyTorch 2.7.1, GPUs Blackwell)',
}
# Minimum driver (major version) for each CUDA build.
MIN_DRIVER = {
    'linux': {'cu118': 450, 'cu124': 525, 'cu128': 570},
    'win': {'cu118': 452, 'cu124': 528, 'cu128': 572},
}

# Fallback when the driver is too old to report compute capability.
NAME_RULES = [
    (r'RTX\s*50\d\d|RTX\s*PRO\s*\d+\s*Blackwell|\bB(100|200)\b|GB\d{3}', 12.0),
    (r'H100|H200|GH200', 9.0),
    (r'RTX\s*40\d\d|\bL4\b|\bL40|RTX\s*\d+\s*Ada', 8.9),
    (r'RTX\s*30\d\d|\bA\d{1,4}\b|RTX\s*A\d{4}', 8.6),
    (r'A100|A30\b', 8.0),
    (r'RTX\s*20\d\d|GTX\s*16\d\d|Quadro\s*RTX|\bT4\b|\bT\d{3,4}\b|TITAN\s*RTX', 7.5),
    (r'V100|TITAN\s*V', 7.0),
    (r'GTX\s*10\d\d|Quadro\s*P\d+|\bP100\b|\bP40\b|\bP4\b|TITAN\s*X[p ]', 6.1),
    (r'GTX\s*9\d\d|Quadro\s*M\d+|\bM\d{2}\b|TITAN\s*X\b', 5.2),
]


def _smi():
    exe = shutil.which('nvidia-smi')
    if not exe and os.name == 'nt':
        for cand in (r'C:\Windows\System32\nvidia-smi.exe',
                     r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'):
            if os.path.isfile(cand):
                exe = cand
                break
    return exe


def _query(exe, fields):
    out = subprocess.run([exe, f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=20)
    if out.returncode != 0:
        return None
    return [[c.strip() for c in line.split(',')] for line in out.stdout.strip().splitlines() if line.strip()]


def detect():
    report = {'gpus': [], 'driver': None, 'profile': 'cpu', 'reason': ''}
    exe = _smi()
    if not exe:
        report['reason'] = 'Nenhuma GPU NVIDIA encontrada (nvidia-smi ausente). Usando CPU.'
        return report
    rows = None
    try:
        rows = _query(exe, 'name,driver_version,compute_cap')
        if rows is None:
            rows = [r + [''] for r in (_query(exe, 'name,driver_version') or [])]
    except Exception:
        rows = None
    if not rows:
        report['reason'] = 'nvidia-smi não respondeu (driver ausente ou com problema). Usando CPU.'
        return report
    for name, driver, cap in rows:
        try:
            cc = float(cap)
        except ValueError:
            cc = next((v for pat, v in NAME_RULES if re.search(pat, name, re.I)), 6.1)
        report['gpus'].append({'name': name, 'compute_capability': cc})
        report['driver'] = driver
    # Use the most capable GPU (training uses cuda:0; the driver orders by default).
    cc = max(g['compute_capability'] for g in report['gpus'])
    try:
        drv = int(str(report['driver']).split('.')[0])
    except ValueError:
        drv = 0
    mins = MIN_DRIVER['win' if os.name == 'nt' else 'linux']
    name = report['gpus'][0]['name']
    if cc < 5.0:
        report['reason'] = f'{name}: arquitetura antiga (CC {cc}) não suportada pelo PyTorch atual. Usando CPU.'
    elif cc >= 10.0:
        if drv >= mins['cu128']:
            report['profile'] = 'cu128'
            report['reason'] = f'{name} (Blackwell, CC {cc}) requer CUDA 12.8 / PyTorch 2.7.'
        else:
            report['reason'] = (f'{name} requer driver NVIDIA ≥ {mins["cu128"]} (instalado: {report["driver"]}). '
                                'Atualize o driver para usar a GPU; por enquanto, CPU.')
    elif cc >= 8.0:
        if drv >= mins['cu124']:
            report['profile'] = 'cu124'
        elif drv >= mins['cu118']:
            report['profile'] = 'cu118'
        report['reason'] = f'{name} (CC {cc}), driver {report["driver"]}.'
    else:
        if drv >= mins['cu118']:
            report['profile'] = 'cu118'
            report['reason'] = f'{name} (CC {cc}), driver {report["driver"]}.'
        else:
            report['reason'] = f'{name}: driver {report["driver"]} antigo demais (mínimo {mins["cu118"]}). Usando CPU.'
    return report


def main(argv):
    forced = os.environ.get('ILT_PROFILE', '').strip().lower()
    report = detect()
    if forced in PROFILES:
        report['profile'] = forced
        report['reason'] = f'Perfil definido manualmente (ILT_PROFILE={forced}).'
    report['description'] = PROFILES[report['profile']]
    if '--key' in argv:
        print(report['profile'])
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
