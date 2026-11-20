#!/usr/bin/env python3
"""Build the ILT release zips: Windows 11, Ubuntu 24.04 and Ubuntu 26.04.

Only the runnable application goes in (no .venv, outputs, logs, tests,
caches or user settings).  Usage:  python packaging/make_release.py [outdir]
"""
from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = '1.1.0'
TOP = f'ILT_{VERSION}'

COMMON_DIRS = ['src', 'assets', 'docs', 'reference_data']
COMMON_FILES = ['README.md', 'CHANGELOG.md', 'VERSION.txt', 'input/train_params.json',
                'requirements-common.txt', 'requirements-cpu.txt', 'requirements-cu118.txt',
                'requirements-cu124.txt', 'requirements-cu128.txt', 'requirements-optional.txt',
                'requirements-web.txt']
LINUX_FILES = ['START_HERE_LINUX.sh', 'setup_linux.sh', 'run_gui.sh', 'diagnose_linux.sh']
WINDOWS_FILES = ['INSTALAR.bat', 'ILT.bat', 'ILT_classico.bat', 'DIAGNOSTICO.bat', 'setup_windows.ps1']
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.mypy_cache'}
SKIP_SUFFIX = {'.pyc', '.pyo', '.log'}
EXECUTABLE = {'.sh'}


def iter_files():
    for d in COMMON_DIRS:
        for p in sorted((ROOT / d).rglob('*')):
            if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_PARTS) and p.suffix not in SKIP_SUFFIX:
                yield p
    for f in COMMON_FILES:
        yield ROOT / f


def add(zf, src: Path, arc: str, crlf=False):
    info = zipfile.ZipInfo(f'{TOP}/{arc}', date_time=time.localtime()[:6])
    mode = 0o755 if src.suffix in EXECUTABLE else 0o644
    info.external_attr = (0o100000 | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    data = src.read_bytes()
    if crlf:
        data = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    zf.writestr(info, data)


def build(out_dir: Path, name: str, extra_files, readme: Path, windows: bool):
    path = out_dir / name
    with zipfile.ZipFile(path, 'w', compresslevel=9) as zf:
        for p in iter_files():
            add(zf, p, p.relative_to(ROOT).as_posix())
        for f in extra_files:
            src = ROOT / f
            add(zf, src, f, crlf=windows and src.suffix in {'.ps1', '.bat'})
        add(zf, readme, 'LEIA-ME_INSTALACAO.txt', crlf=windows)
        for d in ('output', 'input'):
            info = zipfile.ZipInfo(f'{TOP}/{d}/')
            info.external_attr = (0o40755 << 16) | 0x10
            if d == 'output':
                zf.writestr(info, b'')
    return path


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / 'ILT_releases'
    out_dir.mkdir(parents=True, exist_ok=True)
    pk = ROOT / 'packaging'
    made = [
        build(out_dir, f'ILT_{VERSION}_Windows11.zip', WINDOWS_FILES, pk / 'LEIA-ME_WINDOWS.txt', True),
        build(out_dir, f'ILT_{VERSION}_Ubuntu24.04.zip', LINUX_FILES, pk / 'LEIA-ME_UBUNTU_24.04.txt', False),
        build(out_dir, f'ILT_{VERSION}_Ubuntu26.04.zip', LINUX_FILES, pk / 'LEIA-ME_UBUNTU_26.04.txt', False),
    ]
    for m in made:
        print(f'{m}  {m.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
