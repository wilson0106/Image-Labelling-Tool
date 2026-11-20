#!/usr/bin/env python3
"""Background GPU/CPU usage sampler for the sidebar gauges.

One ``nvidia-smi`` query every 2 s on a daemon thread (≈ negligible cost);
falls back to CPU/RAM via psutil when no NVIDIA GPU is available.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time


class SystemMonitor:
    def __init__(self, period=2.0):
        self.period = period
        self.enabled = True
        self.data = {'a_pct': 0, 'a_label': '—', 'a_sub': 'GPU', 'b_pct': 0, 'b_label': '—', 'b_sub': 'VRAM'}
        self.version = 0
        self.device = ''
        self._smi = shutil.which('nvidia-smi')
        threading.Thread(target=self._loop, name='ilt-web-sysmon', daemon=True).start()

    def _sample_gpu(self):
        out = subprocess.run(
            [self._smi, '--query-gpu=name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError('nvidia-smi failed')
        name, util, used, total = [x.strip() for x in out.stdout.strip().splitlines()[0].split(',')]
        used_g, total_g = float(used) / 1024, float(total) / 1024
        self.device = name
        return {'a_pct': float(util), 'a_label': f'{float(util):.0f}%', 'a_sub': 'GPU',
                'b_pct': 100 * used_g / max(total_g, 1e-6), 'b_label': f'{used_g:.1f}G',
                'b_sub': f'VRAM de {total_g:.0f}G'}

    @staticmethod
    def _sample_cpu():
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        return {'a_pct': cpu, 'a_label': f'{cpu:.0f}%', 'a_sub': 'CPU',
                'b_pct': mem.percent, 'b_label': f'{mem.used / 2**30:.1f}G',
                'b_sub': f'RAM de {mem.total / 2**30:.0f}G'}

    def _loop(self):
        while True:
            if self.enabled:
                try:
                    data = self._sample_gpu() if self._smi else self._sample_cpu()
                except Exception:
                    try:
                        data = self._sample_cpu()
                    except Exception:
                        data = None
                if data and data != self.data:
                    self.data = data
                    self.version += 1
            time.sleep(self.period)


_MON = None


def get_monitor():
    global _MON
    if _MON is None:
        _MON = SystemMonitor()
    return _MON
