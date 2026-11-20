#!/usr/bin/env python3
"""Entry point of the ILT web interface (Dash/Plotly, served on 127.0.0.1)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if __name__ == '__main__':
    # The IAT integrated cycle classifies batches inside this process.
    from hw_resources import raise_open_file_limit
    raise_open_file_limit()
    from web.app import main
    main(open_browser='--no-browser' not in sys.argv)
