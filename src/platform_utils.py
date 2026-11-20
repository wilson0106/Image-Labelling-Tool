import os
import subprocess
import sys
from pathlib import Path


def is_frozen():
    return bool(getattr(sys,'frozen',False))


def application_dir():
    """Directory the user launched/copied the application from."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_dir():
    """Root containing bundled read-only resources."""
    base=getattr(sys,'_MEIPASS',None)
    return Path(base).resolve() if base else application_dir()


def open_path(path):
    """Open a file/folder in the native file manager on Windows/macOS/Linux."""
    p=str(Path(path).expanduser().resolve())
    if os.name=='nt':
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform=='darwin':
        subprocess.Popen(['open',p],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(['xdg-open',p],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def worker_command(role,*args,src_dir=None):
    """Command used by the GUI for background scientific workers."""
    if is_frozen():
        return [sys.executable,'--ilt-worker',str(role),*[str(x) for x in args]]
    src=Path(src_dir or Path(__file__).resolve().parent)
    scripts={
        'training':'train.py',
        'classification':'main.py',
        'clustering':'class_clustering.py',
    }
    script=scripts.get(str(role))
    if not script:
        raise ValueError(f'Unknown worker role: {role}')
    return [sys.executable,'-u',str(src/script),*[str(x) for x in args]]
