from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

from ilt_contracts import atomic_write_json

DEFAULT_PIXEL_SIZE_UM_PER_PX = 12.87
PROJECT_SETTINGS_FILENAME = "project_settings.json"


def parse_pixel_size(value: Any, default: float | None = None) -> float:
    """Parse a positive pixel calibration in micrometers per pixel.

    Both decimal point and decimal comma are accepted so the GUI behaves
    naturally in Portuguese locales.  A ValueError is raised for empty,
    non-finite, zero or negative values unless *default* is supplied.
    """
    try:
        text = str(value).strip().replace(",", ".")
        out = float(text)
        if not math.isfinite(out) or out <= 0:
            raise ValueError
        return out
    except Exception as exc:
        if default is not None:
            return float(default)
        raise ValueError("Pixel size deve ser um número positivo em µm/pixel.") from exc


def format_pixel_size(value: Any) -> str:
    val = parse_pixel_size(value, DEFAULT_PIXEL_SIZE_UM_PER_PX)
    # Keep enough precision for microscopy calibration without noisy trailing zeros.
    return f"{val:.8f}".rstrip("0").rstrip(".")


def project_settings_path(project_dir: Path | str) -> Path:
    return Path(project_dir).expanduser().resolve() / PROJECT_SETTINGS_FILENAME


def load_project_settings(project_dir: Path | str) -> Dict[str, Any]:
    path = project_settings_path(project_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def project_pixel_size(project_dir: Path | str, default: float = DEFAULT_PIXEL_SIZE_UM_PER_PX) -> float:
    settings = load_project_settings(project_dir)
    return parse_pixel_size(settings.get("pixel_size_um_per_px", default), default)


def save_project_settings(
    project_dir: Path | str,
    *,
    pixel_size_um_per_px: Any,
    source: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Update project-level calibration without discarding unrelated settings."""
    path = project_settings_path(project_dir)
    payload = load_project_settings(project_dir)
    payload["schema_version"] = int(payload.get("schema_version", 1) or 1)
    payload["pixel_size_um_per_px"] = parse_pixel_size(pixel_size_um_per_px)
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    if source:
        payload["pixel_size_source"] = str(source)
    if extra:
        payload.update(dict(extra))
    atomic_write_json(payload, path, ensure_ascii=False, indent=2)
    return path
