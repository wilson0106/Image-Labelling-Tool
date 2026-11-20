from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project_settings import (
    DEFAULT_PIXEL_SIZE_UM_PER_PX,
    format_pixel_size,
    load_project_settings,
    parse_pixel_size,
    project_pixel_size,
    save_project_settings,
)


class PixelSizeSettingsTests(unittest.TestCase):
    def test_parse_accepts_dot_and_decimal_comma(self):
        self.assertAlmostEqual(parse_pixel_size("12.87"), 12.87)
        self.assertAlmostEqual(parse_pixel_size("13,87"), 13.87)
        self.assertEqual(format_pixel_size("13,8700"), "13.87")

    def test_invalid_pixel_size_is_rejected(self):
        for value in ("", "abc", "0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_pixel_size(value)

    def test_project_setting_roundtrip_and_preserves_other_keys(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "project_settings.json").write_text(
                json.dumps({"keep_me": 7, "pixel_size_um_per_px": 12.87}),
                encoding="utf-8",
            )
            save_project_settings(root, pixel_size_um_per_px="13,87", source="test")
            payload = load_project_settings(root)
            self.assertEqual(payload["keep_me"], 7)
            self.assertAlmostEqual(payload["pixel_size_um_per_px"], 13.87)
            self.assertEqual(payload["pixel_size_source"], "test")
            self.assertAlmostEqual(project_pixel_size(root), 13.87)

    def test_missing_project_setting_uses_stable_default(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertAlmostEqual(
                project_pixel_size(Path(td)), DEFAULT_PIXEL_SIZE_UM_PER_PX
            )



class _DummyVar:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value


class IATPixelCalibrationTests(unittest.TestCase):
    def test_project_calibration_overrides_stored_segmentation_for_physical_units(self):
        from types import SimpleNamespace
        from iat_workspace import IATWorkspace
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save_project_settings(root, pixel_size_um_per_px=13.87, source="test")
            ws = object.__new__(IATWorkspace)
            ws.project = SimpleNamespace(project_dir=root)
            ws.vars = {"pixel_size": _DummyVar("13,87")}
            self.assertAlmostEqual(
                ws._segmentation_pixel_size({"Pixel Size (um/pixel)": 12.87}),
                13.87,
            )

    def test_legacy_project_keeps_segmentation_calibration(self):
        from types import SimpleNamespace
        from iat_workspace import IATWorkspace
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = object.__new__(IATWorkspace)
            ws.project = SimpleNamespace(project_dir=root)
            ws.vars = {"pixel_size": _DummyVar("12.87")}
            self.assertAlmostEqual(
                ws._segmentation_pixel_size({"Pixel Size (um/pixel)": 13.87}),
                13.87,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
