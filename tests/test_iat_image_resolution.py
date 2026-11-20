from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iat_backend import IATProject
from iat_export_mixin import IATExportMixin


class _DummyExport(IATExportMixin):
    pass


class IATImageResolutionTests(unittest.TestCase):
    def _project(self, base: Path):
        project = base / "project"
        (project / "dataframes").mkdir(parents=True)
        (project / "images").mkdir()
        return project

    def test_direct_batch_resolution_is_cached(self):
        with tempfile.TemporaryDirectory() as td:
            project_dir = self._project(Path(td))
            sample_dir = project_dir / "images" / "batch_0001" / "samples"
            sample_dir.mkdir(parents=True)
            image = sample_dir / "a.png"
            image.write_bytes(b"fake")
            project = IATProject(project_dir)
            project.reset_image_resolve_stats()
            first = project.resolve_image("batch_0001.csv", "a.png")
            second = project.resolve_image("batch_0001.csv", "a.png")
            self.assertEqual(first, image)
            self.assertEqual(second, image)
            stats = project.image_resolve_stats()
            self.assertEqual(stats["direct_hit"], 1)
            self.assertEqual(stats["cache_hit"], 1)
            self.assertEqual(stats["global_index_builds"], 0)

    def test_unusual_layout_builds_global_index_only_once(self):
        with tempfile.TemporaryDirectory() as td:
            project_dir = self._project(Path(td))
            odd = project_dir / "images" / "legacy" / "nested"
            odd.mkdir(parents=True)
            image = odd / "special.png"
            image.write_bytes(b"fake")
            project = IATProject(project_dir)
            project.reset_image_resolve_stats()
            self.assertEqual(project.resolve_image("batch_9999.csv", "special.png"), image)
            self.assertEqual(project.resolve_image("batch_9999.csv", "special.png"), image)
            stats = project.image_resolve_stats()
            self.assertEqual(stats["global_index_builds"], 1)
            self.assertEqual(stats["global_index_hit"], 1)
            self.assertEqual(stats["cache_hit"], 1)

    def test_lightweight_batch_read_only_returns_requested_columns(self):
        with tempfile.TemporaryDirectory() as td:
            project_dir = self._project(Path(td))
            df = pd.DataFrame({
                "names": ["a.png", "b.png"],
                "manual_label": ["A", "B"],
                "custom_data": ["id1", "id2"],
                "huge_unused_column": ["x" * 100, "y" * 100],
            })
            df.to_csv(project_dir / "dataframes" / "batch_0001.csv", index=False)
            dummy = _DummyExport()
            dummy.project = IATProject(project_dir)
            out = dummy._export_read_batch("batch_0001.csv", columns=("gid", "names", "manual_label"))
            self.assertEqual(list(out.columns), ["gid", "names", "manual_label"])
            self.assertNotIn("huge_unused_column", out.columns)
            self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
