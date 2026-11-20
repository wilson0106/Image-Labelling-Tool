from __future__ import annotations

import ast
import queue
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from iat_gallery_mixin import IATGalleryMixin
from iat_workspace import IATWorkspace


class _Project:
    def __init__(self):
        self.loaded = []
        self.overview_calls = 0
        self.frames = {
            "batch_0001.csv": pd.DataFrame({
                "gid": ["batch_0001.csv::a", "batch_0001.csv::b"],
                "names": ["a.png", "b.png"],
                "manual_label": ["A", "A"],
            }),
            "batch_0002.csv": pd.DataFrame({
                "gid": ["batch_0002.csv::c"],
                "names": ["c.png"],
                "manual_label": ["B"],
            }),
        }

    def load_batch(self, name):
        self.loaded.append(name)
        return self.frames[name]

    def load_overview(self):
        self.overview_calls += 1
        return pd.concat(list(self.frames.values()), ignore_index=True)


class _ScopeHarness:
    _cluster_selection_rows = IATWorkspace._cluster_selection_rows
    project = None

    def _active_cluster_gids(self):
        return set()


class _AfterHarness:
    _poll_events = IATWorkspace._poll_events

    def __init__(self):
        self._event_q = queue.Queue()
        self._event_q.put(("thumb", "x", None))
        self._op_running = False
        self._poll_last_perf = None
        self._last_poll_lag_log = 0.0
        self.after_calls = []
        self.logs = []
        self.log_lines = []

    def _apply_thumbnail_result(self, *args):
        raise RuntimeError("recoverable")

    def _finish_operation(self, ok, result, done, error, stopped=False):
        self._op_running = False

    def _log(self, msg, tag="normal"):
        self.logs.append((str(msg), tag))

    def winfo_exists(self):
        return True

    def after(self, delay, callback):
        self.after_calls.append((delay, callback))
        return "after-id"


class _GalleryLeaveHarness(IATGalleryMixin):
    def __init__(self):
        self._gallery_cards_token = 4
        self._thumb_generation = 7
        self._thumb_pending = [(1,)]
        self._thumb_pending_keys = {"x"}
        self._thumb_waiters = {"x": [object()]}
        self._thumb_cache = {"keep": object()}
        self._gallery_prefetch_after = None
        self._gallery_prefetch_state = object()
        self._gallery_card_pool = [{"card": object()}]
        self._gallery_visible_widgets = {"g": object()}
        self.gallery_scroll = None


class GalleryClusteringResponsivenessTests(unittest.TestCase):
    def test_scope_reads_only_referenced_batches_without_overview(self):
        h = _ScopeHarness()
        project = _Project()
        timing = {}
        gids = ["batch_0001.csv::a", "batch_0002.csv::c"]
        out = h._cluster_selection_rows(gids=gids, project=project, timing=timing)
        self.assertEqual(out["gid"].tolist(), gids)
        self.assertEqual(project.loaded, ["batch_0001.csv", "batch_0002.csv"])
        self.assertEqual(project.overview_calls, 0)
        self.assertFalse(timing["overview_used"])

    def test_event_monitor_rearms_after_recoverable_handler_error(self):
        h = _AfterHarness()
        h._poll_events()
        self.assertTrue(h.after_calls, "poller must always schedule its next callback")
        self.assertTrue(any("Erro recuperável" in msg for msg, _ in h.logs))

    def test_terminal_event_restores_operation_state(self):
        h = _AfterHarness()
        h._event_q = queue.Queue()
        h._event_q.put(("done", 123))
        h._op_running = True
        h._op_done_cb = None
        h._poll_events()
        self.assertFalse(h._op_running)
        self.assertTrue(h.after_calls)

    def test_gallery_leave_invalidates_jobs_but_preserves_cache(self):
        h = _GalleryLeaveHarness()
        cache_before = dict(h._thumb_cache)
        cards = h._gallery_prepare_leave()
        self.assertEqual(h._thumb_pending, [])
        self.assertEqual(h._thumb_pending_keys, set())
        self.assertEqual(h._thumb_waiters, {})
        self.assertEqual(h._thumb_cache, cache_before)
        self.assertEqual(len(cards), 1)

    def test_missing_artifact_does_not_trigger_generation(self):
        class P:
            def __init__(self, path):
                self.clusters_dir = Path(path)
        class H:
            _cluster_artifact_for_rows = IATWorkspace._cluster_artifact_for_rows
            _cluster_artifact_matches_rows = IATWorkspace._cluster_artifact_matches_rows
            current_cluster_path = None
            project = None
        with tempfile.TemporaryDirectory() as td:
            rows = pd.DataFrame({"gid":["batch_0001.csv::a","batch_0001.csv::b"],"names":["a.png","b.png"],"manual_label":["A","A"]})
            path, mode = H()._cluster_artifact_for_rows(rows, project=P(td), current_path=None)
            self.assertIsNone(path)
            self.assertIsNone(mode)

    def test_clustering_builder_does_not_prepare_scope_synchronously(self):
        src = Path(__file__).resolve().parents[1] / "src" / "iat_workspace.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IATWorkspace")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_clustering")
        calls = [
            n.func.attr for n in ast.walk(method)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        self.assertNotIn("_cluster_selection_rows", calls)

    def test_relabel_code_does_not_remove_projection_selection(self):
        src = Path(__file__).resolve().parents[1] / "src" / "iat_gallery_mixin.py"
        tree = ast.parse(src.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IATGalleryMixin")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply_label")
        text = ast.unparse(method)
        self.assertNotIn("selected_gids.difference_update", text)
        self.assertIn("checked_gids.difference_update", text)


if __name__ == "__main__":
    unittest.main()
