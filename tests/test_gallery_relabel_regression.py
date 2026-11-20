import unittest
import pandas as pd

from iat_gallery_mixin import IATGalleryMixin


class _Var:
    def __init__(self, value):
        self.value = value
    def get(self):
        return self.value


class _Project:
    def __init__(self):
        self.calls = []
        self.saved = False
    def apply_labels(self, gids, label):
        self.calls.append((list(gids), label))
        return len(gids)
    def save_dirty(self, progress=None):
        self.saved = True
        if progress:
            progress(1, 1, "batch_0001.csv")
        return []


class _Harness(IATGalleryMixin):
    def __init__(self):
        gid = "batch_0001.csv::img_001.png"
        self.project = _Project()
        self.checked_gids = {gid}
        self.selected_gids = {gid}
        self.visible_classes = {"old"}
        self.current_df = pd.DataFrame({
            "gid": [gid],
            "manual_label": ["old"],
            "correct_label": ["old"],
        })
        self._analysis_df = pd.DataFrame()
        self._validation_candidates = pd.DataFrame()
        self._gallery_selection_df = self.current_df.copy()
        self._gallery_discarded_gids = set()
        self.allb_gids = [gid]
        self._gallery_anchor_gid = gid
        self.page_idx = 0
        self.logs = []
        self.patched = False
        self.vars = {
            "new_label": _Var("new"),
            "save_auto": _Var(True),
            "discard_after": _Var(False),
        }

    def tr(self, key):
        return key

    def _log(self, msg, tag="normal"):
        self.logs.append((str(msg), tag))

    def _scatter_class_column(self, df):
        return "manual_label"

    def _apply_filters(self):
        pass

    def _gallery_source(self):
        return self._gallery_selection_df.copy()

    def _patch_gallery_page(self, src, updated_labels=None):
        self.patched = True

    def _run_operation(self, label, worker, done=None, *args, **kwargs):
        def progress(*_args):
            pass
        result = worker(progress)
        if done:
            done(result)


class GalleryRelabelRegressionTests(unittest.TestCase):
    def test_apply_label_executes_backend_and_updates_live_state(self):
        h = _Harness()
        h.apply_label()
        gid = "batch_0001.csv::img_001.png"
        self.assertEqual(h.project.calls, [([gid], "new")])
        self.assertTrue(h.project.saved)
        self.assertEqual(h.current_df.loc[0, "manual_label"], "new")
        self.assertEqual(h._gallery_selection_df.loc[0, "manual_label"], "new")
        self.assertNotIn(gid, h.checked_gids)
        self.assertIn(gid, h.selected_gids)  # projection selection is preserved
        self.assertTrue(h.patched)


if __name__ == "__main__":
    unittest.main()
