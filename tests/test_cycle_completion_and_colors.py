"""Regressions for IAT cycle callbacks and stable class colors (no Tk display needed)."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from iat_backend import get_class_color, get_color
from iat_model_cycle_mixin import IATModelCycleMixin


class _Value:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _Widget:
    def __init__(self, alive=True):
        self.alive = alive
        self.state = None

    def winfo_exists(self):
        return int(self.alive)

    def configure(self, **kwargs):
        if not self.alive:
            raise RuntimeError("invalid command name")
        self.state = kwargs.get("state")


class _Cycle(IATModelCycleMixin):
    def __init__(self, root, tab):
        self.project = type("Project", (), {"project_dir": Path(root)})()
        self.current_tab = tab
        self._cycle_train_btn = _Widget(False)
        self._cycle_apply_btn = _Widget(False)
        self._cycle_eval_btn = _Widget(False)
        self._cycle_model_combo = _Widget(False)
        self._cycle_target_manual_host = _Widget(False)
        self._cycle_temp_dataset = None
        self._cycle_stage_progress = {
            "training": _Value(0),
            "cluster": _Value(0),
        }
        self._cycle_status = _Value("")
        self._cycle_trained_version = None
        self._cycle_model_path = None
        self._cycle_labels_path = None
        self.refreshed = 0
        self.target_updated = 0
        self.finalized = []
        self.reloaded = 0

    def lang(self):
        return "pt"

    def _cycle_refresh_saved_models(self, prefer=None):
        self.refreshed += 1

    def _cycle_target_mode_changed(self):
        self.target_updated += 1

    def _cycle_finalize_version(self, version, result, clustered):
        self.finalized.append((version, clustered))

    def _cycle_reload_project_after_apply(self):
        self.reloaded += 1


class CycleCompletionAndColorsTests(unittest.TestCase):
    def _model_dir(self, tmp):
        path = Path(tmp) / "models" / "v001"
        path.mkdir(parents=True)
        (path / "model_v001").write_bytes(b"weights")
        (path / "labels_v001.json").write_text('["A"]', encoding="utf-8")
        return path

    def test_completed_training_survives_destroyed_tab_widgets(self):
        with tempfile.TemporaryDirectory() as td:
            h = _Cycle(td, "logs")
            model_dir = self._model_dir(td)
            with patch("iat_model_cycle_mixin.cleanup_training_dataset_view"), \
                 patch("iat_model_cycle_mixin.save_cycle_metadata") as metadata:
                h._cycle_training_finished(
                    (True, 0, model_dir, "v001", ["batch_0001.csv"], {}, "")
                )
            self.assertEqual(h._cycle_stage_progress["training"].get(), 100)
            self.assertEqual(h._cycle_trained_version, "v001")
            self.assertEqual(h._cycle_model_path, model_dir / "model_v001")
            self.assertIn("pronto", h._cycle_status.get())
            self.assertEqual(h.refreshed, 0)
            self.assertEqual(h.target_updated, 0)
            self.assertEqual(metadata.call_args.args[2]["status"], "trained")

    def test_completed_training_enables_live_controls(self):
        with tempfile.TemporaryDirectory() as td:
            h = _Cycle(td, "model_cycle")
            for name in ("_cycle_train_btn", "_cycle_apply_btn", "_cycle_eval_btn",
                         "_cycle_model_combo", "_cycle_target_manual_host"):
                setattr(h, name, _Widget(True))
            model_dir = self._model_dir(td)
            with patch("iat_model_cycle_mixin.cleanup_training_dataset_view"), \
                 patch("iat_model_cycle_mixin.save_cycle_metadata"):
                h._cycle_training_finished((True, 0, model_dir, "v001", [], {}, ""))
            self.assertEqual(h._cycle_train_btn.state, "normal")
            self.assertEqual(h._cycle_apply_btn.state, "normal")
            self.assertEqual(h._cycle_eval_btn.state, "normal")
            self.assertEqual(h.refreshed, 1)
            self.assertEqual(h.target_updated, 1)

    def test_clustering_completion_survives_destroyed_button(self):
        with tempfile.TemporaryDirectory() as td:
            h = _Cycle(td, "logs")
            h._cycle_cluster_finished((True, 0, "v001", {}, ""))
            self.assertEqual(h._cycle_stage_progress["cluster"].get(), 100)
            self.assertEqual(h.finalized, [("v001", True)])
            self.assertEqual(h.reloaded, 1)

    def test_class_colors_stay_equal_with_different_batch_class_sets(self):
        a = ["Calanus", "Oithona", "Oncaea"]
        b = ["Oithona", "Temora"]
        color_a = {name: get_class_color(name) for name in a}
        color_b = {name: get_class_color(name) for name in b}
        self.assertEqual(color_a["Oithona"], color_b["Oithona"])
        self.assertEqual(get_class_color("Calanus"), get_class_color("Calanus"))
        self.assertRegex(color_a["Calanus"], r"^#[0-9a-f]{6}$")
        self.assertNotEqual(color_a["Calanus"], color_a["Oncaea"])
        self.assertEqual(get_class_color(""), get_color(0))


if __name__ == "__main__":
    unittest.main()
