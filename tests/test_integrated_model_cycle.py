import tempfile
import unittest
from pathlib import Path

import pandas as pd

from project_model_cycle import allocate_counts, build_reference_assignments, next_model_version


class _FakeProject:
    def __init__(self, root):
        self.project_dir = Path(root)
        self.images_dir = self.project_dir / "images"
        self.images_dir.mkdir()
        self._frames = {}

    def load_batch(self, name, force=False):
        return self._frames[name]

    def resolve_image(self, csv_name, image_name):
        bid = csv_name.split("_project")[0]
        path = self.images_dir / bid / "samples" / image_name
        return path if path.exists() else None


class IntegratedModelCycleTests(unittest.TestCase):
    def test_reference_assignments_use_only_selected_batches_and_allow_grouping(self):
        with tempfile.TemporaryDirectory() as td:
            project = _FakeProject(td)
            for bid in ("batch_0001", "batch_0002"):
                folder = project.images_dir / bid / "samples"
                folder.mkdir(parents=True)
                rows = []
                for cls, count in (("A", 4), ("B", 2)):
                    for idx in range(count):
                        name = f"{bid}_{cls}_{idx}.png"
                        (folder / name).write_bytes(b"x")
                        rows.append({"names": name, "manual_label": cls})
                project._frames[f"{bid}_project.csv"] = pd.DataFrame(rows)

            plan = {
                "A": {"enabled": True, "final": "AB", "train": 50, "valid": 25, "test": 25, "mode": "percent"},
                "B": {"enabled": True, "final": "AB", "train": 1, "valid": 1, "test": 0, "mode": "count"},
            }
            assignments = build_reference_assignments(
                project, ["batch_0001_project.csv", "batch_0002_project.csv"], plan, seed=17
            )
            self.assertEqual(len(assignments), 10)
            self.assertEqual({x.final_class for x in assignments}, {"AB"})
            self.assertTrue(all("batch_0001" in x.image_name or "batch_0002" in x.image_name for x in assignments))

    def test_percentage_and_count_allocation_are_per_class(self):
        self.assertEqual(
            allocate_counts(100, 70, 15, 15, "percent"),
            {"train": 70, "valid": 15, "test": 15},
        )
        self.assertEqual(
            allocate_counts(37, 20, 7, 4, "count"),
            {"train": 20, "valid": 7, "test": 4},
        )

    def test_model_versions_are_project_local(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(next_model_version(td), "v001")
            (Path(td) / "models" / "v001").mkdir(parents=True)
            (Path(td) / "models" / "v003").mkdir()
            self.assertEqual(next_model_version(td), "v004")


    def test_model_application_versions_embeddings_and_preserves_reference_manual_labels(self):
        import json
        import sys
        import types
        import numpy as np
        from unittest.mock import patch
        from project_model_cycle import apply_model_in_place

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project_dir = root / "project"
            dataframes = project_dir / "dataframes"
            embeddings = project_dir / "embeddings"
            dataframes.mkdir(parents=True)
            embeddings.mkdir()
            (project_dir / "images" / "batch_0001" / "samples").mkdir(parents=True)

            csv_name = "batch_0001_project.csv"
            old_embed = embeddings / "batch_0001.npz"
            np.savez_compressed(old_embed, feats=np.ones((2, 3), dtype=np.float32))
            original = pd.DataFrame({
                "names": ["a.png", "b.png"],
                "manual_label": ["human_A", "human_B"],
                "correct_label": ["human_A", "human_B"],
                "preds1": ["old", "old"],
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
                "embed_file": ["embeddings/batch_0001.npz"] * 2,
                "embed_idx": [0, 1],
            })
            original.to_csv(dataframes / csv_name, index=False)

            class FakeProject:
                def __init__(self):
                    self.project_dir = project_dir
                    self.images_dir = project_dir / "images"
                    self.embeddings_dir = embeddings
                    self.dataframes_dir = dataframes
                    self.batch_cache = {}
                    self._overview_cache = None
                    self._image_global_index = None
                def load_batch(self, name, force=False):
                    return pd.read_csv(self.dataframes_dir / name)

            model_file = root / "model.bin"
            model_file.write_bytes(b"model")
            labels_file = root / "labels.json"
            labels_file.write_text(json.dumps({"new_A": 0, "new_B": 1}), encoding="utf-8")

            fake_features = types.ModuleType("features")
            fake_features.get_model = lambda **kwargs: object()
            fake_features.compute_features = lambda images, bid, model, weights, labels: (
                np.arange(6, dtype=np.float32).reshape(2, 3),
                ["a.png", "b.png"],
                np.array([0, 1]),
                [
                    {"top1": ("new_A", .9), "top2": ("new_B", .1), "top3": ("new_B", 0.0)},
                    {"top1": ("new_B", .8), "top2": ("new_A", .2), "top3": ("new_A", 0.0)},
                ],
            )
            with patch.dict(sys.modules, {"features": fake_features}):
                result = apply_model_in_place(
                    FakeProject(), [csv_name], model_file, labels_file, "v002",
                    preserve_manual_batches=[csv_name],
                )

            updated = pd.read_csv(dataframes / csv_name)
            self.assertEqual(updated["manual_label"].tolist(), ["human_A", "human_B"])
            self.assertEqual(updated["feature_model_version"].astype(str).tolist(), ["v002", "v002"])
            self.assertTrue(updated["projection_stale"].astype(str).str.lower().isin(["true", "1"]).all())
            self.assertEqual(set(updated["embed_file"].astype(str)), {"embeddings/v002/batch_0001.npz"})
            self.assertTrue((project_dir / "embeddings" / "v002" / "batch_0001.npz").is_file())
            self.assertTrue((project_dir / "model_history" / "v002" / "dataframes_before" / csv_name).is_file())
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_partial_model_application_rolls_back_committed_csvs_on_later_mismatch(self):
        import json
        import sys
        import types
        import numpy as np
        from unittest.mock import patch
        from project_model_cycle import apply_model_in_place

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            project_dir = root / "project"
            dataframes = project_dir / "dataframes"
            embeddings = project_dir / "embeddings"
            dataframes.mkdir(parents=True)
            embeddings.mkdir()
            (project_dir / "images").mkdir(exist_ok=True)

            csvs = []
            originals = {}
            for b in (1, 2):
                csv_name = f"batch_{b:04d}_project.csv"
                csvs.append(csv_name)
                df = pd.DataFrame({
                    "names": [f"{b}_a.png", f"{b}_b.png"],
                    "manual_label": ["old_A", "old_B"],
                    "correct_label": ["old_A", "old_B"],
                    "preds1": ["old_A", "old_B"],
                    "x": [1.0, 2.0], "y": [3.0, 4.0],
                    "embed_file": [f"embeddings/batch_{b:04d}.npz"] * 2,
                    "embed_idx": [0, 1],
                })
                df.to_csv(dataframes / csv_name, index=False)
                originals[csv_name] = (dataframes / csv_name).read_bytes()

            class FakeProject:
                def __init__(self):
                    self.project_dir = project_dir
                    self.images_dir = project_dir / "images"
                    self.embeddings_dir = embeddings
                    self.dataframes_dir = dataframes
                    self.batch_cache = {}
                    self._overview_cache = None
                    self._image_global_index = None
                def load_batch(self, name, force=False):
                    return pd.read_csv(self.dataframes_dir / name)

            model_file = root / "model.bin"; model_file.write_bytes(b"model")
            labels_file = root / "labels.json"; labels_file.write_text(json.dumps({"A": 0, "B": 1}), encoding="utf-8")
            fake_features = types.ModuleType("features")
            fake_features.get_model = lambda **kwargs: object()

            def compute_features(images, bid, model, weights, labels):
                b = int(bid.split("_")[-1])
                names = [f"{b}_a.png", f"{b}_b.png"] if b == 1 else [f"{b}_a.png"]
                n = len(names)
                confs = [
                    {"top1": ("A", .9), "top2": ("B", .1), "top3": ("B", 0.0)}
                    for _ in range(n)
                ]
                return np.zeros((n, 3), np.float32), names, np.zeros(n, int), confs

            fake_features.compute_features = compute_features
            with patch.dict(sys.modules, {"features": fake_features}):
                with self.assertRaises(RuntimeError):
                    apply_model_in_place(FakeProject(), csvs, model_file, labels_file, "v003")

            for csv_name in csvs:
                self.assertEqual((dataframes / csv_name).read_bytes(), originals[csv_name])
            manifest = json.loads(
                (project_dir / "model_history" / "v003" / "application_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "rolled_back_after_failure")


    def test_model_application_dtype_normalization_accepts_numeric_updates(self):
        import numpy as np
        from project_model_cycle import _prepare_model_application_dtypes

        df = pd.DataFrame({
            "pred": pd.Series(["0", "1"], dtype="string"),
            "confs1": pd.Series(["0.1", "0.2"], dtype="string"),
            "confs2": pd.Series(["0.0", "0.0"], dtype="string"),
            "confs3": pd.Series(["0.0", "0.0"], dtype="string"),
            "embed_idx": pd.Series(["0", "1"], dtype="string"),
            "preds1": pd.Series(["old", "old"], dtype="string"),
            "embed_file": pd.Series(["old.npz", "old.npz"], dtype="string"),
            "model_version": pd.Series(["v001", "v001"], dtype="string"),
            "feature_model_version": pd.Series(["v001", "v001"], dtype="string"),
            "projection_stale": pd.Series(["False", "False"], dtype="string"),
        })
        out = _prepare_model_application_dtypes(df)
        mask = pd.Series([True, True])
        out.loc[mask, "pred"] = [1, 0]
        out.loc[mask, "confs1"] = [0.9, 0.8]
        out.loc[mask, "embed_idx"] = np.array([5, 6])
        out.loc[mask, "preds1"] = ["A", "B"]
        out.loc[mask, "model_version"] = "v002"
        out.loc[mask, "projection_stale"] = [True, False]
        self.assertEqual(out["pred"].tolist(), [1, 0])
        self.assertEqual(out["confs1"].tolist(), [0.9, 0.8])
        self.assertEqual(out["preds1"].tolist(), ["A", "B"])



    def test_recompute_projection_updates_xy_and_clears_stale_flag(self):
        import sys
        import types
        import numpy as np
        from unittest.mock import patch
        from project_model_cycle import recompute_projection_for_version

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataframes = root / "dataframes"
            embeddings = root / "embeddings" / "v002"
            dataframes.mkdir(parents=True)
            embeddings.mkdir(parents=True)

            csvs = []
            for b in (1, 2):
                csv_name = f"batch_{b:04d}_project.csv"
                csvs.append(csv_name)
                feats = np.array([[b, 0.0], [b, 1.0], [b, 2.0]], dtype=np.float32)
                np.savez_compressed(embeddings / f"batch_{b:04d}.npz", feats=feats)
                pd.DataFrame({
                    "names": [f"{b}_a.png", f"{b}_b.png", f"{b}_c.png"],
                    "x": [99.0, 99.0, 99.0],
                    "y": [88.0, 88.0, 88.0],
                    "embed_file": [f"embeddings/v002/batch_{b:04d}.npz"] * 3,
                    "embed_idx": [0, 1, 2],
                    "feature_model_version": ["v002"] * 3,
                    "projection_model_version": ["v001"] * 3,
                    "projection_stale": [True] * 3,
                }).to_csv(dataframes / csv_name, index=False)

            class FakeProject:
                def __init__(self):
                    self.project_dir = root
                    self.dataframes_dir = dataframes
                    self.batch_cache = {}
                    self._overview_cache = None
                def csv_names(self):
                    return list(csvs)
                def load_batch(self, name, force=False):
                    return pd.read_csv(self.dataframes_dir / name)

            class FakeEmbedding:
                def __init__(self, feats):
                    self.feats = np.asarray(feats, dtype=float)
                def __array__(self, dtype=None):
                    arr = self.feats[:, :2] * 10.0
                    return arr.astype(dtype) if dtype is not None else arr

            fake_projections = types.ModuleType("projections")
            fake_projections.opentsne_fit = lambda feats: FakeEmbedding(feats)
            fake_projections.opentsne_transform = lambda feats, base: np.asarray(feats)[:, :2] * 10.0

            with patch.dict(sys.modules, {"projections": fake_projections}):
                result = recompute_projection_for_version(FakeProject(), "v002")

            self.assertEqual(result["reference_batch"], "batch_0001_project.csv")
            self.assertEqual(result["images"], 6)
            for csv_name in csvs:
                df = pd.read_csv(dataframes / csv_name)
                self.assertTrue((df["projection_model_version"].astype(str) == "v002").all())
                self.assertFalse(df["projection_stale"].astype(str).str.lower().isin(["true", "1"]).any())
                self.assertFalse((df["x"] == 99.0).any())
                self.assertFalse((df["y"] == 88.0).any())
            ref = root / "model_history" / "v002" / "projection_reference.json"
            self.assertTrue(ref.is_file())

    def test_saved_model_discovery_does_not_require_retraining(self):
        from iat_model_cycle_mixin import IATModelCycleMixin

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for version in ("v001", "v003"):
                model_dir = root / "models" / version
                model_dir.mkdir(parents=True)
                (model_dir / f"model_{version}").write_bytes(b"model")
                (model_dir / f"labels_{version}.json").write_text('{"A": 0}', encoding="utf-8")

            obj = object.__new__(IATModelCycleMixin)
            obj.project = type("P", (), {"project_dir": root})()
            entries = obj._cycle_saved_model_entries()
            self.assertEqual(list(entries), ["v001", "v003"])
            self.assertEqual(entries["v003"][0].name, "model_v003")
            self.assertEqual(entries["v003"][1].name, "labels_v003.json")


if __name__ == "__main__":
    unittest.main()
