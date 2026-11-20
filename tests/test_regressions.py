from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scipy.cluster.hierarchy import is_valid_linkage

from ilt_contracts import (
    ensure_unique_sanitized_labels,
    resolve_scatter_class_column,
    safe_perplexity,
)
from ilt_clustering import hierarchical_linkage
import class_clustering
from iat_backend import (
    IATProject,
    analyze_embedding_space,
    generate_cluster_artifact,
    reconcile_class_cluster_artifacts,
    sort_gallery_multi,
    strip_segmentation_columns,
    validation_capabilities,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_project(base: Path, batches: int = 6, per_batch: int = 20, dim: int = 8) -> Path:
    project = base / "project"
    (project / "dataframes").mkdir(parents=True)
    (project / "embeddings").mkdir()
    (project / "clusters").mkdir()

    rng = np.random.default_rng(7)
    for b in range(1, batches + 1):
        bn = f"batch_{b:04d}_project.csv"
        ef = f"batch_{b:04d}.npz"
        feats = rng.normal(size=(per_batch, dim)).astype(np.float32)
        feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
        np.savez_compressed(project / "embeddings" / ef, feats=feats)

        names = [f"img_{b:02d}_{i:03d}.png" for i in range(per_batch)]
        labels = ["old_class" if i < per_batch // 2 else "stable" for i in range(per_batch)]
        df = pd.DataFrame({
            "names": names,
            "custom_data": names,
            "manual_label": labels,
            "correct_label": labels,
            "source_label": ["source"] * per_batch,
            "preds1": labels,
            "confs1": np.linspace(0.2, 0.95, per_batch),
            "var_norm": np.linspace(0.0, 1.0, per_batch),
            "colors": [1 if x == "old_class" else 2 for x in labels],
            "x": np.arange(per_batch, dtype=float),
            "y": np.arange(per_batch, dtype=float) * 0.5,
            "embed_file": [f"embeddings/{ef}"] * per_batch,
            "embed_idx": np.arange(per_batch, dtype=int),
        })
        df.to_csv(project / "dataframes" / bn, index=False)
    return project


class ContractTests(unittest.TestCase):
    def test_scatter_semantics_are_explicit(self):
        cols = {
            "manual_label", "source_label", "preds1", "pred",
            "hdbscan_cluster", "confs1", "review_priority",
        }
        self.assertEqual(resolve_scatter_class_column("manual_label", cols), "manual_label")
        self.assertEqual(resolve_scatter_class_column("prediction", cols), "preds1")
        self.assertEqual(resolve_scatter_class_column("source_label", cols), "source_label")
        self.assertEqual(resolve_scatter_class_column("hdbscan_cluster", cols), "hdbscan_cluster")
        # Continuous diagnostics filter classes by the current human label.
        self.assertEqual(resolve_scatter_class_column("confs1", cols), "manual_label")
        self.assertEqual(resolve_scatter_class_column("review_priority", cols), "manual_label")

    def test_sanitized_class_collision_is_rejected(self):
        with self.assertRaises(ValueError):
            ensure_unique_sanitized_labels(["a/b", "a\\b"])

    def test_safe_perplexity_small_n(self):
        self.assertGreater(safe_perplexity(2), 0)
        self.assertLess(safe_perplexity(2), 2)
        self.assertLess(safe_perplexity(10), 10)


class ClusteringEngineTests(unittest.TestCase):
    def test_full_linkage_valid(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 12)).astype(np.float32)
        result = hierarchical_linkage(
            X, metric="cosine", method="average",
            pca_dim=8, max_full=1500, max_centroids=1500,
        )
        self.assertEqual(result.mode, "full")
        self.assertEqual(result.linkage.shape, (79, 4))
        self.assertTrue(is_valid_linkage(result.linkage, throw=False))

    def test_centroid_regression_3883(self):
        # Exact N from the real failure that produced index 3882 out-of-bounds.
        rng = np.random.default_rng(123)
        X = rng.normal(size=(3883, 12)).astype(np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        result = hierarchical_linkage(
            X, metric="cosine", method="average",
            pca_dim=0, max_full=1500, max_centroids=1500,
        )
        self.assertEqual(result.mode, "centroid")
        self.assertEqual(result.linkage.shape, (3882, 4))
        self.assertTrue(is_valid_linkage(result.linkage, throw=False))

    def test_standalone_requires_valid_manual_label_when_present(self):
        df = pd.DataFrame({
            "manual_label": [True, False],
            "preds1": ["a", "b"],
        })
        with self.assertRaises(RuntimeError):
            class_clustering._pick_label_column(df)

    def test_standalone_wrapper_uses_same_engine(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(240, 10)).astype(np.float32)
        Z_wrapper = class_clustering._centroid_expanded_linkage(
            X, method="average", metric="cosine",
            max_centroids=60, verbose=False,
        )
        # The shared function is deterministic under random_state=0.
        from ilt_clustering import centroid_expanded_linkage
        Z_shared = centroid_expanded_linkage(
            X, method="average", metric="cosine", max_centroids=60,
        )
        self.assertTrue(np.array_equal(Z_wrapper, Z_shared))


    def test_mixed_feature_versions_are_rejected_unless_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dfs = root / "dataframes"
            dfs.mkdir()
            for idx, version in enumerate(("v001", "v002"), 1):
                pd.DataFrame({
                    "names": [f"img_{idx}.png"],
                    "embed_file": [f"embeddings/{version}/batch_{idx:04d}.npz"],
                    "embed_idx": [0],
                    "manual_label": ["A"],
                    "feature_model_version": [version],
                }).to_csv(dfs / f"batch_{idx:04d}_project.csv", index=False)
            with self.assertRaises(RuntimeError):
                class_clustering._read_index_table(dfs)
            filtered = class_clustering._read_index_table(dfs, feature_version="v002")
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered.attrs["feature_model_version"], "v002")


class IATPersistenceTests(unittest.TestCase):
    def test_relabel_persists_across_small_cache_and_reconciles_clusters(self):
        old_env = os.environ.get("ILT_IAT_BATCH_CACHE")
        os.environ["ILT_IAT_BATCH_CACHE"] = "2"
        try:
            with tempfile.TemporaryDirectory() as td:
                project_dir = make_project(Path(td))
                project = IATProject(project_dir)
                ov = project.load_overview()

                old_rows = ov.loc[ov["manual_label"].astype(str) == "old_class"].copy()
                generate_cluster_artifact(
                    project, "old_class", rows=old_rows,
                    metric="cosine", method="average",
                    pca_dim=0, max_full=1500, max_centroids=1500,
                )
                self.assertTrue((project_dir / "clusters" / "old_class.npz").exists())

                gids = old_rows["gid"].astype(str).tolist()
                a = gids[::2]
                b = gids[1::2]
                self.assertEqual(project.apply_labels(a, "new_A"), len(a))
                self.assertEqual(project.apply_labels(b, "new_B"), len(b))
                project.save_dirty()

                # Fresh process-equivalent reload: old manual class must stay extinct.
                fresh = IATProject(project_dir)
                ov2 = fresh.load_overview()
                counts = ov2["manual_label"].astype(str).value_counts().to_dict()
                self.assertEqual(counts.get("old_class", 0), 0)
                self.assertEqual(counts.get("new_A", 0), len(a))
                self.assertEqual(counts.get("new_B", 0), len(b))

                result = reconcile_class_cluster_artifacts(
                    fresh,
                    class_labels=["old_class", "new_A", "new_B"],
                    metric="cosine", method="average",
                    pca_dim=0, max_full=1500, max_centroids=1500,
                    discover_stale=True,
                )
                self.assertFalse(result["errors"], result["errors"])
                self.assertFalse((project_dir / "clusters" / "old_class.npz").exists())
                self.assertTrue((project_dir / "clusters" / "new_A.npz").exists())
                self.assertTrue((project_dir / "clusters" / "new_B.npz").exists())
        finally:
            if old_env is None:
                os.environ.pop("ILT_IAT_BATCH_CACHE", None)
            else:
                os.environ["ILT_IAT_BATCH_CACHE"] = old_env

    def test_assisted_validation_is_read_only_on_disk(self):
        caps = validation_capabilities()
        if not caps.get("hdbscan"):
            self.skipTest("HDBSCAN unavailable in this environment")

        with tempfile.TemporaryDirectory() as td:
            project_dir = make_project(Path(td), batches=2, per_batch=20, dim=8)
            project = IATProject(project_dir)
            before = {
                p: sha256(p)
                for folder in ("dataframes", "embeddings")
                for p in sorted((project_dir / folder).glob("*"))
            }

            rows = project.load_overview().head(30).copy()
            analysis, meta = analyze_embedding_space(
                project,
                rows,
                projection="pca",
                pca_dim=6,
                knn_k=5,
                min_cluster_size=4,
            )
            self.assertEqual(len(analysis), 30)
            self.assertIn("review_priority", analysis.columns)
            self.assertIn("hdbscan_outlier", analysis.columns)

            after = {p: sha256(p) for p in before}
            self.assertEqual(before, after)

    def test_transient_validation_columns_are_not_persisted(self):
        df = pd.DataFrame({
            "names": ["a.png"],
            "manual_label": ["a"],
            "review_priority": [0.9],
            "knn_agreement": [0.2],
            "hdbscan_outlier": [1.0],
            "analysis_x": [2.0],
            "analysis_y": [3.0],
        })
        stripped = strip_segmentation_columns(df)
        self.assertIn("manual_label", stripped.columns)
        for col in ("review_priority", "knn_agreement", "hdbscan_outlier", "analysis_x", "analysis_y"):
            self.assertNotIn(col, stripped.columns)

    def test_gallery_az_means_manual_label(self):
        df = pd.DataFrame({
            "names": ["003.png", "001.png", "002.png"],
            "manual_label": ["zeta", "alpha", "beta"],
            "preds1": ["alpha", "zeta", "beta"],
        })
        out = sort_gallery_multi(df, ["manual_label_asc"])
        self.assertEqual(out["manual_label"].tolist(), ["alpha", "beta", "zeta"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
