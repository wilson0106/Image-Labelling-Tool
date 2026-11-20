from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path

from training_dataset import (
    DATASET_MODE_CLASSES,
    DATASET_MODE_INVALID,
    DATASET_MODE_SPLIT,
    allocate_from_percentages,
    allocation_changes_dataset,
    build_training_assignments,
    build_training_class_plan,
    cleanup_training_dataset_view,
    create_training_dataset_view,
    inspect_training_dataset,
    list_split_image_files,
    mapping_changes_dataset,
    normalize_count_allocation,
    scan_training_dataset,
    write_training_split_csv,
)


class TrainingDatasetServiceTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        for split in ("train", "valid", "test"):
            for class_name in ("A", "B", "C"):
                folder = dataset / split / class_name
                folder.mkdir(parents=True, exist_ok=True)
        counts = {
            "train": {"A": 2, "B": 3, "C": 1},
            "valid": {"A": 1, "B": 2, "C": 1},
            "test": {"A": 1, "B": 1, "C": 1},
        }
        for split, classes in counts.items():
            for class_name, amount in classes.items():
                for idx in range(amount):
                    (dataset / split / class_name / f"{split}_{class_name}_{idx}.png").write_bytes(b"img")
        (dataset / "train" / "A" / "notes.txt").write_text("ignore", encoding="utf-8")
        return dataset

    def _make_flat_dataset(self, root: Path) -> Path:
        dataset = root / "flat"
        for class_name, amount in {"Taxon_A": 10, "Taxon_B": 7, "Taxon_C": 4}.items():
            folder = dataset / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for idx in range(amount):
                (folder / f"{class_name}_{idx:02d}.png").write_bytes(b"img")
            (folder / "ignore.txt").write_text("x", encoding="utf-8")
        return dataset

    def test_scan_counts_supported_images_in_one_consistent_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(Path(tmp))
            counts, key = scan_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            self.assertEqual(Path(key), dataset.resolve())
            self.assertEqual(counts["A"], {"train": 2, "valid": 1, "test": 1})
            self.assertEqual(counts["B"], {"train": 3, "valid": 2, "test": 1})
            self.assertEqual(counts["C"], {"train": 1, "valid": 1, "test": 1})

    def test_inspection_detects_both_supported_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dataset = self._make_dataset(root)
            split = inspect_training_dataset(
                split_dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            self.assertEqual(split.mode, DATASET_MODE_SPLIT)
            self.assertEqual(split.totals, {"A": 4, "B": 6, "C": 3})

            flat_dataset = self._make_flat_dataset(root)
            flat = inspect_training_dataset(
                flat_dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            self.assertEqual(flat.mode, DATASET_MODE_CLASSES)
            self.assertEqual(flat.totals, {"Taxon_A": 10, "Taxon_B": 7, "Taxon_C": 4})
            self.assertEqual(flat.counts["Taxon_A"], {"train": 0, "valid": 0, "test": 0})

    def test_partial_split_layout_is_rejected_instead_of_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            (dataset / "train" / "A").mkdir(parents=True)
            (dataset / "train" / "A" / "img.png").write_bytes(b"img")
            inspection = inspect_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            self.assertEqual(inspection.mode, DATASET_MODE_INVALID)
            self.assertEqual(inspection.counts, {})

    def test_split_file_listing_is_sorted_and_ignores_non_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(Path(tmp))
            files = list_split_image_files(dataset / "train")
            names = [
                Path(path).relative_to(dataset / "train").as_posix()
                for path in files
            ]
            self.assertEqual(
                names,
                [
                    "A/train_A_0.png", "A/train_A_1.png",
                    "B/train_B_0.png", "B/train_B_1.png", "B/train_B_2.png",
                    "C/train_C_0.png",
                ],
            )

    def test_percentage_allocation_is_deterministic_and_uses_all_at_100(self):
        self.assertEqual(
            allocate_from_percentages(
                7, {"train": 70, "valid": 15, "test": 15}
            ),
            {"train": 5, "valid": 1, "test": 1},
        )
        result = allocate_from_percentages(
            11, {"train": 60, "valid": 20, "test": 10}
        )
        self.assertEqual(sum(result.values()), 10)
        with self.assertRaisesRegex(ValueError, "split_percentage_exceeds_100"):
            allocate_from_percentages(
                10, {"train": 80, "valid": 20, "test": 1}
            )

    def test_explicit_count_allocation_validates_available_total(self):
        self.assertEqual(
            normalize_count_allocation(
                10, {"train": 6, "valid": 2, "test": 1}
            ),
            {"train": 6, "valid": 2, "test": 1},
        )
        with self.assertRaisesRegex(ValueError, "split_allocation_exceeds_total"):
            normalize_count_allocation(
                10, {"train": 8, "valid": 2, "test": 1}
            )

    def test_class_plan_groups_filters_and_uses_requested_allocations(self):
        counts = {
            "A": {"train": 2, "valid": 1, "test": 1},
            "B": {"train": 3, "valid": 2, "test": 1},
            "C": {"train": 1, "valid": 1, "test": 1},
        }
        allocations = {
            "A": {"train": 3, "valid": 1, "test": 0},
            "B": {"train": 4, "valid": 1, "test": 1},
            "C": {"train": 1, "valid": 1, "test": 1},
        }
        mapping, summary, errors = build_training_class_plan(
            counts,
            {"A": "AB", "B": "AB", "C": "C"},
            {"A": True, "B": True, "C": False},
            allocations=allocations,
            available_totals={"A": 4, "B": 6, "C": 3},
        )
        self.assertEqual(mapping, {"A": "AB", "B": "AB"})
        self.assertEqual(summary["AB"], {"train": 7, "valid": 2, "test": 1})
        self.assertEqual(errors, [])
        self.assertTrue(mapping_changes_dataset(mapping, counts))

    def test_class_plan_reports_invalid_and_empty_split_states(self):
        counts = {
            "A": {"train": 0, "valid": 1, "test": 0},
        }
        mapping, summary, errors = build_training_class_plan(
            counts,
            {"A": "Merged"},
            {"A": True},
        )
        self.assertEqual(mapping, {"A": "Merged"})
        self.assertIn("class_without_train", errors)
        self.assertIn("split_empty_after_mapping", errors)

        invalid_names = ["../bad", "A\\bad", ".", "..", ""] + (["A:B", "CON", "name."] if os.name == "nt" else [])
        for invalid_name in invalid_names:
            with self.subTest(invalid_name=invalid_name):
                mapping, summary, errors = build_training_class_plan(
                    counts,
                    {"A": invalid_name},
                    {"A": True},
                )
                self.assertEqual(mapping, {})
                self.assertEqual(summary, {})
                self.assertEqual(errors, ["class_name_invalid"])

    def test_exact_presplit_allocation_preserves_each_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_dataset(Path(tmp))
            inspection = inspect_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            mapping = {name: name for name in inspection.counts}
            assignments = build_training_assignments(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                inspection.counts,
                dataset_mode=inspection.mode,
                seed=123,
            )
            observed = {
                (item.original_class, item.image_name): item.split
                for item in assignments
            }
            self.assertEqual(observed[("A", "train_A_0.png")], "train")
            self.assertEqual(observed[("A", "valid_A_0.png")], "valid")
            self.assertEqual(observed[("A", "test_A_0.png")], "test")
            self.assertFalse(
                allocation_changes_dataset(
                    inspection.mode, inspection.counts, inspection.counts
                )
            )

    def test_flat_assignments_are_reproducible_and_follow_requested_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._make_flat_dataset(Path(tmp))
            inspection = inspect_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            mapping = {name: name for name in inspection.counts}
            allocations = {
                name: allocate_from_percentages(
                    inspection.totals[name],
                    {"train": 70, "valid": 15, "test": 15},
                )
                for name in mapping
            }
            first = build_training_assignments(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                allocations,
                dataset_mode=inspection.mode,
                seed=42,
            )
            second = build_training_assignments(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                allocations,
                dataset_mode=inspection.mode,
                seed=42,
            )
            key = lambda items: [
                (x.original_class, x.image_name, x.split) for x in items
            ]
            self.assertEqual(key(first), key(second))
            for class_name in mapping:
                per_split = {
                    split: sum(
                        1 for item in first
                        if item.original_class == class_name and item.split == split
                    )
                    for split in ("train", "valid", "test")
                }
                self.assertEqual(per_split, allocations[class_name])
            self.assertTrue(
                allocation_changes_dataset(
                    inspection.mode, inspection.counts, allocations
                )
            )

    def test_split_manifest_has_exact_requested_three_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_flat_dataset(root)
            inspection = inspect_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            mapping = {"Taxon_A": "Merged", "Taxon_B": "Merged"}
            allocations = {
                "Taxon_A": {"train": 6, "valid": 2, "test": 2},
                "Taxon_B": {"train": 5, "valid": 1, "test": 1},
            }
            assignments = build_training_assignments(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                allocations,
                dataset_mode=inspection.mode,
                seed=0,
            )
            csv_path = write_training_split_csv(
                assignments, root / "dataset_split.csv"
            )
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows[0], ["imagem", "classe_taxonomica", "destino"]
            )
            self.assertEqual(len(rows) - 1, len(assignments))
            self.assertTrue(
                all(row[1] in {"Taxon_A", "Taxon_B"} for row in rows[1:])
            )
            self.assertTrue(
                all(row[2] in {"train", "valid", "test"} for row in rows[1:])
            )

    def test_temporary_grouped_view_is_non_destructive_and_cleanup_is_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = self._make_dataset(tmp_path)
            before = sorted(
                str(p.relative_to(dataset))
                for p in dataset.rglob("*")
                if p.is_file()
            )
            counts, _ = scan_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            mapping, summary, errors = build_training_class_plan(
                counts,
                {"A": "AB", "B": "AB", "C": "C"},
                {"A": True, "B": True, "C": False},
            )
            self.assertEqual(errors, [])

            view = create_training_dataset_view(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                summary,
                temp_parent=tmp_path / "views",
            )
            try:
                self.assertEqual(len(list((view / "train" / "AB").glob("*.png"))), 5)
                self.assertEqual(len(list((view / "valid" / "AB").glob("*.png"))), 3)
                self.assertEqual(len(list((view / "test" / "AB").glob("*.png"))), 2)
                self.assertFalse((view / "train" / "C").exists())
                after = sorted(
                    str(p.relative_to(dataset))
                    for p in dataset.rglob("*")
                    if p.is_file()
                )
                self.assertEqual(before, after)
            finally:
                cleanup_training_dataset_view(view)

            self.assertFalse(view.exists())
            cleanup_training_dataset_view(view)

    def test_flat_grouped_view_materializes_train_valid_test_without_touching_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._make_flat_dataset(root)
            before = sorted(
                str(p.relative_to(dataset))
                for p in dataset.rglob("*")
                if p.is_file()
            )
            inspection = inspect_training_dataset(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
            )
            mapping = {
                "Taxon_A": "Taxon_AB",
                "Taxon_B": "Taxon_AB",
                "Taxon_C": "Taxon_C",
            }
            allocations = {
                name: allocate_from_percentages(
                    inspection.totals[name],
                    {"train": 70, "valid": 15, "test": 15},
                )
                for name in mapping
            }
            _, summary, errors = build_training_class_plan(
                inspection.counts,
                mapping,
                {name: True for name in mapping},
                allocations=allocations,
                available_totals=inspection.totals,
            )
            self.assertEqual(errors, [])
            assignments = build_training_assignments(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                allocations,
                dataset_mode=inspection.mode,
                seed=7,
            )
            view = create_training_dataset_view(
                dataset,
                {"train": "train", "valid": "valid", "test": "test"},
                mapping,
                summary,
                allocations=allocations,
                dataset_mode=inspection.mode,
                seed=7,
                assignments=assignments,
                temp_parent=root / "views",
            )
            try:
                self.assertTrue((view / "train" / "Taxon_AB").is_dir())
                self.assertTrue((view / "valid" / "Taxon_AB").is_dir())
                self.assertTrue((view / "test" / "Taxon_C").is_dir())
                self.assertEqual(
                    len(list((view / "train" / "Taxon_AB").glob("*.png"))),
                    allocations["Taxon_A"]["train"]
                    + allocations["Taxon_B"]["train"],
                )
                after = sorted(
                    str(p.relative_to(dataset))
                    for p in dataset.rglob("*")
                    if p.is_file()
                )
                self.assertEqual(before, after)
            finally:
                cleanup_training_dataset_view(view)


if __name__ == "__main__":
    unittest.main()
