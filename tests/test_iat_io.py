from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iat_io import CopyJob, copy_jobs


class IATIOTests(unittest.TestCase):
    def test_copy_jobs_copies_files_and_reports_throughput(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "src"
            dst = base / "dst"
            src.mkdir()
            jobs = []
            for i in range(8):
                p = src / f"image_{i}.dat"
                p.write_bytes((f"payload-{i}" * 100).encode())
                jobs.append(CopyJob(p, dst / f"class_{i%2}" / p.name, i))
            events = []
            report = copy_jobs(jobs, progress=lambda cur, total, text: events.append((cur, total)), workers=2)
            self.assertEqual(report.copied, 8)
            self.assertEqual(report.failed, 0)
            self.assertEqual(report.succeeded_tokens, set(range(8)))
            self.assertGreater(report.bytes_copied, 0)
            self.assertGreaterEqual(report.elapsed_seconds, 0)
            self.assertTrue(events)
            self.assertEqual(events[-1], (8, 8))
            for job in jobs:
                self.assertEqual(job.src.read_bytes(), job.dest.read_bytes())

    def test_copy_jobs_reports_failure_without_aborting_other_files(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            good = base / "good.dat"
            good.write_text("ok")
            missing = base / "missing.dat"
            jobs = [
                CopyJob(good, base / "out" / "good.dat", "good"),
                CopyJob(missing, base / "out" / "missing.dat", "missing"),
            ]
            report = copy_jobs(jobs, workers=2)
            self.assertEqual(report.copied, 1)
            self.assertEqual(report.failed, 1)
            self.assertIn("good", report.succeeded_tokens)
            self.assertIn("missing", report.failed_tokens)


    def test_copy_jobs_bounds_submitted_futures_for_large_job_list(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "src"
            dst = base / "dst"
            src.mkdir()
            jobs = []
            for i in range(160):
                p = src / f"image_{i:04d}.dat"
                p.write_bytes(b"x" * 32)
                jobs.append(CopyJob(p, dst / f"c{i%3}" / p.name, i))
            report = copy_jobs(
                jobs,
                workers=2,
                chunk_size=32,
                max_pending_factor=3,
            )
            self.assertEqual(report.copied, len(jobs))
            self.assertLessEqual(report.max_in_flight, 6)
            self.assertEqual(report.requested, len(jobs))


if __name__ == "__main__":
    unittest.main()
