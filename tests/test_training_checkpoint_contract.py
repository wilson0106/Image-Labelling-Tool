from __future__ import annotations
import tempfile
import unittest
from pathlib import Path

import torch

from train import EarlyStopper, _load_published_model_for_evaluation


class _DummyScaler:
    def state_dict(self):
        return {}


class TrainingCheckpointContractTests(unittest.TestCase):
    def test_first_zero_metric_still_publishes_best_and_evaluation_reloads_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            stopper = EarlyStopper(patience=2, max_val_acc=0.0)
            stopper.early_stop(
                model, "valid_acc", 0.0, optimizer, _DummyScaler(),
                str(root), "demo", epoch=1, elapsed=1.0,
            )
            best = root / "model_demo"
            self.assertTrue(best.is_file())
            saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(10.0)
            meta = _load_published_model_for_evaluation(model, str(best), torch.device("cpu"))
            self.assertEqual(meta.get("epoch"), 1)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, saved[key]))


if __name__ == "__main__":
    unittest.main()
