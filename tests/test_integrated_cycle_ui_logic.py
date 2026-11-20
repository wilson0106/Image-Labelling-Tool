
import unittest

from iat_model_cycle_mixin import IATModelCycleMixin


class _Var:
    def __init__(self, value=False):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class _Project:
    def csv_names(self):
        return ["batch_0001_project.csv", "batch_0002_project.csv", "batch_0003_project.csv"]


class _Harness(IATModelCycleMixin):
    def __init__(self):
        self.project = _Project()
        self._cycle_batch_vars = {
            "batch_0001_project.csv": _Var(True),
            "batch_0002_project.csv": _Var(False),
            "batch_0003_project.csv": _Var(False),
        }
        self._cycle_target_mode = _Var("manual")
        self._cycle_target_vars = {
            "batch_0002_project.csv": _Var(True),
            "batch_0003_project.csv": _Var(False),
        }
        self._cycle_target_row_widgets = {}


class IntegratedCycleSelectionLogicTests(unittest.TestCase):
    def test_manual_target_selection_is_respected_and_excludes_reference(self):
        h = _Harness()
        self.assertEqual(h._cycle_target_batches(), ["batch_0002_project.csv"])
        h._cycle_target_vars["batch_0002_project.csv"].set(False)
        h._cycle_target_vars["batch_0003_project.csv"].set(True)
        self.assertEqual(h._cycle_target_batches(), ["batch_0003_project.csv"])

    def test_select_all_targets_changes_processing_state(self):
        h = _Harness()
        h._cycle_set_all_targets(True)
        self.assertEqual(
            h._cycle_target_batches(),
            ["batch_0002_project.csv", "batch_0003_project.csv"],
        )
        h._cycle_set_all_targets(False)
        self.assertEqual(h._cycle_target_batches(), [])


if __name__ == "__main__":
    unittest.main()
