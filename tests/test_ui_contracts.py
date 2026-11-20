from __future__ import annotations

import ast
import unittest
from pathlib import Path


class UIContractTests(unittest.TestCase):
    def _method(self, filename, class_name, method_name):
        src = Path(__file__).resolve().parents[1] / "src" / filename
        tree = ast.parse(src.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        return next(node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name)

    def test_iat_log_builder_is_bound_instance_method(self):
        method = self._method("iat_workspace.py", "IATWorkspace", "_build_logs")
        decorators = {
            dec.id for dec in method.decorator_list if isinstance(dec, ast.Name)
        }
        self.assertNotIn("staticmethod", decorators)
        self.assertGreaterEqual(len(method.args.args), 1)
        self.assertEqual(method.args.args[0].arg, "self")

    def test_state_refresh_is_bound_instance_method(self):
        method = self._method("iat_workspace.py", "IATWorkspace", "_refresh_state_cards")
        decorators = {
            dec.id for dec in method.decorator_list if isinstance(dec, ast.Name)
        }
        self.assertNotIn("staticmethod", decorators)
        self.assertEqual(method.args.args[0].arg, "self")


    def test_training_scroll_area_is_bound_instance_method(self):
        method = self._method("iat_training_export_mixin.py", "IATTrainingExportMixin", "_training_scroll_area")
        decorators = {
            dec.id for dec in method.decorator_list if isinstance(dec, ast.Name)
        }
        self.assertNotIn("staticmethod", decorators)
        self.assertGreaterEqual(len(method.args.args), 2)
        self.assertEqual(method.args.args[0].arg, "self")
        self.assertEqual(method.args.args[1].arg, "parent")



if __name__ == "__main__":
    unittest.main()
