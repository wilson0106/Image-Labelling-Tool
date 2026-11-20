"""Regressões: cores estáveis, legenda exportável e quadros animados."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image

from iat_temporal_figure import (
    legend_lines, make_class_palette, render_figure, save_animated_gif, _lab,
)
from matplotlib.colors import to_rgb


class TemporalFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.DataFrame({
            "classe": ["Calanus", "Calanus", "Copepoda", "Copepoda", "som"],
            "x": [-5., -3., 2., 4., 12.],
            "y": [1., 3., 4., 3., -9.],
        })
        cls.a = cls.data.iloc[[0, 2, 3]]
        cls.b = cls.data.iloc[[1, 4]]
        cls.colors = make_class_palette(cls.data["classe"].unique())

    def test_colors_stay_fixed_when_filters_and_modes_change(self):
        full = make_class_palette(["som", "Copepoda", "Calanus"])
        changed_order = make_class_palette(["Calanus", "som", "Copepoda"])
        self.assertEqual(full, changed_order)
        self.assertEqual(full["som"], self.colors["som"])
        self.assertEqual(full["Calanus"], self.colors["Calanus"])

    def test_colors_are_perceptually_separated_for_common_size_projects(self):
        colors = list(make_class_palette(["classe_{}".format(i) for i in range(20)]).values())
        self.assertEqual(len(set(colors)), 20)
        labs = _lab([to_rgb(c) for c in colors])
        diffs = np.linalg.norm(labs[:, None, :] - labs[None, :, :], axis=-1)
        np.fill_diagonal(diffs, np.inf)
        self.assertGreater(float(diffs.min()), 18.0)

    def test_individual_and_comparison_legends_have_correct_counts(self):
        one = dict(legend_lines([("Dia 1", self.a)], self.colors))
        self.assertEqual(one["Calanus"], "Calanus (1)")
        self.assertEqual(one["Copepoda"], "Copepoda (2)")
        pair = dict(legend_lines([("Dia 1", self.a), ("Dia 2", self.b)], self.colors))
        self.assertEqual(pair["Calanus"], "Calanus (A: 1 | B: 1)")
        self.assertEqual(pair["Copepoda"], "Copepoda (A: 2 | B: 0)")
        self.assertEqual(pair["som"], "som (A: 0 | B: 1)")

    def test_legend_is_a_separate_figure_axis_not_on_data(self):
        fig = Figure(figsize=(11, 6))
        axes = render_figure(fig, [("01/02", self.a), ("02/02", self.b)],
                             (-10, 20, -20, 20),
                             {"points": True, "density": False, "contours": False},
                             self.colors)
        self.assertEqual(len(axes), 2)
        self.assertEqual(len(fig.axes), 3)
        self.assertFalse(fig.axes[-1].axison)
        self.assertGreater(fig.axes[-1].get_position().x0, axes[-1].get_position().x1)
        texts = [t.get_text() for t in fig.axes[-1].texts]
        self.assertTrue(any("Calanus (A: 1 | B: 1)" == txt for txt in texts))
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "compare.png"
            fig.savefig(target, facecolor="#091828")
            self.assertGreater(target.stat().st_size, 5000)

    def test_gif_renders_frame_specific_legends(self):
        subset = self.data.copy()
        frames = [
            ("Dia A", pd.Series([True, False, True, True, False], index=subset.index)),
            ("Dia B", pd.Series([False, True, False, False, True], index=subset.index)),
        ]
        progress = []
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "anima.gif"
            count = save_animated_gif(
                target, subset, frames, (-10, 20, -20, 20),
                {"points": True, "density": False, "contours": False}, self.colors,
                duration_ms=250, progress=lambda i, n: progress.append((i, n)))
            self.assertEqual(count, 2)
            self.assertEqual(progress, [(1, 2), (2, 2)])
            with Image.open(target) as animation:
                self.assertEqual(animation.n_frames, 2)
                self.assertEqual(animation.info["duration"], 250)
            self.assertGreater(target.stat().st_size, 1500)


    def test_legends_fit_inside_figure_on_compact_comparison(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        names = [
            'Appendicularia', 'Chaetognatha', 'Copepoda', 'Detrito_associado',
            'Detrito_bolha', 'Detrito_concentrado', 'Detrito_filamentoso',
            'Doliolida', 'Fito', 'Organismos_gelatinosos', 'Penilia',
            'Podonidae',
        ]
        rows = pd.DataFrame({
            'classe': names,
            'x': np.arange(len(names), dtype=float),
            'y': np.arange(len(names), dtype=float),
        })
        palette = make_class_palette(names)
        for size in [(9, 5), (11, 4.5), (18, 10)]:
            with self.subTest(size=size):
                fig = Figure(figsize=size, dpi=100)
                render_figure(
                    fig, [('02/02/2026', rows), ('05/02/2026', rows)],
                    (-1, 14, -1, 14),
                    {'points': True, 'density': False, 'contours': False},
                    palette,
                )
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                renderer = canvas.get_renderer()
                legend = fig.axes[-1]
                legend_box = legend.get_window_extent(renderer=renderer)
                actual = [t.get_text() for t in legend.texts[1:]]
                self.assertEqual(len(actual), len(names))
                self.assertTrue(any('Detrito_concentrado' in t for t in actual))
                for item in legend.texts[1:]:
                    rectangle = item.get_window_extent(renderer=renderer)
                    self.assertLessEqual(rectangle.x1, legend_box.x1 + 2)
                    self.assertGreaterEqual(rectangle.x0, legend_box.x0 - 2)


if __name__ == "__main__":
    unittest.main()
