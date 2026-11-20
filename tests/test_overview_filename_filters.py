"""Regression tests: Overview border control and image-filename visual filters."""
import pathlib
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from iat_filename_filters import ImageNameFilter, filter_image_names


class FilenameVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.names = pd.Series([
            "stationB_20261022_083015_som.png",
            "stationB_2026-10-22_14-30-55_copepod.jpg",
            "stationA_22-10-2026_090530_som.png",
            "stationB_20261022231500_som.png",
            r"C:\\samples\\stationA_20261104_143015.png",
            "stationA_sem_data.jpg",
        ])

    def test_date_range_in_multiple_formats_and_unparseable_names(self):
        mask = filter_image_names(self.names, ImageNameFilter(date_from="2026-10-22", date_to="2026-10-22"))
        self.assertEqual(mask.tolist(), [True, True, True, True, False, False])

    def test_date_and_time_are_combined(self):
        mask = filter_image_names(self.names, ImageNameFilter(
            date_from="2026-10-22", date_to="2026-10-22",
            time_from="08:00", time_to="10:00"))
        self.assertEqual(mask.tolist(), [True, False, True, False, False, False])

    def test_midnight_crossing(self):
        mask = filter_image_names(self.names, ImageNameFilter(time_from="22:00", time_to="09:00"))
        self.assertEqual(mask.tolist(), [True, False, False, True, False, False])

    def test_attribute_regex_and_plain_name_text(self):
        selected = filter_image_names(self.names, ImageNameFilter(
            attribute_regex=r"(station[A-Z])_", attribute_value="stationB"))
        self.assertEqual(selected.tolist(), [True, True, False, True, False, False])
        plain = filter_image_names(self.names, ImageNameFilter(attribute_value="som"))
        self.assertEqual(plain.tolist(), [True, False, True, True, False, False])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            ImageNameFilter(date_from="2026-13-01").validate()
        with self.assertRaises(ValueError):
            ImageNameFilter(date_from="2026-10-25", date_to="2026-10-20").validate()
        with self.assertRaises(ValueError):
            ImageNameFilter(attribute_regex=r"station[A-Z]", attribute_value="stationB").validate()

    def test_default_is_no_filter(self):
        self.assertEqual(filter_image_names(self.names, ImageNameFilter()).tolist(), [True]*6)

    def test_depth_range_inclusive_and_missing_value_excluded(self):
        names = pd.Series([
            "2026-02-02_11_47_11.839_+0000999_x1702_y244.png",
            "2026-02-02_11_47_11.839_+0001000_x1702_y244.png",
            "2026-02-02_11_47_11.839_+0001537_x1702_y244.png",
            "2026-02-02_11_47_11.839_+0002000_x1702_y244.png",
            "2026-02-02_11_47_11.839_+0002001_x1702_y244.png",
            "2026-02-02_11_47_11.839_x1702_y244.png",
        ])
        mask = filter_image_names(names, ImageNameFilter(depth_min="1000", depth_max="2000"))
        self.assertEqual(mask.tolist(), [False, True, True, True, False, False])

    def test_depth_combines_with_date_time_and_attribute(self):
        names = pd.Series([
            "estacaoB_2026-02-02_11_47_11.839_+0001537_x1702_y244.png",
            "estacaoA_2026-02-02_11_47_11.839_+0001537_x1702_y244.png",
            "estacaoB_2026-02-02_14_47_11.839_+0001537_x1702_y244.png",
            "estacaoB_2026-02-03_11_47_11.839_+0001537_x1702_y244.png",
            "estacaoB_2026-02-02_11_47_11.839_+0002500_x1702_y244.png",
        ])
        mask = filter_image_names(names, ImageNameFilter(
            date_from="2026-02-02", date_to="2026-02-02",
            time_from="11:00", time_to="12:00",
            depth_min="1500", depth_max="1600",
            attribute_regex=r"(?P<valor>estacao[A-Z])_", attribute_value="estacaoB",
        ))
        self.assertEqual(mask.tolist(), [True, False, False, False, False])

    def test_depth_one_sided_limits_and_comma(self):
        names = pd.Series([
            "_+0001537_x1.png", "_+0001600_x2.png",
            "_+0002000_x3.png", "_+0000800_x4.png",
        ])
        self.assertEqual(filter_image_names(
            names, ImageNameFilter(depth_min="1600")).tolist(),
            [False, True, True, False])
        self.assertEqual(filter_image_names(
            names, ImageNameFilter(depth_max="1537,0")).tolist(),
            [True, False, False, True])
        self.assertEqual(filter_image_names(
            names, ImageNameFilter(depth_min="1600", depth_max="")).tolist(),
            [False, True, True, False])

    def test_depth_invalid_ranges(self):
        for kwargs in ({"depth_min":"abc"},
                       {"depth_min":"1001", "depth_max":"1000"},
                       {"depth_max":"NaN"},
                       {"depth_min":"Infinity"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ImageNameFilter(**kwargs).validate()

    def test_depth_filter_not_used_by_default(self):
        names = pd.Series(["a_+001000_x1.png", "sem_profundidade.png"])
        self.assertEqual(filter_image_names(names, ImageNameFilter()).tolist(), [True, True])
        mask = filter_image_names(names, ImageNameFilter(depth_min="0"))
        self.assertEqual(mask.tolist(), [True, False])

    def test_original_names_are_untouched(self):
        before=self.names.copy(deep=True)
        filter_image_names(self.names, ImageNameFilter(time_from="08:00"))
        pd.testing.assert_series_equal(before,self.names)


if __name__ == "__main__":
    unittest.main()
