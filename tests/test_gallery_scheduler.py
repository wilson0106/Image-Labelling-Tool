
import unittest

from iat_gallery_mixin import IATGalleryMixin


class _Future:
    def __init__(self):
        self.callback = None

    def add_done_callback(self, callback):
        self.callback = callback


class _Executor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        future = _Future()
        self.submitted.append((fn, args, future))
        return future


class _GalleryHarness(IATGalleryMixin):
    def __init__(self):
        self._thumb_cache = {}
        self._thumb_waiters = {}
        self._thumb_pending = []
        self._thumb_pending_keys = set()
        self._thumb_inflight = {}
        self._thumb_generation = 1
        self._thumb_seq = 0
        self._thumb_max_inflight = 8
        self._thumb_cache_limit = 1600
        self._thumb_executor = _Executor()
        self._gallery_prefetch_after = None

    def _prepare_thumbnail(self, *args):
        return None


class GallerySchedulerTests(unittest.TestCase):
    def test_thumbnail_queue_has_bounded_inflight_work(self):
        h = _GalleryHarness()
        for idx in range(500):
            key = ("batch", f"img_{idx}.png", False, None)
            h._queue_thumbnail_request(
                key, "batch", f"img_{idx}.png", False, "otsu",
                priority=2, label=None
            )
        self.assertEqual(len(h._thumb_inflight), 8)
        self.assertEqual(len(h._thumb_executor.submitted), 8)
        self.assertEqual(len(h._thumb_pending), 492)

    def test_new_page_discards_not_started_thumbnail_work(self):
        h = _GalleryHarness()
        for idx in range(100):
            key = ("batch", f"img_{idx}.png", False, None)
            h._queue_thumbnail_request(
                key, "batch", f"img_{idx}.png", False, "otsu",
                priority=2, label=None
            )
        self.assertEqual(len(h._thumb_inflight), 8)
        self.assertEqual(len(h._thumb_pending), 92)
        old_generation = h._thumb_generation
        new_generation = h._gallery_begin_thumbnail_generation()
        self.assertEqual(new_generation, old_generation + 1)
        self.assertEqual(h._thumb_pending, [])
        self.assertEqual(h._thumb_pending_keys, set())
        # Already running work is allowed to finish; it is strictly bounded.
        self.assertEqual(len(h._thumb_inflight), 8)


if __name__ == "__main__":
    unittest.main()
