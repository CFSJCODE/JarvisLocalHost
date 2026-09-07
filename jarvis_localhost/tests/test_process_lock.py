from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis_localhost.integrations.process_lock import InterProcessFileLock


class InterProcessFileLockTests(unittest.TestCase):
    def test_exclusive_lock_is_recoverable_without_deleting_lock_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.lock"
            first = InterProcessFileLock(path)
            second = InterProcessFileLock(path)

            self.assertTrue(first.acquire(blocking=False))
            self.assertTrue(first.locked)
            self.assertFalse(first.acquire(blocking=False))
            self.assertFalse(second.acquire(blocking=False))

            first.release()
            self.assertFalse(first.locked)
            self.assertTrue(path.is_file())
            self.assertTrue(second.acquire(blocking=False))
            second.release()

    def test_context_manager_releases_after_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resource.lock"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with InterProcessFileLock(path):
                    raise RuntimeError("boom")

            replacement = InterProcessFileLock(path)
            self.assertTrue(replacement.acquire(blocking=False))
            replacement.release()


if __name__ == "__main__":
    unittest.main()
