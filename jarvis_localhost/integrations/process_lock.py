"""Small crash-safe interprocess file lock for local Jarvis resources."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import BinaryIO


class InterProcessFileLock:
    """Hold an OS record lock without deleting the stable lock file.

    Windows uses ``msvcrt.locking`` and POSIX uses ``flock``. Both operating
    systems release the kernel lock automatically if the owning process exits,
    so a crash cannot leave a permanent textual "lock" behind.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve(strict=False)
        self._handle: BinaryIO | None = None
        self._guard = threading.Lock()

    @property
    def locked(self) -> bool:
        with self._guard:
            return self._handle is not None

    def acquire(self, *, blocking: bool = False) -> bool:
        with self._guard:
            if self._handle is not None:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+b")
            try:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                    os.fsync(handle.fileno())
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                    msvcrt.locking(handle.fileno(), mode, 1)
                else:
                    import fcntl

                    flags = fcntl.LOCK_EX
                    if not blocking:
                        flags |= fcntl.LOCK_NB
                    fcntl.flock(handle.fileno(), flags)
            except OSError:
                handle.close()
                return False
            self._handle = handle
            return True

    def release(self) -> None:
        with self._guard:
            handle = self._handle
            if handle is None:
                return
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle = None
                handle.close()

    def __enter__(self) -> "InterProcessFileLock":
        if not self.acquire(blocking=True):
            raise RuntimeError(f"could not acquire process lock: {self.path}")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


__all__ = ["InterProcessFileLock"]
