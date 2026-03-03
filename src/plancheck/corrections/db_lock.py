"""Cross-process file lock for SQLite single-writer enforcement.

This project supports running multiple processes (GUI + CLI/batch) against the
same SQLite DB. SQLite can handle multi-reader + single-writer, but write
contention can produce 'database is locked' errors unless writers are
coordinated.

We use an external lock file to serialize *writers* across processes.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class FileLockTimeout(TimeoutError):
    """Raised when a file lock cannot be acquired within the timeout."""


@contextmanager
def acquire_lock(
    lock_path: str | Path,
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.05,
) -> Iterator[None]:
    """Acquire an exclusive lock on *lock_path*.

    Uses a non-blocking lock attempt in a retry loop so we can implement a
    deterministic timeout and produce a clear exception.
    """

    lock_file_path = Path(lock_path)
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    with open(lock_file_path, "a+b") as f:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    # Lock 1 byte at the start of the file.
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if (time.monotonic() - start) >= timeout_sec:
                    raise FileLockTimeout(f"Timed out acquiring lock: {lock_file_path}")
                time.sleep(poll_interval_sec)

        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Best-effort unlock. If unlock fails due to an OS edge case,
                # the file descriptor close should still release the lock.
                pass
