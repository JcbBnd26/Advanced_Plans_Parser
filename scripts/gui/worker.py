"""Background pipeline worker for the Advanced Plan Parser GUI.

Runs pipeline operations in a background thread and communicates
results back to the tkinter main thread via ``queue.Queue`` + ``root.after()``.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Logging bridge – captures logging output and forwards to the GUI
# ---------------------------------------------------------------------------


class QueueHandler(logging.Handler):
    """Logging handler that posts records to a ``queue.Queue``."""

    def __init__(self, q: queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            level = record.levelname  # DEBUG/INFO/WARNING/ERROR
            self.q.put(("log", msg, level))
        except Exception:
            self.handleError(record)


class StdoutCapture:
    """File-like object that captures print() output and posts to a queue."""

    def __init__(self, q: queue.Queue, original: Any, level: str = "INFO") -> None:
        self.q = q
        self.original = original
        self.level = level

    def write(self, text: str) -> None:
        if text and text.strip():
            self.q.put(("log", text.rstrip("\n"), self.level))
        # Also write to original so terminal still works
        if self.original:
            self.original.write(text)

    def flush(self) -> None:
        if self.original:
            self.original.flush()


# ---------------------------------------------------------------------------
# PipelineWorker
# ---------------------------------------------------------------------------


class PipelineWorker:
    """Manages background execution of pipeline operations.

    Usage::

        worker = PipelineWorker(root, log_panel, stage_bar)
        worker.run(callable, args, kwargs, on_done=callback)

    The *callable* runs in a daemon thread.  ``print()`` and ``logging``
    output is redirected to *log_panel*.  *stage_bar* is updated when
    ``("stage", name, status)`` messages are posted to the internal queue.
    """

    def __init__(
        self,
        root: Any,
        log_panel: Any | None = None,
        stage_bar: Any | None = None,
    ) -> None:
        self.root = root
        self.log_panel = log_panel
        self.stage_bar = stage_bar
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._polling = False
        self._on_done: Callable | None = None
        self._original_stdout = None
        self._original_stderr = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    def run(
        self,
        target: Callable,
        args: tuple = (),
        kwargs: dict | None = None,
        on_done: Callable | None = None,
    ) -> None:
        """Start *target* in a background thread."""
        if self.is_running:
            return
        self._cancel_event.clear()
        self._on_done = on_done
        kwargs = kwargs or {}

        def wrapper():
            # Capture stdout/stderr
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = StdoutCapture(self._queue, self._original_stdout, "INFO")
            sys.stderr = StdoutCapture(self._queue, self._original_stderr, "ERROR")

            # Add queue handler to root logger
            handler = QueueHandler(self._queue)
            handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
            logging.root.addHandler(handler)
            old_level = logging.root.level
            if logging.root.level > logging.INFO:
                logging.root.setLevel(logging.INFO)

            t0 = time.perf_counter()
            error = None
            result = None
            try:
                result = target(*args, **kwargs)
            except Exception as exc:
                error = exc
                self._queue.put(("log", f"ERROR: {exc}", "ERROR"))
            finally:
                elapsed = time.perf_counter() - t0
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
                logging.root.removeHandler(handler)
                logging.root.setLevel(old_level)
                self._queue.put(("done", result, error, elapsed))

        self._thread = threading.Thread(target=wrapper, daemon=True)
        self._thread.start()
        self._start_polling()

    def cancel(self) -> None:
        """Signal the worker to cancel (cooperative – target must check)."""
        self._cancel_event.set()
        if self.log_panel:
            self.log_panel.write("Cancellation requested...", "WARNING")

    def _start_polling(self) -> None:
        """Poll the message queue from the main thread."""
        if self._polling:
            return
        self._polling = True
        self._poll()

    def _poll(self) -> None:
        """Process queued messages and reschedule."""
        try:
            while True:
                msg = self._queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass

        if self.is_running:
            self.root.after(50, self._poll)
        else:
            # Drain remaining messages
            try:
                while True:
                    msg = self._queue.get_nowait()
                    self._handle_message(msg)
            except queue.Empty:
                pass
            self._polling = False

    def _handle_message(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "log":
            _, text, level = msg
            if self.log_panel:
                self.log_panel.write(text, level)
        elif kind == "stage":
            _, stage_name, status = msg
            if self.stage_bar:
                self.stage_bar.set_stage(stage_name, status)
            if self.log_panel and status == "running":
                self.log_panel.write(f"▸ {stage_name}...", "STAGE")
        elif kind == "done":
            _, result, error, elapsed = msg
            if self.log_panel:
                if error:
                    self.log_panel.write(
                        f"\nPipeline failed after {elapsed:.1f}s: {error}", "ERROR"
                    )
                else:
                    self.log_panel.write(
                        f"\nPipeline finished in {elapsed:.1f}s", "SUCCESS"
                    )
            if self._on_done:
                self._on_done(result, error, elapsed)

    def post_stage(self, stage: str, status: str) -> None:
        """Post a stage update (call from within the worker target)."""
        self._queue.put(("stage", stage, status))

    def post_log(self, text: str, level: str = "INFO") -> None:
        """Post a log message (call from within the worker target)."""
        self._queue.put(("log", text, level))
