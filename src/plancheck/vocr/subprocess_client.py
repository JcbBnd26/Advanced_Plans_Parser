"""PaddleOCR subprocess client — manages a persistent OCR worker process.

This module provides a thread-safe client that communicates with the
subprocess worker. The worker runs PaddleOCR on its main thread, avoiding
the threading issues that cause hangs when PaddleOCR is used from a
background thread.

Public API
----------
VOCRSubprocessClient  — Singleton client managing the worker process
get_vocr_client       — Get or create the singleton client
shutdown_vocr_client  — Gracefully shut down the worker
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from ..config import GroupingConfig

log = logging.getLogger(__name__)

# Singleton instance
_client: VOCRSubprocessClient | None = None
_client_lock = threading.Lock()


class VOCRSubprocessClient:
    """Client for the PaddleOCR subprocess worker.

    Thread-safe: multiple threads can call ocr_image() concurrently;
    requests are serialized via an internal lock.
    """

    # Max stderr lines to keep in buffer (circular)
    _STDERR_BUFFER_SIZE = 100

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._initialized = False
        # Stderr drain thread and buffer
        self._stderr_thread: threading.Thread | None = None
        self._stderr_stop = threading.Event()
        self._stderr_buffer: list[str] = []

    def _start_process(self) -> None:
        """Start the worker subprocess."""
        if self._process is not None and self._process.poll() is None:
            return  # Already running

        log.info("Starting VOCR subprocess worker...")

        # Find the Python executable in the same venv
        python_exe = sys.executable

        # Subprocess creation flags for Windows (hide console)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NO_WINDOW

        # Set up environment with src in PYTHONPATH
        env = os.environ.copy()
        src_dir = str(Path(__file__).parent.parent.parent)  # src/
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = src_dir + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = src_dir
        # Disable Paddle connectivity check
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        self._process = subprocess.Popen(
            [python_exe, "-m", "plancheck.vocr.subprocess_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
            creationflags=creationflags,
            env=env,
        )
        self._initialized = False
        log.info("VOCR subprocess started (PID=%d)", self._process.pid)

        # Start stderr drain thread to prevent buffer deadlock
        self._stderr_stop.clear()
        self._stderr_buffer.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="VOCR-stderr-drain",
        )
        self._stderr_thread.start()

    def _ensure_running(self) -> None:
        """Ensure the worker process is running, restart if crashed."""
        if self._process is None or self._process.poll() is not None:
            if self._process is not None:
                log.warning(
                    "VOCR subprocess died (exit=%s), restarting...",
                    self._process.returncode,
                )
                # Stop old stderr thread if any
                self._stop_stderr_thread()
            self._start_process()

    def _drain_stderr(self) -> None:
        """Background thread that drains stderr to prevent buffer deadlock."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            for line in self._process.stderr:
                if self._stderr_stop.is_set():
                    break
                line = line.rstrip()
                # Add to circular buffer
                self._stderr_buffer.append(line)
                if len(self._stderr_buffer) > self._STDERR_BUFFER_SIZE:
                    self._stderr_buffer.pop(0)
                # Log important messages
                if "error" in line.lower() or "exception" in line.lower():
                    log.warning("[VOCR-Worker] %s", line)
        except Exception:  # noqa: BLE001 — drain thread must not crash
            pass

    def _stop_stderr_thread(self) -> None:
        """Stop the stderr drain thread."""
        self._stderr_stop.set()
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)
        self._stderr_thread = None

    def get_stderr_output(self) -> list[str]:
        """Get recent stderr output from the worker (for debugging)."""
        return list(self._stderr_buffer)

    def _send_request(self, request: dict) -> dict:
        """Send a request and wait for response."""
        self._ensure_running()

        if self._process is None or self._process.stdin is None:
            raise RuntimeError("VOCR subprocess not available")

        try:
            # Send request
            line = json.dumps(request, ensure_ascii=False) + "\n"
            self._process.stdin.write(line)
            self._process.stdin.flush()

            # Read response
            if self._process.stdout is None:
                raise RuntimeError("VOCR subprocess stdout not available")

            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("VOCR subprocess closed connection")

            return json.loads(response_line)

        except (BrokenPipeError, OSError) as e:
            log.error("VOCR subprocess communication error: %s", e)
            self.shutdown()
            raise RuntimeError(f"VOCR subprocess communication failed: {e}") from e

    def initialize(self, cfg: "GroupingConfig | None" = None) -> None:
        """Pre-initialize the OCR engine in the subprocess."""
        with self._lock:
            self._ensure_running()

            cfg_dict = self._config_to_dict(cfg) if cfg else {}
            response = self._send_request({"cmd": "init", "cfg": cfg_dict})

            if response.get("status") != "ok":
                raise RuntimeError(f"VOCR init failed: {response.get('error')}")

            self._initialized = True
            log.info("VOCR subprocess initialized")

    def ocr_image(
        self,
        image: Image.Image,
        page_num: int,
        cfg: "GroupingConfig | None" = None,
    ) -> tuple[list[dict], list[float]]:
        """Run OCR on an image via the subprocess.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to OCR (will be converted to RGB if needed).
        page_num : int
            Page number for token metadata.
        cfg : GroupingConfig, optional
            Configuration settings.

        Returns
        -------
        tokens : list[dict]
            List of token dicts with keys: page, x0, y0, x1, y1, text, confidence
        confs : list[float]
            Parallel list of confidence scores.
        """
        with self._lock:
            # Encode image to base64 PNG
            if image.mode != "RGB":
                image = image.convert("RGB")

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

            cfg_dict = self._config_to_dict(cfg) if cfg else {}

            response = self._send_request(
                {
                    "cmd": "ocr",
                    "image_b64": image_b64,
                    "page_num": page_num,
                    "cfg": cfg_dict,
                }
            )

            if response.get("status") != "ok":
                raise RuntimeError(f"VOCR OCR failed: {response.get('error')}")

            return response.get("tokens", []), response.get("confs", [])

    def ping(self) -> bool:
        """Check if the worker is responsive."""
        with self._lock:
            try:
                self._ensure_running()
                response = self._send_request({"cmd": "ping"})
                return response.get("status") == "ok"
            except Exception:
                return False

    def shutdown(self) -> None:
        """Gracefully shut down the worker subprocess."""
        with self._lock:
            if self._process is None:
                return

            # Stop stderr thread first
            self._stop_stderr_thread()

            try:
                if self._process.poll() is None:
                    # Send shutdown command
                    try:
                        line = json.dumps({"cmd": "shutdown"}) + "\n"
                        if self._process.stdin:
                            self._process.stdin.write(line)
                            self._process.stdin.flush()
                    except Exception:
                        pass

                    # Wait a bit for graceful exit
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        log.warning("VOCR subprocess didn't exit gracefully, killing...")
                        self._process.kill()
                        self._process.wait(timeout=1)

                log.info("VOCR subprocess shut down")

            except Exception as e:
                log.error("Error shutting down VOCR subprocess: %s", e)

            finally:
                self._process = None
                self._initialized = False

    @staticmethod
    def _config_to_dict(cfg: "GroupingConfig | None") -> dict[str, Any]:
        """Extract VOCR-relevant config fields to a dict."""
        if cfg is None:
            return {}

        return {
            "vocr_model_tier": cfg.vocr_model_tier,
            "vocr_device": getattr(cfg, "vocr_device", "cpu"),
            "vocr_use_orientation_classify": cfg.vocr_use_orientation_classify,
            "vocr_use_doc_unwarping": cfg.vocr_use_doc_unwarping,
            "vocr_use_textline_orientation": cfg.vocr_use_textline_orientation,
            "vocr_min_confidence": cfg.vocr_min_confidence,
            "vocr_max_tile_px": cfg.vocr_max_tile_px,
            "vocr_tile_overlap": cfg.vocr_tile_overlap,
            "vocr_tile_dedup_iou": cfg.vocr_tile_dedup_iou,
            "vocr_min_text_length": cfg.vocr_min_text_length,
            "vocr_strip_whitespace": cfg.vocr_strip_whitespace,
        }

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass


def get_vocr_client() -> VOCRSubprocessClient:
    """Get or create the singleton VOCR subprocess client."""
    global _client
    with _client_lock:
        if _client is None:
            _client = VOCRSubprocessClient()
        return _client


def shutdown_vocr_client() -> None:
    """Shut down the singleton VOCR subprocess client."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.shutdown()
            _client = None
