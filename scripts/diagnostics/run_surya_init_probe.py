"""Run a clean Surya initialization probe and capture only Python output.

This diagnostic avoids shell transcript noise such as PowerShell prompt lines
and ``NativeCommandError`` wrappers by spawning a child Python process and
capturing its merged stdout/stderr directly.

Usage:
    python -m scripts.diagnostics.run_surya_init_probe
    python -m scripts.diagnostics.run_surya_init_probe --device gpu
    python -m scripts.diagnostics.run_surya_init_probe --output logs/surya_probe.txt
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

DEFAULT_OUTPUT = Path("surya_test_output.txt")
_CHILD_FLAG = "--child-probe"
DEFAULT_TIMEOUT_SECONDS = 300
HEARTBEAT_SECONDS = 15


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a clean Surya initialization probe"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run Surya on (default: cpu)",
    )
    parser.add_argument(
        "--mode",
        choices=["backend", "config", "raw"],
        default="backend",
        help="Initialization path to exercise (default: backend)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write captured output (default: surya_test_output.txt)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout in seconds for the child probe (default: 300)",
    )
    parser.add_argument(
        "--trace-imports",
        action="store_true",
        help="Enable Python import timing output for the child probe",
    )
    parser.add_argument(
        _CHILD_FLAG,
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def _set_probe_environment() -> None:
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _emit_step(t0: float, message: str) -> None:
    print(f"[{time.perf_counter() - t0:7.1f}s] {message}", flush=True)


def _probe_raw_surya(t0: float, device: str) -> None:
    _emit_step(t0, "step 1: importing surya.detection")
    from surya.detection import DetectionPredictor

    _emit_step(t0, "step 1 ok: imported surya.detection")
    _emit_step(t0, "step 2: importing surya.recognition")
    from surya.recognition import FoundationPredictor, RecognitionPredictor

    _emit_step(t0, "step 2 ok: imported surya.recognition")

    device_str = "cuda" if device == "gpu" else "cpu"

    _emit_step(t0, "step 3: constructing DetectionPredictor")
    _det_predictor = DetectionPredictor(device=device_str)
    _emit_step(t0, "step 3 ok: DetectionPredictor ready")

    _emit_step(t0, "step 4: constructing FoundationPredictor")
    foundation = FoundationPredictor(device=device_str)
    _emit_step(t0, "step 4 ok: FoundationPredictor ready")

    _emit_step(t0, "step 5: constructing RecognitionPredictor")
    _rec_predictor = RecognitionPredictor(foundation)
    _emit_step(t0, "step 5 ok: RecognitionPredictor ready")


def _probe_backend_surya(t0: float, device: str) -> None:
    _emit_step(t0, "step 1: importing plancheck.config")
    from plancheck.config import GroupingConfig

    _emit_step(t0, "step 1 ok: imported plancheck.config")
    _emit_step(t0, "step 2: importing plancheck.vocr.backends.surya")
    from plancheck.vocr.backends.surya import SuryaOCRBackend

    _emit_step(t0, "step 2 ok: imported backend module")

    _emit_step(t0, "step 3: building GroupingConfig")
    cfg = GroupingConfig(vocr_backend="surya", vocr_device=device)
    _emit_step(t0, "step 3 ok: config ready")

    _emit_step(t0, "step 4: constructing SuryaOCRBackend")
    backend = SuryaOCRBackend(device=device, cfg=cfg)
    _emit_step(t0, "step 4 ok: backend object ready")

    _emit_step(t0, "step 5: calling backend._ensure_initialized()")
    backend._ensure_initialized()
    _emit_step(t0, "step 5 ok: backend initialized")


def _probe_config_imports(t0: float) -> None:
    _emit_step(t0, "step 1: importing plancheck.config.constants")
    from plancheck.config import constants as _constants  # noqa: F401

    _emit_step(t0, "step 1 ok: imported constants")
    _emit_step(t0, "step 2: importing plancheck.config.exceptions")
    from plancheck.config import exceptions as _exceptions  # noqa: F401

    _emit_step(t0, "step 2 ok: imported exceptions")
    _emit_step(t0, "step 3: importing plancheck.config.subconfigs")
    from plancheck.config import subconfigs as _subconfigs  # noqa: F401

    _emit_step(t0, "step 3 ok: imported subconfigs")
    _emit_step(t0, "step 4: importing plancheck.config.validation")
    from plancheck.config import validation as _validation  # noqa: F401

    _emit_step(t0, "step 4 ok: imported validation")
    _emit_step(t0, "step 5: importing plancheck.config.pipeline")
    from plancheck.config import pipeline as _pipeline  # noqa: F401

    _emit_step(t0, "step 5 ok: imported pipeline")


def _run_child_probe(device: str, mode: str) -> int:
    t0 = time.perf_counter()

    try:
        _set_probe_environment()
        _emit_step(t0, "env set")
        _emit_step(t0, f"probe mode={mode} device={device}")

        if mode == "backend":
            _probe_backend_surya(t0, device)
        elif mode == "config":
            _probe_config_imports(t0)
        else:
            _probe_raw_surya(t0, device)

        _emit_step(t0, "probe ok")
        return 0
    except Exception:
        traceback.print_exc()
        return 1


def _build_child_command(
    script_path: Path,
    device: str,
    mode: str,
    trace_imports: bool,
) -> List[str]:
    command = [sys.executable]
    if trace_imports:
        command.extend(["-X", "importtime"])
    command.extend(
        [
            "-u",
            str(script_path),
            _CHILD_FLAG,
            "--device",
            device,
            "--mode",
            mode,
        ]
    )
    return command


def _stream_child_output(
    proc: subprocess.Popen[str], output_file
) -> queue.Queue[str | None]:
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_queue.put(line)
        output_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    return output_queue


def _run_parent_probe(
    device: str,
    mode: str,
    output_path: Path,
    timeout_seconds: int,
    trace_imports: bool,
) -> int:
    script_path = Path(__file__).resolve()
    command = _build_child_command(script_path, device, mode, trace_imports)
    start_time = time.monotonic()
    next_heartbeat = start_time + HEARTBEAT_SECONDS

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        output_queue = _stream_child_output(proc, output_file)
        stream_closed = False

        while True:
            try:
                item = output_queue.get(timeout=1)
                if item is None:
                    stream_closed = True
                else:
                    output_file.write(item)
                    output_file.flush()
            except queue.Empty:
                pass

            now = time.monotonic()

            if proc.poll() is None and now >= next_heartbeat:
                elapsed = int(now - start_time)
                output_file.write(f"[parent] waiting... {elapsed}s elapsed\n")
                output_file.flush()
                next_heartbeat += HEARTBEAT_SECONDS

            if proc.poll() is not None and stream_closed and output_queue.empty():
                break

            if proc.poll() is None and now - start_time >= timeout_seconds:
                proc.kill()
                proc.wait()
                output_file.write(
                    f"TIMEOUT: child probe exceeded {timeout_seconds} seconds\n"
                )
                output_file.flush()
                print(f"Probe timed out. Output saved to {output_path}")
                return 124

    print(f"Probe exit code: {proc.returncode}")
    print(f"Output saved to: {output_path}")
    return proc.returncode


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.child_probe:
        return _run_child_probe(args.device, args.mode)

    return _run_parent_probe(
        args.device,
        args.mode,
        args.output,
        args.timeout,
        args.trace_imports,
    )


if __name__ == "__main__":
    raise SystemExit(main())
