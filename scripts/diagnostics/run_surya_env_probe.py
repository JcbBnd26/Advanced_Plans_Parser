"""Run isolated Surya environment and dependency diagnostics.

This script writes a clean, text-only diagnostic report by executing each
dependency check in a fresh subprocess with an independent timeout. That keeps
one hanging import from obscuring the earlier results.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import List

DEFAULT_OUTPUT = Path("surya_env_probe.txt")
DEFAULT_TIMEOUT_SECONDS = 30


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated Surya environment and dependency diagnostics"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the diagnostic report",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-check timeout in seconds (default: 30)",
    )
    return parser


def _env_lines() -> List[str]:
    keys = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    lines = []
    for key in keys:
        value = os.environ.get(key, "<unset>")
        lines.append(f"{key}={value}")
    return lines


def _checks() -> List[tuple[str, str]]:
    return [
        (
            "python runtime",
            dedent(
                """
                import os
                import sys

                print(f"version={sys.version.splitlines()[0]}")
                print(f"executable={sys.executable}")
                print(f"platform={sys.platform}")
                print(f"os_name={os.name}")
                """
            ).strip(),
        ),
        (
            "find_spec torch",
            "import importlib.util; print(importlib.util.find_spec('torch'))",
        ),
        (
            "find_spec transformers",
            "import importlib.util; print(importlib.util.find_spec('transformers'))",
        ),
        (
            "find_spec cv2",
            "import importlib.util; print(importlib.util.find_spec('cv2'))",
        ),
        (
            "find_spec huggingface_hub",
            "import importlib.util; print(importlib.util.find_spec('huggingface_hub'))",
        ),
        (
            "find_spec surya",
            "import importlib.util; print(importlib.util.find_spec('surya'))",
        ),
        (
            "import numpy",
            "import numpy; print(numpy.__version__)",
        ),
        (
            "import pillow",
            "from PIL import Image; print(Image.__version__)",
        ),
        (
            "import torch",
            dedent(
                """
                import torch

                print(torch.__version__)
                print(f"cuda_available={torch.cuda.is_available()}")
                print(f"cuda_device_count={torch.cuda.device_count()}")
                """
            ).strip(),
        ),
        (
            "import transformers",
            "import transformers; print(transformers.__version__)",
        ),
        (
            "import cv2",
            "import cv2; print(cv2.__version__)",
        ),
        (
            "import huggingface_hub",
            "import huggingface_hub; print(huggingface_hub.__version__)",
        ),
        (
            "import surya",
            "import surya; print(getattr(surya, '__file__', '<no file>'))",
        ),
        (
            "import surya.detection",
            dedent(
                """
                import surya.detection

                print(getattr(surya.detection, '__file__', '<no file>'))
                print(sorted(name for name in dir(surya.detection) if name.endswith('Predictor')))
                """
            ).strip(),
        ),
        (
            "import surya.recognition",
            dedent(
                """
                import surya.recognition

                print(getattr(surya.recognition, '__file__', '<no file>'))
                print(sorted(name for name in dir(surya.recognition) if name.endswith('Predictor')))
                """
            ).strip(),
        ),
    ]


def _run_check(name: str, code: str, timeout: int) -> List[str]:
    result_lines = [f"== {name} =="]
    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result_lines.append(f"status=TIMEOUT ({timeout}s)")
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        if stdout:
            result_lines.append("stdout:")
            result_lines.extend(stdout.splitlines())
        if stderr:
            result_lines.append("stderr:")
            result_lines.extend(stderr.splitlines())
        result_lines.append("")
        return result_lines

    result_lines.append(f"status=EXIT {completed.returncode}")
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if stdout:
        result_lines.append("stdout:")
        result_lines.extend(stdout.splitlines())
    if stderr:
        result_lines.append("stderr:")
        result_lines.extend(stderr.splitlines())
    result_lines.append("")
    return result_lines


def main() -> int:
    args = _build_parser().parse_args()
    lines = [
        "Surya Environment Probe",
        f"python={sys.executable}",
        "",
        "Environment:",
        *_env_lines(),
        "",
    ]

    for name, code in _checks():
        lines.extend(_run_check(name, code, args.timeout))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Environment probe saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
