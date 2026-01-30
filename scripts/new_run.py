import argparse
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a new run directory with timestamp"
    )
    parser.add_argument(
        "name", nargs="?", default=None, help="Optional name suffix for the run"
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        run_name = f"run_{stamp}_{args.name}"
    else:
        run_name = f"run_{stamp}"

    base = Path("runs")
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Subfolders for artifacts.
    for sub in ["inputs", "artifacts", "overlays", "exports", "logs"]:
        (run_dir / sub).mkdir(exist_ok=True)

    print(run_dir)


if __name__ == "__main__":
    main()
