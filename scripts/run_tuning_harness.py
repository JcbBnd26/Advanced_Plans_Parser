"""Auto-tuning harness for TOCR knobs.

Feeds knob values into the pipeline, captures manifest diagnostics,
computes a quality score, and reports results.  Designed to be driven
by a human, an AI agent, or an automated optimizer.

Usage
-----
Interactive (manual knob values)::

    python scripts/run_tuning_harness.py --pdf input/MyPlan.pdf --page 3

From a JSON recipe (batch of configs to try)::

    python scripts/run_tuning_harness.py --pdf input/MyPlan.pdf --page 3 \\
        --recipe recipes/odot_plans.json

Recipe format::

    [
      {"label": "baseline", "knobs": {}},
      {"label": "tight margins", "knobs": {"tocr_margin_pts": 36}},
      {"label": "no rotated",   "knobs": {"tocr_keep_rotated": false, "tocr_margin_pts": 36}}
    ]

Any knob not listed in "knobs" keeps its default value.

Output
------
* One run directory per config trial
* A ``scripts/tuning_reports/tuning_<timestamp>.json`` comparison report
* Console summary table

Recipe management
-----------------
Save the winning knobs as a named recipe::

    python scripts/run_tuning_harness.py --pdf input/MyPlan.pdf --page 2 \\
        --use-defaults --save-winner odot_plans

List saved recipes::

    python scripts/run_tuning_harness.py --list-recipes

Re-use a saved recipe::

    python scripts/run_tuning_harness.py --pdf input/Other.pdf --page 0 \\
        --recipe recipes/odot_plans.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_pdf_batch import cleanup_old_runs, run_pdf  # noqa: E402

from plancheck.config import GroupingConfig  # noqa: E402

# ── Directories for harness output ─────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = Path(__file__).parent
RECIPES_DIR = _PROJECT_ROOT / "recipes"
REPORTS_DIR = _SCRIPTS_DIR / "tuning_reports"

# ── All TOCR knobs and their types (for validation) ────────────────────

TOCR_KNOBS: dict[str, type] = {
    "tocr_x_tolerance": float,
    "tocr_y_tolerance": float,
    "tocr_extra_attrs": bool,
    "tocr_filter_control_chars": bool,
    "tocr_dedup_iou": float,
    "tocr_min_word_length": int,
    "tocr_min_font_size": float,
    "tocr_max_font_size": float,
    "tocr_strip_whitespace_tokens": bool,
    "tocr_clip_to_page": bool,
    "tocr_margin_pts": float,
    "tocr_keep_rotated": bool,
    "tocr_normalize_unicode": bool,
    "tocr_case_fold": bool,
    "tocr_collapse_whitespace": bool,
    "tocr_min_token_density": float,
    "tocr_mojibake_threshold": float,
    "tocr_use_text_flow": bool,
    "tocr_keep_blank_chars": bool,
}

# ── All VOCR knobs and their types (for validation) ────────────────────

VOCR_KNOBS: dict[str, type] = {
    "vocr_model_tier": str,
    "vocr_use_orientation_classify": bool,
    "vocr_use_doc_unwarping": bool,
    "vocr_use_textline_orientation": bool,
    "vocr_resolution": int,
    "vocr_min_confidence": float,
    "vocr_max_tile_px": int,
    "vocr_tile_overlap": float,
    "vocr_tile_dedup_iou": float,
    "vocr_min_text_length": int,
    "vocr_strip_whitespace": bool,
    "vocr_max_det_skew": float,
    "vocr_heartbeat_interval": float,
}

# Combined dict for universal knob validation.
ALL_KNOBS: dict[str, type] = {**TOCR_KNOBS, **VOCR_KNOBS}

# ── VOCRPP knobs (OCR image preprocessing) ─────────────────────────────

VOCRPP_KNOBS: dict[str, type] = {
    "vocrpp_grayscale": bool,
    "vocrpp_autocontrast": bool,
    "vocrpp_clahe": bool,
    "vocrpp_clahe_clip_limit": float,
    "vocrpp_clahe_grid_size": int,
    "vocrpp_median_denoise": bool,
    "vocrpp_median_kernel": int,
    "vocrpp_adaptive_binarize": bool,
    "vocrpp_binarize_block_size": int,
    "vocrpp_binarize_constant": float,
    "vocrpp_sharpen": bool,
    "vocrpp_sharpen_radius": int,
    "vocrpp_sharpen_percent": int,
}

# ── Reconcile knobs ────────────────────────────────────────────────────

RECONCILE_KNOBS: dict[str, type] = {
    "ocr_reconcile_allowed_symbols": str,
    "ocr_reconcile_resolution": int,
    "ocr_reconcile_confidence": float,
    "ocr_reconcile_iou_threshold": float,
    "ocr_reconcile_center_tol_x": float,
    "ocr_reconcile_center_tol_y": float,
    "ocr_reconcile_proximity_pts": float,
    "ocr_reconcile_anchor_margin": float,
    "ocr_reconcile_symbol_pad": float,
    "ocr_reconcile_debug": bool,
    "ocr_reconcile_digit_band_tol_mult": float,
    "ocr_reconcile_digit_overshoot": float,
    "ocr_reconcile_char_width_fallback": float,
    "ocr_reconcile_line_neighbour_tol_mult": float,
    "ocr_reconcile_line_neighbour_min_tol": float,
    "ocr_reconcile_digit_ratio": float,
    "ocr_reconcile_slash_width_mult": float,
    "ocr_reconcile_pct_width_mult": float,
    "ocr_reconcile_degree_width_mult": float,
    "ocr_reconcile_accept_proximity": float,
    "ocr_reconcile_accept_iou": float,
    "ocr_reconcile_accept_coverage": float,
    "ocr_reconcile_max_debug": int,
}

# ── Grouping knobs ─────────────────────────────────────────────────────

GROUPING_KNOBS: dict[str, type] = {
    "grouping_histogram_density": float,
    "grouping_histogram_bins": int,
    "grouping_line_overlap_ratio": float,
    "grouping_space_gap_fallback": float,
    "grouping_space_gap_percentile": float,
    "grouping_partition_width_guard_mult": float,
    "grouping_partition_decay": float,
    "grouping_partition_floor": float,
    "grouping_note_majority": float,
    "grouping_note_max_rows": int,
    "grouping_col_gap_fallback_mult": float,
    "grouping_block_merge_mult": float,
    "grouping_notes_x_tolerance": float,
    "grouping_notes_y_gap_max": float,
    "grouping_notes_first_gap_mult": float,
    "grouping_link_x_tolerance": float,
}

# ── Legend / abbreviation / revision knobs ─────────────────────────────

LEGEND_KNOBS: dict[str, type] = {
    "legend_enclosure_tolerance": float,
    "legend_max_symbol_size": float,
    "legend_symbol_min_area": float,
    "legend_symbol_max_area": float,
    "legend_column_x_tolerance": float,
    "legend_text_y_tolerance": float,
    "legend_text_x_gap_max": float,
    "legend_unboxed_x_margin": float,
    "legend_unboxed_x_extent": float,
    "legend_unboxed_y_extent": float,
}

# ── Font metrics knobs ─────────────────────────────────────────────────

FONT_METRICS_KNOBS: dict[str, type] = {
    "font_metrics_inflation_threshold": float,
    "font_metrics_min_samples": int,
    "font_metrics_confidence_min": float,
    "font_metrics_visual_dpi": int,
    "font_metrics_dark_threshold": int,
}

# ── Overlay / debug visualisation knobs ────────────────────────────────

OVERLAY_KNOBS: dict[str, type] = {
    "overlay_label_font_base": int,
    "overlay_label_font_floor": int,
    "overlay_label_bg_alpha": int,
    "overlay_table_fill_alpha": int,
    "overlay_same_line_overlap": float,
    "overlay_proximity_pts": float,
}

# ── Preprocessing knobs ────────────────────────────────────────────────

PREPROCESS_KNOBS: dict[str, type] = {
    "preprocess_min_rotation": float,
}

# Update ALL_KNOBS with every category
ALL_KNOBS.update(VOCRPP_KNOBS)
ALL_KNOBS.update(RECONCILE_KNOBS)
ALL_KNOBS.update(GROUPING_KNOBS)
ALL_KNOBS.update(LEGEND_KNOBS)
ALL_KNOBS.update(FONT_METRICS_KNOBS)
ALL_KNOBS.update(OVERLAY_KNOBS)
ALL_KNOBS.update(PREPROCESS_KNOBS)


def _quality_score(page_result: dict) -> dict:
    """Compute a quality score from a single page's manifest data.

    Returns a dict with the breakdown and a final ``score`` number.
    Higher is better.

    The formula is intentionally simple so you can read and tweak it.
    An AI agent can propose a better formula later.
    """
    counts = page_result.get("counts", {})
    stages = page_result.get("stages", {})
    tocr = stages.get("tocr", {})
    tocr_counts = tocr.get("counts", {})

    tokens_total = tocr_counts.get("tokens_total", 0)
    tokens_raw = tocr_counts.get("tokens_raw", 0)
    degenerate = tocr_counts.get("tokens_degenerate_skipped", 0)
    encoding_issues = tocr_counts.get("char_encoding_issues", 0)
    duplicates = tocr_counts.get("tokens_duplicate_removed", 0)
    density = tocr_counts.get("token_density_per_sqin", 0.0)
    has_error = tocr.get("status") == "failed"

    # Downstream quality signals
    blocks = counts.get("blocks", 0)
    tables = counts.get("tables", 0)
    notes_columns = counts.get("notes_columns", 0)
    legend_regions = counts.get("legend_regions", 0)
    abbreviation_regions = counts.get("abbreviation_regions", 0)

    # Penalties (subtracted)
    penalty_degenerate = degenerate * 2
    penalty_encoding = encoding_issues * 5
    penalty_duplicates = duplicates * 1
    penalty_error = 1000 if has_error else 0
    total_penalty = (
        penalty_degenerate + penalty_encoding + penalty_duplicates + penalty_error
    )

    # Rewards (added)
    reward_tokens = tokens_total  # more clean tokens = better
    reward_density = density * 2  # denser pages are usually content-rich
    reward_structure = (blocks * 3) + (tables * 10) + (notes_columns * 5)
    reward_regions = (legend_regions * 8) + (abbreviation_regions * 8)
    total_reward = reward_tokens + reward_density + reward_structure + reward_regions

    score = round(total_reward - total_penalty, 1)

    return {
        "score": score,
        "tokens_total": tokens_total,
        "tokens_raw": tokens_raw,
        "density": density,
        "blocks": blocks,
        "tables": tables,
        "notes_columns": notes_columns,
        "legend_regions": legend_regions,
        "abbreviation_regions": abbreviation_regions,
        "penalty_degenerate": penalty_degenerate,
        "penalty_encoding": penalty_encoding,
        "penalty_duplicates": penalty_duplicates,
        "penalty_error": penalty_error,
        "total_reward": round(total_reward, 1),
        "total_penalty": total_penalty,
    }


def run_trial(
    pdf: Path,
    page: int,
    knobs: dict,
    label: str,
    run_root: Path,
    resolution: int = 200,
) -> dict:
    """Run the pipeline with a specific set of knob values.

    Returns a dict with the label, knobs used, manifest page data,
    quality score, and run directory path.
    """
    # Build config: start from defaults, overlay requested knobs
    cfg_kwargs: dict = {}
    for k, v in knobs.items():
        if k not in ALL_KNOBS:
            print(f"  WARNING: '{k}' is not a known knob — skipping")
            continue
        expected_type = ALL_KNOBS[k]
        try:
            cfg_kwargs[k] = expected_type(v)
        except (ValueError, TypeError) as e:
            print(f"  WARNING: knob '{k}' value {v!r} invalid ({e}) — using default")

    cfg = GroupingConfig(**cfg_kwargs)

    # Run pipeline on just the one page
    run_prefix = f"tune_{label}_{pdf.stem.replace(' ', '_')[:15]}"
    print(f"\n{'='*60}")
    print(f"Trial: {label}")
    print(f"Knobs: {cfg_kwargs or '(all defaults)'}")
    print(f"{'='*60}")

    run_dir = run_pdf(
        pdf=pdf,
        start=page,
        end=page + 1,
        resolution=resolution,
        run_root=run_root,
        run_prefix=run_prefix,
        cfg=cfg,
    )

    # Read back the manifest
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    page_data = manifest["pages"][0] if manifest.get("pages") else {}

    # Score it
    score_data = _quality_score(page_data)

    return {
        "label": label,
        "knobs": cfg_kwargs,
        "run_dir": str(run_dir),
        "score": score_data,
        "page_result": page_data,
    }


def print_comparison(trials: list[dict]) -> None:
    """Print a side-by-side comparison table to the console."""
    if not trials:
        return

    # Header
    print(f"\n{'='*80}")
    print("TUNING COMPARISON")
    print(f"{'='*80}")

    # Table header
    cols = [
        ("Label", 20),
        ("Score", 8),
        ("Tokens", 7),
        ("Density", 8),
        ("Blocks", 7),
        ("Tables", 7),
        ("Penalty", 8),
    ]
    header = "".join(name.ljust(width) for name, width in cols)
    print(header)
    print("-" * len(header))

    # Rows
    best_score = max(t["score"]["score"] for t in trials)
    for t in trials:
        s = t["score"]
        marker = " <-- BEST" if s["score"] == best_score else ""
        row = (
            f"{t['label'][:19]:<20}"
            f"{s['score']:<8}"
            f"{s['tokens_total']:<7}"
            f"{s['density']:<8}"
            f"{s['blocks']:<7}"
            f"{s['tables']:<7}"
            f"{s['total_penalty']:<8}"
            f"{marker}"
        )
        print(row)

    print()

    # Show what knobs each trial used
    print("Knobs per trial:")
    for t in trials:
        knob_str = ", ".join(f"{k}={v}" for k, v in t["knobs"].items()) or "(defaults)"
        print(f"  {t['label']}: {knob_str}")
    print()


# ── Recipe management ──────────────────────────────────────────────────


def save_recipe(name: str, knobs: dict, metadata: dict | None = None) -> Path:
    """Save a knob configuration as a named recipe JSON file.

    Recipes are stored in the project-root ``recipes/`` folder.
    Returns the path to the saved recipe.
    """
    RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    recipe_path = RECIPES_DIR / f"{safe_name}.json"

    recipe_data = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "knobs": knobs,
    }
    if metadata:
        recipe_data["metadata"] = metadata

    recipe_path.write_text(json.dumps(recipe_data, indent=2))
    print(f"Recipe saved: {recipe_path}")
    return recipe_path


def load_recipe(path: Path) -> list[dict]:
    """Load a recipe file.

    Accepts two formats:
    - A list of trials: [{"label": ..., "knobs": {...}}, ...]
    - A single saved recipe: {"name": ..., "knobs": {...}}

    Always returns a list of trial dicts.
    """
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    return [{"label": data.get("name", path.stem), "knobs": data.get("knobs", {})}]


def list_recipes() -> list[dict]:
    """List all saved recipes with their knobs."""
    if not RECIPES_DIR.exists():
        return []
    recipes = []
    for p in sorted(RECIPES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            recipes.append({
                "file": p.name,
                "name": data.get("name", p.stem),
                "created_at": data.get("created_at", "unknown"),
                "knobs": data.get("knobs", {}),
                "metadata": data.get("metadata"),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return recipes


def print_recipes() -> None:
    """Print all saved recipes to the console."""
    recipes = list_recipes()
    if not recipes:
        print(f"No saved recipes found in {RECIPES_DIR}")
        return
    print(f"\nSaved recipes ({RECIPES_DIR}):")
    print("-" * 70)
    for r in recipes:
        knob_str = ", ".join(f"{k}={v}" for k, v in r["knobs"].items()) or "(defaults)"
        print(f"  {r['file']:<30} {r['name']}")
        print(f"    Created: {r['created_at']}")
        print(f"    Knobs:   {knob_str}")
        if r.get("metadata"):
            meta = r["metadata"]
            if "pdf" in meta:
                print(f"    Tuned on: {meta['pdf']}")
            if "score" in meta:
                print(f"    Score:    {meta['score']}")
        print()


def _default_recipe() -> list[dict]:
    """A built-in recipe that tests the most impactful knobs."""
    return [
        {
            "label": "baseline",
            "knobs": {},
        },
        {
            "label": "margin_18pt",
            "knobs": {"tocr_margin_pts": 18.0},
        },
        {
            "label": "margin_36pt",
            "knobs": {"tocr_margin_pts": 36.0},
        },
        {
            "label": "no_rotated",
            "knobs": {"tocr_keep_rotated": False},
        },
        {
            "label": "normalize",
            "knobs": {"tocr_normalize_unicode": True},
        },
        {
            "label": "min_font_4pt",
            "knobs": {"tocr_min_font_size": 4.0},
        },
        {
            "label": "kitchen_sink",
            "knobs": {
                "tocr_margin_pts": 24.0,
                "tocr_min_font_size": 3.0,
                "tocr_normalize_unicode": True,
                "tocr_keep_rotated": False,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-tuning harness for TOCR knobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdf", type=Path, default=None, help="PDF file to test")
    parser.add_argument(
        "--page", type=int, default=0, help="Page number to test (0-indexed, default 0)"
    )
    parser.add_argument(
        "--save-winner",
        type=str,
        default=None,
        metavar="NAME",
        help="Save the winning knob config as a named recipe (e.g., 'odot_plans')",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="List all saved recipes and exit",
    )
    parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="JSON file with list of knob configs to try (see docstring for format)",
    )
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use the built-in default recipe (tests margin, rotation, normalization, font size)",
    )
    parser.add_argument(
        "--resolution", type=int, default=200, help="Overlay render DPI (default 200)"
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for run output (default: runs/)",
    )
    parser.add_argument(
        "--keep-runs", type=int, default=50, help="Max old runs to keep (default 50)"
    )
    parser.add_argument(
        "--knob",
        action="append",
        nargs=2,
        metavar=("NAME", "VALUE"),
        help="Set a single knob (can repeat). Example: --knob tocr_margin_pts 36",
    )
    args = parser.parse_args()

    # Handle --list-recipes (no PDF needed)
    if args.list_recipes:
        print_recipes()
        sys.exit(0)

    if args.pdf is None:
        print("Error: --pdf is required (unless using --list-recipes)")
        sys.exit(1)

    if not args.pdf.exists():
        print(f"Error: PDF not found: {args.pdf}")
        sys.exit(1)

    # Determine recipe
    recipe: list[dict]
    if args.recipe:
        if not args.recipe.exists():
            print(f"Error: recipe file not found: {args.recipe}")
            sys.exit(1)
        recipe = load_recipe(args.recipe)
        print(f"Loaded recipe with {len(recipe)} trials from {args.recipe}")
    elif args.use_defaults:
        recipe = _default_recipe()
        print(f"Using built-in default recipe ({len(recipe)} trials)")
    elif args.knob:
        # Single trial from --knob flags
        knobs = {}
        for name, value in args.knob:
            # Try to parse as number or bool
            if value.lower() in ("true", "false"):
                knobs[name] = value.lower() == "true"
            else:
                try:
                    knobs[name] = float(value)
                    if knobs[name] == int(knobs[name]):
                        knobs[name] = int(knobs[name])
                except ValueError:
                    knobs[name] = value
        recipe = [{"label": "custom", "knobs": knobs}]
    else:
        # Default: just run baseline
        recipe = [{"label": "baseline", "knobs": {}}]
        print("No recipe or knobs specified — running baseline only.")
        print(
            "Tip: use --use-defaults for a standard comparison, or --recipe for custom."
        )

    # Clean old runs
    cleanup_old_runs(args.run_root, keep=args.keep_runs)

    # Run each trial
    trials: list[dict] = []
    for i, trial_spec in enumerate(recipe, 1):
        label = trial_spec.get("label", f"trial_{i}")
        knobs = trial_spec.get("knobs", {})
        print(f"\n[{i}/{len(recipe)}] Starting trial: {label}")
        result = run_trial(
            pdf=args.pdf,
            page=args.page,
            knobs=knobs,
            label=label,
            run_root=args.run_root,
            resolution=args.resolution,
        )
        trials.append(result)

    # Print comparison
    print_comparison(trials)

    # Save report to scripts/tuning_reports/
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"tuning_{stamp}.json"
    best = max(trials, key=lambda t: t["score"]["score"])
    report = {
        "created_at": datetime.now().isoformat(),
        "pdf": str(args.pdf.resolve()),
        "page": args.page,
        "resolution": args.resolution,
        "trials": [
            {
                "label": t["label"],
                "knobs": t["knobs"],
                "run_dir": t["run_dir"],
                "score": t["score"],
            }
            for t in trials
        ],
        "best": best["label"],
        "available_knobs": {k: v.__name__ for k, v in ALL_KNOBS.items()},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {report_path}")

    # Final recommendation
    print(f"\nBest trial: {best['label']} (score: {best['score']['score']})")
    if best["knobs"]:
        print(f"Winning knobs: {json.dumps(best['knobs'], indent=2)}")
    else:
        print("Winning config: defaults (no knobs changed)")

    # Save winner as a named recipe if requested
    if args.save_winner:
        save_recipe(
            name=args.save_winner,
            knobs=best["knobs"],
            metadata={
                "pdf": str(args.pdf.resolve()),
                "page": args.page,
                "score": best["score"]["score"],
                "tuning_report": str(report_path),
            },
        )


if __name__ == "__main__":
    main()
