#!/usr/bin/env python
"""LLM Cost & Latency Benchmark (Phase 0.2).

This script:
1. Assembles a realistic prompt from sample plan-page data (or a synthetic
   stand-in if no real data is available).
2. Estimates token counts for each provider/model.
3. Optionally sends the prompt to live providers to measure actual latency
   and response quality.
4. Writes a Markdown cost-budget table to ``docs/LLM_BUDGET.md``.

Usage::

    # Offline (cost estimation only, no LLM calls):
    python scripts/diagnostics/run_llm_benchmark.py

    # Live benchmark against a running Ollama instance:
    python scripts/diagnostics/run_llm_benchmark.py --live --providers ollama

    # Live against all configured providers:
    python scripts/diagnostics/run_llm_benchmark.py --live --providers ollama openai anthropic \\
        --openai-key sk-... --anthropic-key sk-ant-...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using tiktoken (OpenAI tokeniser)."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~4 chars per token for English text
        return max(1, len(text) // 4)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Pricing tables (USD per 1K tokens, as of mid-2025)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict] = {
    "ollama/llama3.1:8b": {
        "input_per_1k": 0.0,
        "output_per_1k": 0.0,
        "label": "Llama 3.1 8B (local)",
    },
    "openai/gpt-4o-mini": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
        "label": "GPT-4o-mini",
    },
    "openai/gpt-4-turbo": {
        "input_per_1k": 0.01,
        "output_per_1k": 0.03,
        "label": "GPT-4 Turbo",
    },
    "anthropic/claude-3-haiku-20240307": {
        "input_per_1k": 0.00025,
        "output_per_1k": 0.00125,
        "label": "Claude 3 Haiku",
    },
    "anthropic/claude-sonnet-4-20250514": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
        "label": "Claude Sonnet 4",
    },
}

# ---------------------------------------------------------------------------
# Sample prompt assembly
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a construction-plan compliance checker.  Given the extracted text
    from a sheet's notes column, identify any potential building-code
    violations, ambiguous references, or missing information.  Return a JSON
    array of findings.
"""
)

_SAMPLE_NOTES = textwrap.dedent(
    """\
    GENERAL NOTES:
    1. ALL WORK SHALL COMPLY WITH THE 2021 INTERNATIONAL BUILDING CODE (IBC)
       AND LOCAL AMENDMENTS.
    2. CONTRACTOR SHALL VERIFY ALL DIMENSIONS AND CONDITIONS IN THE FIELD
       PRIOR TO COMMENCEMENT OF WORK.
    3. FIRE-RATED ASSEMBLIES SHALL BE CONSTRUCTED IN ACCORDANCE WITH UL
       DESIGN NO. U305 (1-HOUR) OR U411 (2-HOUR) AS INDICATED ON PLANS.
    4. EXIT CORRIDORS SHALL MAINTAIN A MINIMUM CLEAR WIDTH OF 44 INCHES.
       (IBC 1005.1)
    5. MAXIMUM TRAVEL DISTANCE TO AN EXIT SHALL NOT EXCEED 250 FEET IN
       FULLY SPRINKLERED BUILDINGS.  (IBC 1017.2)
    6. PROVIDE 1 ACCESSIBLE ENTRANCE PER IBC 1105.1.  ALL ACCESSIBLE ROUTES
       SHALL COMPLY WITH ICC A117.1.
    7. ROOF LIVE LOAD: 20 PSF.  GROUND SNOW LOAD: 30 PSF.
    8. SEISMIC DESIGN CATEGORY: D.  IMPORTANCE FACTOR: 1.25.
    9. ALL STRUCTURAL STEEL SHALL CONFORM TO ASTM A992 (WIDE FLANGES) OR
       ASTM A500 GR. B (HSS).
    10. CONCRETE: f'c = 4000 PSI @ 28 DAYS, NORMAL WEIGHT (150 PCF).
    11. REBAR: ASTM A615, GRADE 60.
    12. ELECTRICAL: NEC 2023 EDITION.  PANEL SCHEDULES ON SHEET E-201.
    13. PLUMBING: IPC 2021.  FIXTURE COUNTS PER TABLE 403.1.
    14. MECHANICAL: IMC 2021.  MINIMUM OUTSIDE AIR PER ASHRAE 62.1.
    15. INSULATION: ROOF R-30, WALLS R-19, SLAB PERIMETER R-10.
    16. SEE SHEET A-501 FOR WALL TYPES AND DOOR SCHEDULE.
    17. SEE STRUCTURAL SHEETS S-100 THROUGH S-105 FOR FOUNDATION PLAN.
    18. ALL DIMENSIONS ARE TO FACE OF STUD U.N.O.
    19. DO NOT SCALE DRAWINGS.
    20. REFER TO SPECIFICATIONS SECTIONS 03300 (CAST-IN-PLACE CONCRETE),
        05120 (STRUCTURAL STEEL), AND 09250 (GYPSUM BOARD) FOR ADDITIONAL
        REQUIREMENTS.
"""
)

# Repeat the notes to simulate a realistic multi-page context (~8K tokens)
_FULL_PROMPT = _SAMPLE_NOTES * 4


def _build_prompt() -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""
    return _SYSTEM_PROMPT, _FULL_PROMPT


# ---------------------------------------------------------------------------
# Live benchmark
# ---------------------------------------------------------------------------


def _benchmark_live(
    provider: str,
    model: str,
    api_key: str,
    api_base: str,
    system_prompt: str,
    user_prompt: str,
    policy: str,
) -> dict:
    """Call the LLM and measure latency + response length."""
    # Import the existing LLMClient
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from plancheck.checks.llm_checks import LLMClient

    client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=0.1,
        policy=policy,
    )

    t0 = time.perf_counter()
    try:
        response = client.chat(system_prompt, user_prompt)
        elapsed = time.perf_counter() - t0
        return {
            "success": True,
            "latency_s": round(elapsed, 2),
            "response_chars": len(response),
            "response_tokens_est": _estimate_tokens(response),
            "response_preview": response[:200],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "latency_s": round(elapsed, 2),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def _calc_cost(
    key: str,
    input_tokens: int,
    output_tokens_est: int = 500,
) -> dict:
    """Calculate cost for a single query and for 100 compliance queries."""
    info = PRICING[key]
    cost_in = (input_tokens / 1000) * info["input_per_1k"]
    cost_out = (output_tokens_est / 1000) * info["output_per_1k"]
    cost_query = cost_in + cost_out
    return {
        "label": info["label"],
        "input_tokens": input_tokens,
        "output_tokens_est": output_tokens_est,
        "cost_per_query": cost_query,
        "cost_100_queries": cost_query * 100,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    estimates: list[dict],
    live_results: dict[str, dict] | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown report."""

    lines = [
        "# LLM Cost & Latency Budget",
        "",
        "> Auto-generated by `scripts/diagnostics/run_llm_benchmark.py`.",
        "",
        "## Methodology",
        "",
        "- **Prompt**: Synthetic plan-notes text (~4× a typical notes column)",
        f"- **Input tokens (est.)**: {estimates[0]['input_tokens']:,}",
        "- **Output tokens (est.)**: 500 per query",
        "- **Compliance scenario**: 100 requirement queries per plan set",
        "",
        "## Cost Estimates",
        "",
        "| Provider | Input Tokens | Cost/Query | Compliance (100 q) |",
        "|---|---:|---:|---:|",
    ]

    for e in estimates:
        cq = f"${e['cost_per_query']:.4f}" if e["cost_per_query"] > 0 else "$0.00"
        c100 = f"${e['cost_100_queries']:.2f}" if e["cost_100_queries"] > 0 else "$0.00"
        lines.append(f"| {e['label']} | {e['input_tokens']:,} | {cq} | {c100} |")

    if live_results:
        lines += [
            "",
            "## Live Benchmark Results",
            "",
            "| Provider | Latency | Output Tokens (est.) | Status |",
            "|---|---:|---:|---|",
        ]
        for key, res in live_results.items():
            label = PRICING.get(key, {}).get("label", key)
            if res["success"]:
                lines.append(
                    f"| {label} | {res['latency_s']:.1f}s "
                    f"| ~{res['response_tokens_est']} | OK |"
                )
            else:
                lines.append(
                    f"| {label} | {res['latency_s']:.1f}s | — | "
                    f"FAIL: {res['error'][:60]} |"
                )

        # Extrapolate compliance-run time
        lines += [
            "",
            "### Projected Compliance Run (100 queries)",
            "",
            "| Provider | Est. Wall Time |",
            "|---|---:|",
        ]
        for key, res in live_results.items():
            label = PRICING.get(key, {}).get("label", key)
            if res["success"]:
                total_s = res["latency_s"] * 100
                mins = math.ceil(total_s / 60)
                lines.append(f"| {label} | ~{mins} min |")

    lines += [
        "",
        "## Decision Notes",
        "",
        "- **Local Ollama** is cost-free and keeps data on-device "
        "(satisfies `local_only` policy).",
        "- **GPT-4o-mini / Claude Haiku** offer the best price/performance "
        "for cloud budgets.",
        "- **GPT-4 Turbo / Claude Sonnet** should be reserved for "
        "high-value compliance checks where accuracy trumps cost.",
        "- Caching (Phase 1.1) will reduce repeated-query costs by "
        "an estimated 60-80%.",
        "",
    ]

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Report written to {output_path}")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM cost & latency benchmark")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live benchmarks against configured providers",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=[],
        help="Providers to benchmark live (e.g. ollama openai anthropic)",
    )
    parser.add_argument("--openai-key", default="", help="OpenAI API key")
    parser.add_argument("--anthropic-key", default="", help="Anthropic API key")
    parser.add_argument("--ollama-base", default="http://localhost:11434")
    parser.add_argument(
        "--output",
        default="docs/LLM_BUDGET.md",
        help="Output Markdown file path",
    )
    args = parser.parse_args()

    system_prompt, user_prompt = _build_prompt()
    full_text = system_prompt + user_prompt

    # Token counting
    input_tokens_tiktoken = _count_tokens_tiktoken(full_text)
    input_tokens_est = _estimate_tokens(full_text)
    print(f"Prompt length: {len(full_text):,} chars")
    print(f"Token estimate (tiktoken): {input_tokens_tiktoken:,}")
    print(f"Token estimate (heuristic): {input_tokens_est:,}")
    print()

    # Cost estimates for all models
    estimates = []
    for key in PRICING:
        e = _calc_cost(key, input_tokens_tiktoken)
        estimates.append(e)
        print(
            f"  {e['label']:25s}  cost/query: ${e['cost_per_query']:.4f}"
            f"  100-query: ${e['cost_100_queries']:.2f}"
        )

    print()

    # Live benchmarks
    live_results: dict[str, dict] | None = None
    if args.live and args.providers:
        live_results = {}
        provider_model_map = {
            "ollama": ("ollama", "llama3.1:8b", "", args.ollama_base),
            "openai": ("openai", "gpt-4o-mini", args.openai_key, ""),
            "anthropic": (
                "anthropic",
                "claude-3-haiku-20240307",
                args.anthropic_key,
                "",
            ),
        }
        for prov_name in args.providers:
            if prov_name not in provider_model_map:
                print(f"  Unknown provider '{prov_name}', skipping")
                continue
            prov, model, key, base = provider_model_map[prov_name]
            lookup_key = f"{prov}/{model}"
            # Use cloud_allowed for cloud providers during benchmarking
            policy = "local_only" if prov == "ollama" else "cloud_allowed"
            print(f"  Benchmarking {lookup_key} ...")
            result = _benchmark_live(
                prov, model, key, base, system_prompt, user_prompt, policy
            )
            live_results[lookup_key] = result
            if result["success"]:
                print(
                    f"    OK — {result['latency_s']:.1f}s, "
                    f"~{result['response_tokens_est']} output tokens"
                )
            else:
                print(f"    FAILED: {result['error']}")
        print()

    # Generate report
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[2] / output_path

    report = _generate_report(estimates, live_results, output_path)
    print(report)


if __name__ == "__main__":
    main()
