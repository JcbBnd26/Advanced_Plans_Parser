#!/usr/bin/env python
"""Interactive query tool for construction plan documents.

Run a pipeline on a PDF page (or load a saved run), index the content,
and enter an interactive Q&A loop.

Usage::

    # From a PDF:
    python scripts/query/interactive_query.py plan.pdf --page 0

    # From a saved run directory:
    python scripts/query/interactive_query.py --run runs/run_20260219_180300_IFC_page2

    # With a specific LLM provider:
    python scripts/query/interactive_query.py plan.pdf --provider openai --model gpt-4o-mini --api-key sk-...

Commands inside the interactive loop:
    /search <text>   — semantic search only (no LLM)
    /page <N>        — filter subsequent queries to page N
    /page all        — remove page filter
    /history         — show query history
    /cost            — show cost so far
    /clear           — clear the response cache
    /help            — show commands
    /quit or /exit   — exit
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _load_from_pdf(pdf_path: str, page: int) -> "PageResult":
    """Run the pipeline on a single page and return the PageResult."""
    from plancheck.config import GroupingConfig
    from plancheck.pipeline import run_pipeline

    cfg = GroupingConfig()
    print(f"Running pipeline on {pdf_path}, page {page} ...")
    pr = run_pipeline(pdf_path, page, cfg=cfg)
    print(f"Pipeline complete — {len(pr.blocks)} blocks extracted.")
    return pr


def _load_from_run(run_dir: str) -> "DocumentResult":
    """Load a saved run directory."""
    from plancheck.export.run_loader import load_run

    print(f"Loading run from {run_dir} ...")
    dr = load_run(run_dir)
    n_pages = len(dr.pages) if dr.pages else 0
    print(f"Loaded {n_pages} page(s).")
    return dr


def _run_interactive(engine: "DocumentQueryEngine") -> None:
    """Enter the interactive Q&A loop."""
    page_filter: int | None = None

    print("\n" + "=" * 60)
    print("  Plan Document Query Engine — Interactive Mode")
    print("  Type a question, or /help for commands.")
    print("=" * 60 + "\n")

    while True:
        try:
            raw = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────
        if raw.startswith("/"):
            cmd_parts = raw.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye.")
                break

            elif cmd == "/help":
                print(
                    "Commands:\n"
                    "  /search <text>  — search without LLM\n"
                    "  /page <N>       — filter to page N\n"
                    "  /page all       — remove page filter\n"
                    "  /history        — show query history\n"
                    "  /cost           — show cost summary\n"
                    "  /clear          — clear response cache\n"
                    "  /quit           — exit\n"
                )

            elif cmd == "/search":
                if not arg:
                    print("Usage: /search <query text>")
                    continue
                results = engine.search_only(arg, n_results=5, page_filter=page_filter)
                if not results:
                    print("  (no results)")
                for r in results:
                    c = r.chunk
                    print(
                        f"  [{r.rank}] score={r.score:.3f}  page={c.page}  "
                        f"type={c.region_type}"
                    )
                    print(f"      {c.text[:120]}")
                print()

            elif cmd == "/page":
                if arg.lower() in ("all", "none", ""):
                    page_filter = None
                    print("  Page filter removed.")
                else:
                    try:
                        page_filter = int(arg)
                        print(f"  Filtering to page {page_filter}.")
                    except ValueError:
                        print("  Usage: /page <number> or /page all")

            elif cmd == "/history":
                hist = engine.history
                if not hist:
                    print("  (no history)")
                for i, h in enumerate(hist, 1):
                    cached = " [cached]" if h.get("cached") else ""
                    print(f"  {i}. {h['question'][:60]}{cached}")
                print()

            elif cmd == "/cost":
                s = engine.cost_summary
                print(f"  Calls: {s['call_count']}")
                print(f"  Input tokens: {s['total_input_tokens']:,}")
                print(f"  Output tokens: {s['total_output_tokens']:,}")
                print(f"  Est. cost: ${s['total_cost_usd']:.4f}")
                print()

            elif cmd == "/clear":
                engine.clear_cache()
                print("  Cache cleared.")

            else:
                print(f"  Unknown command: {cmd}  (try /help)")

            continue

        # ── Regular question ──────────────────────────────────
        result = engine.query(raw, page_filter=page_filter)

        if result.cached:
            print("  [cached]")
        print()
        print(result.text)
        print()

        if result.sources:
            print("  Sources:")
            for s in result.sources[:5]:
                page = s.get("page", "?")
                rtype = s.get("region_type", "?")
                excerpt = s.get("excerpt", "")[:80]
                print(f"    - Page {page} ({rtype}): {excerpt}")
            print()

        if result.meta:
            m = result.meta
            print(
                f"  [{m.provider}/{m.model} — {m.latency_s:.1f}s, "
                f"~{m.input_tokens}+{m.output_tokens} tokens, "
                f"${m.cost_usd:.4f}]"
            )
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive query tool for construction plan documents"
    )
    parser.add_argument("pdf", nargs="?", help="Path to PDF file")
    parser.add_argument("--page", type=int, default=0, help="Page number (default: 0)")
    parser.add_argument("--run", help="Path to a saved run directory")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument(
        "--api-base", default="http://localhost:11434", help="API base URL"
    )
    parser.add_argument(
        "--policy",
        default="local_only",
        choices=["local_only", "cloud_allowed", "cloud_with_consent"],
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    if not args.pdf and not args.run:
        parser.error("Provide either a PDF path or --run <directory>")

    from plancheck.llm.query_engine import DocumentQueryEngine

    if args.run:
        dr = _load_from_run(args.run)
        engine = DocumentQueryEngine.from_document_result(
            dr,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            policy=args.policy,
        )
    else:
        pr = _load_from_pdf(args.pdf, args.page)
        engine = DocumentQueryEngine.from_page_result(
            pr,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            policy=args.policy,
        )

    print(f"Index contains {engine.index.count} chunks.")
    _run_interactive(engine)


if __name__ == "__main__":
    main()
