#!/usr/bin/env python
"""Query-engine evaluation harness.

Runs a question bank against the DocumentQueryEngine, scores answers
using keyword matching and (optionally) LLM-as-judge, and writes a
scored report.

Usage::

    # Basic keyword-only eval against a run directory:
    python scripts/diagnostics/run_query_eval.py --run runs/run_20260219_180300_IFC_page2

    # With LLM-as-judge scoring (requires LLM connectivity):
    python scripts/diagnostics/run_query_eval.py --run runs/... --llm-judge

    # Custom question bank:
    python scripts/diagnostics/run_query_eval.py --run runs/... --bank path/to/questions.json

    # Output to a specific directory:
    python scripts/diagnostics/run_query_eval.py --run runs/... --out-dir reports/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

DEFAULT_BANK = _root / "tests" / "query" / "question_bank.json"

# ── scoring helpers ───────────────────────────────────────────


def keyword_score(answer: str, keywords: list[str]) -> float:
    """Return fraction of required keywords found (case-insensitive regex)."""
    if not keywords:
        return 1.0  # no keywords → auto-pass
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if re.search(re.escape(kw).lower(), answer_lower))
    return hits / len(keywords)


def llm_judge_score(
    question: str,
    reference: str,
    answer: str,
    llm_client: "LLMClient",
) -> dict:
    """Use an LLM to rate the answer on a 1-5 scale for relevance."""
    prompt = (
        "You are an expert construction-document reviewer evaluating an AI "
        "assistant's answer.\n\n"
        f"**Question:** {question}\n"
        f"**Reference answer:** {reference}\n"
        f"**Assistant's answer:** {answer}\n\n"
        "Rate the assistant's answer on a 1-5 scale:\n"
        "  1 = completely wrong or irrelevant\n"
        "  2 = partially relevant but largely incorrect\n"
        "  3 = relevant but missing key information\n"
        "  4 = mostly correct with minor omissions\n"
        "  5 = excellent, complete and accurate\n\n"
        'Respond with a JSON object: {"score": <int>, "reason": "<brief explanation>"}'
    )
    try:
        raw = llm_client.chat(prompt)
        # Try to parse JSON from the response
        m = re.search(r"\{[^}]+\}", raw)
        if m:
            data = json.loads(m.group())
            return {
                "llm_score": int(data.get("score", 0)),
                "llm_reason": data.get("reason", ""),
            }
    except Exception as exc:
        return {"llm_score": 0, "llm_reason": f"Error: {exc}"}
    return {"llm_score": 0, "llm_reason": "Could not parse LLM response"}


# ── main harness ──────────────────────────────────────────────


def run_eval(
    engine: "DocumentQueryEngine",
    questions: list[dict],
    *,
    use_llm_judge: bool = False,
    judge_client: "LLMClient | None" = None,
    verbose: bool = True,
) -> list[dict]:
    """Run all questions through *engine* and return scored results."""
    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        qid = q.get("id", f"q{i:02d}")
        text = q["question"]
        difficulty = q.get("difficulty", "unknown")
        topic = q.get("topic", "unknown")
        required_kw = q.get("required_keywords", [])
        reference = q.get("expected_answer", "")

        if verbose:
            print(
                f"  [{i}/{len(questions)}] {qid}: {text[:60]}...", end=" ", flush=True
            )

        t0 = time.perf_counter()
        try:
            result = engine.query(text)
            answer = result.text
            elapsed = time.perf_counter() - t0
            error = None
        except Exception as exc:
            answer = ""
            elapsed = time.perf_counter() - t0
            error = str(exc)

        kw_score = keyword_score(answer, required_kw)
        has_content = len(answer.strip()) > 10
        n_sources = (
            len(result.sources) if hasattr(result, "sources") and result.sources else 0
        )

        entry = {
            "id": qid,
            "question": text,
            "topic": topic,
            "difficulty": difficulty,
            "answer": answer[:500],  # truncate for report
            "answer_length": len(answer),
            "keyword_score": round(kw_score, 3),
            "has_content": has_content,
            "num_sources": n_sources,
            "latency_s": round(elapsed, 2),
            "cached": getattr(result, "cached", False) if error is None else False,
            "error": error,
        }

        if use_llm_judge and judge_client and error is None:
            judge = llm_judge_score(text, reference, answer, judge_client)
            entry.update(judge)

        results.append(entry)

        if verbose:
            status = f"kw={kw_score:.0%}"
            if "llm_score" in entry:
                status += f" llm={entry['llm_score']}/5"
            status += f" ({elapsed:.1f}s)"
            print(status)

    return results


def compute_summary(results: list[dict]) -> dict:
    """Aggregate scores by difficulty and topic."""
    total = len(results)
    if total == 0:
        return {}

    kw_scores = [r["keyword_score"] for r in results]
    latencies = [r["latency_s"] for r in results]
    errored = sum(1 for r in results if r["error"])
    has_content_pct = sum(1 for r in results if r["has_content"]) / total

    summary = {
        "total_questions": total,
        "errors": errored,
        "avg_keyword_score": round(sum(kw_scores) / total, 3),
        "keyword_perfect": sum(1 for s in kw_scores if s >= 1.0),
        "has_content_pct": round(has_content_pct, 3),
        "avg_latency_s": round(sum(latencies) / total, 2),
        "max_latency_s": round(max(latencies), 2),
    }

    # By difficulty
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            avg_kw = sum(r["keyword_score"] for r in subset) / len(subset)
            summary[f"{diff}_avg_kw_score"] = round(avg_kw, 3)
            summary[f"{diff}_count"] = len(subset)

    # LLM judge stats (if present)
    llm_scores = [
        r["llm_score"] for r in results if "llm_score" in r and r["llm_score"] > 0
    ]
    if llm_scores:
        summary["llm_judge_avg"] = round(sum(llm_scores) / len(llm_scores), 2)
        summary["llm_judge_count"] = len(llm_scores)

    return summary


def write_report(
    results: list[dict],
    summary: dict,
    out_dir: Path,
    run_name: str,
) -> Path:
    """Write a Markdown evaluation report and raw JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON data
    json_path = out_dir / f"eval_{run_name}_{ts}.json"
    json_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    # Markdown report
    md_path = out_dir / f"eval_{run_name}_{ts}.md"
    lines = [
        f"# Query Eval Report — {run_name}",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Per-Question Results", ""]

    for r in results:
        status = (
            "✅"
            if r["keyword_score"] >= 0.8 and not r["error"]
            else "❌" if r["error"] else "⚠️"
        )
        lines.append(f"### {status} {r['id']} — {r['question'][:60]}")
        lines.append(f"- **Topic:** {r['topic']}  **Difficulty:** {r['difficulty']}")
        lines.append(
            f"- **Keyword score:** {r['keyword_score']:.0%}  "
            f"**Sources:** {r['num_sources']}  **Latency:** {r['latency_s']}s"
        )
        if r.get("llm_score"):
            lines.append(
                f"- **LLM Judge:** {r['llm_score']}/5 — {r.get('llm_reason', '')}"
            )
        if r["error"]:
            lines.append(f"- **Error:** {r['error']}")
        else:
            # Show first 200 chars of answer
            ans_preview = r["answer"][:200].replace("\n", " ")
            lines.append(f"- **Answer preview:** {ans_preview}")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nReport written to:\n  {md_path}\n  {json_path}")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Query engine evaluation harness")
    parser.add_argument("--run", required=True, help="Path to a saved run directory")
    parser.add_argument("--bank", default=str(DEFAULT_BANK), help="Question bank JSON")
    parser.add_argument("--out-dir", default="reports", help="Output directory")
    parser.add_argument("--provider", default="ollama", help="LLM provider")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--api-base", default="http://localhost:11434")
    parser.add_argument(
        "--policy",
        default="local_only",
        choices=["local_only", "cloud_allowed", "cloud_with_consent"],
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM-as-judge for relevance scoring",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to first N questions (0 = all)"
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    # Load question bank
    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(f"Error: question bank not found: {bank_path}")
        sys.exit(1)
    bank = json.loads(bank_path.read_text())
    questions = bank["questions"]
    if args.limit > 0:
        questions = questions[: args.limit]

    print(f"Loaded {len(questions)} questions from {bank_path.name}")

    # Load run
    from plancheck.export.run_loader import load_run
    from plancheck.llm.query_engine import DocumentQueryEngine

    dr = load_run(args.run)
    n_pages = len(dr.pages) if dr.pages else 0
    print(f"Loaded run with {n_pages} page(s)")

    engine = DocumentQueryEngine.from_document_result(
        dr,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
        policy=args.policy,
    )
    print(f"Index contains {engine.index.count} chunks\n")

    # Optional judge client
    judge_client = None
    if args.llm_judge:
        from plancheck.llm.client import LLMClient

        judge_client = LLMClient(
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            policy=args.policy,
        )

    # Run eval
    print("Running evaluation...")
    results = run_eval(
        engine,
        questions,
        use_llm_judge=args.llm_judge,
        judge_client=judge_client,
        verbose=not args.quiet,
    )

    summary = compute_summary(results)
    run_name = Path(args.run).name

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Evaluation Summary — {run_name}")
    print(f"{'=' * 50}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()

    write_report(results, summary, Path(args.out_dir), run_name)


if __name__ == "__main__":
    main()
