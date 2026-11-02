"""python -m btu_rag.eval [--mode retrieval|agent] [--judge] [--only TAG]"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..catalogue import cached_catalogue
from ..config import ROOT, Settings, has_credentials
from ..graph import Agent
from ..llm import ClaudeLLM
from .runner import aggregate, check_thresholds, failed_cases, load_cases, run_agent_case, run_retrieval_case, to_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers on the golden set.")
    parser.add_argument("--mode", choices=["retrieval", "agent"], default="retrieval")
    parser.add_argument("--judge", action="store_true", help="score faithfulness with an LLM judge (agent mode)")
    parser.add_argument("--only", help="run only cases with this tag")
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    settings = Settings.from_env()
    catalogue = cached_catalogue(str(settings.catalogue_path))
    llm = judge = None
    if args.mode == "agent":
        if not has_credentials():
            print("Agent mode needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN). "
                  "Use --mode retrieval for the free, key-less run.", file=sys.stderr)
            return 2
        llm = ClaudeLLM(settings)
        judge = ClaudeLLM(settings, model=settings.judge_model) if args.judge else None
    agent = Agent(catalogue, llm, settings.top_k)

    cases = [c for c in load_cases() if not args.only or args.only in c.tags]
    results = []
    for i, case in enumerate(cases, 1):
        result = (run_retrieval_case(agent, case) if args.mode == "retrieval"
                  else run_agent_case(agent, case, judge))
        results.append(result)
        print(f"[{i}/{len(cases)}] {case.id}", file=sys.stderr)

    summary = aggregate(results)
    gate_failures = check_thresholds(summary, args.mode)
    report = {
        "mode": args.mode,
        "model": settings.model if llm else None,
        "judge_model": settings.judge_model if judge else None,
        "catalogue_generated_at": catalogue.generated_at,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "gate_failures": gate_failures,
        "failures_detail": failed_cases(results),
        "results": results,
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval-{args.mode}-{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    markdown = to_markdown(report)
    (out_dir / f"eval-{args.mode}-{stamp}.md").write_text(markdown)
    print(markdown)
    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
