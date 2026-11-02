"""Runs the golden set and aggregates scores.

Two modes:
  retrieval  rule-based routing + search only. Free, deterministic, runs in CI.
  agent      the full LangGraph pipeline. With a model it also scores the generated
             answer (facts, citations, abstention) and, optionally, faithfulness via a judge.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..graph import MIN_COVERAGE, Agent, _records
from ..llm import LLM
from ..retriever import parse_request
from . import metrics as m
from .judge import judge_faithfulness

GOLDEN = Path(__file__).with_name("golden.jsonl")
ABSTAINED = {"no_evidence", "out_of_scope"}

# Regression gates: set a little below the current baseline so a real drop fails the run.
THRESHOLDS = {
    "retrieval": {"hit@1": 0.80, "hit@5": 0.85, "mrr": 0.80, "abstention_accuracy": 0.50},
    "agent": {"hit@5": 0.85, "fact_recall": 0.80, "citation_precision": 0.95, "abstention_accuracy": 0.75,
              "faithfulness": 0.90},
}


@dataclass
class Case:
    id: str
    question: str
    relevant: set[str] = field(default_factory=set)
    must_include: list[str] = field(default_factory=list)
    expect: str = "answer"
    tags: list[str] = field(default_factory=list)


def load_cases(path: Path = GOLDEN) -> list[Case]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["relevant"] = set(row.get("relevant", []))
            cases.append(Case(**row))
    return cases


def _retrieval_scores(retrieved: list[str], case: Case) -> dict:
    return {
        "hit@1": m.hit_at_k(retrieved, case.relevant, 1),
        "hit@5": m.hit_at_k(retrieved, case.relevant, 5),
        "recall@5": m.recall_at_k(retrieved, case.relevant, 5),
        "mrr": m.reciprocal_rank(retrieved, case.relevant),
    }


def run_retrieval_case(agent: Agent, case: Case, k: int = 5) -> dict:
    query, filters = parse_request(case.question, agent.catalogue.programme_names)
    hits = agent.retriever.search(query, filters, k)
    retrieved = [h.module.id for h in hits]
    would_answer = any(h.exact_id or h.coverage >= MIN_COVERAGE for h in hits)
    result = {"id": case.id, "question": case.question, "tags": case.tags, "retrieved": retrieved}
    if case.expect == "no_answer":
        result["abstention_accuracy"] = float(not would_answer)
    else:
        result |= _retrieval_scores(retrieved, case)
    return result


def run_agent_case(agent: Agent, case: Case, judge: LLM | None = None) -> dict:
    started = time.perf_counter()
    state = agent.ask(case.question)
    latency = time.perf_counter() - started
    evidence = [h.module.id for h in state["hits"]]
    result = {
        "id": case.id, "question": case.question, "tags": case.tags, "mode": state["mode"],
        "answer": state["answer"], "retrieved": evidence, "citations": state["citations"],
        "latency_s": round(latency, 2), "trace": state.get("trace", []),
    }
    abstained = state["mode"] in ABSTAINED
    if case.expect == "no_answer":
        result["abstention_accuracy"] = float(abstained)
        return result
    result |= _retrieval_scores(evidence, case)
    result["abstention_accuracy"] = float(not abstained)
    result["fact_recall"] = m.fact_recall(state["answer"], case.must_include)
    if state["mode"] == "generated":
        result["citation_precision"] = m.citation_precision(state["citations"], evidence)
        result["cites_relevant"] = m.cites_relevant(state["citations"], case.relevant)
        if judge is not None:
            verdict = judge_faithfulness(judge, _records(state["hits"]), case.question, state["answer"])
            result["faithfulness"] = verdict["faithfulness"]
            result["unsupported_claims"] = verdict["unsupported"]
    return result


METRICS = ["hit@1", "hit@5", "recall@5", "mrr", "abstention_accuracy", "fact_recall",
           "citation_precision", "cites_relevant", "faithfulness", "latency_s"]


def aggregate(results: list[dict]) -> dict:
    summary = {name: m.mean([r[name] for r in results if name in r]) for name in METRICS}
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        for tag in r["tags"]:
            by_tag[tag].append(r)
    tags = {
        tag: {"cases": len(rs), "hit@5": m.mean([r["hit@5"] for r in rs if "hit@5" in r]),
              "abstention_accuracy": m.mean([r["abstention_accuracy"] for r in rs if "abstention_accuracy" in r])}
        for tag, rs in sorted(by_tag.items())
    }
    return {"cases": len(results), "metrics": {k: v for k, v in summary.items() if v is not None}, "by_tag": tags}


def check_thresholds(summary: dict, mode: str) -> list[str]:
    failures = []
    for name, minimum in THRESHOLDS[mode].items():
        value = summary["metrics"].get(name)
        if value is not None and value < minimum:
            failures.append(f"{name} = {value} is below {minimum}")
    return failures


def failed_cases(results: list[dict]) -> list[dict]:
    keys = ("hit@5", "abstention_accuracy", "fact_recall", "faithfulness")
    return [r for r in results if any(r.get(k, 1) < 1 for k in keys)]


def to_markdown(report: dict) -> str:
    lines = [f"# Eval report: {report['mode']} mode", "",
             f"{report['summary']['cases']} cases · model: {report['model'] or 'none (rule-based)'} · "
             f"catalogue {report['catalogue_generated_at'][:10]}", "",
             "| Metric | Score | Gate |", "| --- | --- | --- |"]
    gates = THRESHOLDS[report["mode"]]
    for name, value in report["summary"]["metrics"].items():
        gate = f"≥ {gates[name]}" if name in gates else ""
        lines.append(f"| {name} | {value} | {gate} |")
    lines += ["", "## By tag", "", "| Tag | Cases | hit@5 | Abstention |", "| --- | --- | --- | --- |"]
    for tag, row in report["summary"]["by_tag"].items():
        lines.append(f"| {tag} | {row['cases']} | {row['hit@5'] if row['hit@5'] is not None else '–'} | "
                     f"{row['abstention_accuracy'] if row['abstention_accuracy'] is not None else '–'} |")
    lines += ["", "## Failing cases", ""]
    for r in report["failures_detail"] or []:
        lines.append(f"- **{r['id']}**: {r['question']} → retrieved {r['retrieved'][:3] or 'nothing'}"
                     + (f", mode `{r['mode']}`" if "mode" in r else ""))
        for claim in r.get("unsupported_claims", []):
            lines.append(f"  - unsupported: {claim['claim']} ({claim['reason']})")
    if not report["failures_detail"]:
        lines.append("None.")
    lines += ["", f"Gates: {'PASS' if not report['gate_failures'] else 'FAIL — ' + '; '.join(report['gate_failures'])}"]
    return "\n".join(lines) + "\n"
