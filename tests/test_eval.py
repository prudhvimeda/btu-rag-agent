from conftest import FakeLLM

from btu_rag.eval import metrics as m
from btu_rag.eval.judge import judge_faithfulness
from btu_rag.eval.runner import aggregate, check_thresholds, load_cases, run_agent_case, run_retrieval_case
from btu_rag.graph import Agent


def test_fact_recall_matches_whole_numbers_and_alternatives():
    assert m.fact_recall("It has 8 ECTS.", ["8"]) == 1.0
    assert m.fact_recall("It has 18 ECTS.", ["8"]) == 0.0
    assert m.fact_recall("Taught in Deutsch.", ["German|Deutsch"]) == 1.0
    assert m.fact_recall("anything", []) == 1.0


def test_ranking_metrics():
    assert m.reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3
    assert m.hit_at_k(["a", "b"], {"b"}, 1) == 0.0
    assert m.recall_at_k(["a", "b"], {"b", "z"}, 5) == 0.5


def test_citation_precision():
    assert m.citation_precision(["1", "2"], ["1"]) == 0.5
    assert m.citation_precision([], ["1"]) == 0.0


def test_golden_set_is_well_formed(catalogue):
    cases = load_cases()
    assert len({c.id for c in cases}) == len(cases) >= 30
    for c in cases:
        assert c.expect in {"answer", "no_answer"}
        assert all(i in catalogue.by_id for i in c.relevant), c.id
        assert bool(c.relevant) == (c.expect == "answer"), c.id


def test_retrieval_eval_meets_its_gates(catalogue):
    """Regression gate: fails if a retrieval change makes the golden set worse."""
    agent = Agent(catalogue, llm=None)
    summary = aggregate([run_retrieval_case(agent, c) for c in load_cases()])
    assert check_thresholds(summary, "retrieval") == []


def test_agent_case_scores_answer_and_judge(catalogue):
    llm = FakeLLM(plan={"search_query": "compiler construction"},
                  answer="Compiler Construction is worth 8 ECTS [12350].")

    class Judge(FakeLLM):
        def json(self, system, prompt, schema, *, max_tokens=2000):
            return {"claims": [{"claim": "8 ECTS", "supported": True, "reason": "credits_ects is 8"},
                               {"claim": "taught in English", "supported": False, "reason": "record says German"}]}

    case = next(c for c in load_cases() if c.id == "credits-compiler")
    result = run_agent_case(Agent(catalogue, llm), case, judge=Judge())
    assert result["fact_recall"] == 1.0 and result["citation_precision"] == 1.0
    assert result["faithfulness"] == 0.5 and result["unsupported_claims"][0]["claim"] == "taught in English"


def test_judge_with_no_claims_counts_as_faithful():
    class Empty(FakeLLM):
        def json(self, *a, **k):
            return {"claims": []}

    assert judge_faithfulness(Empty(), "{}", "q", "The catalogue does not say.")["faithfulness"] == 1.0
