import pytest

from btu_rag.catalogue import cached_catalogue
from btu_rag.config import ROOT


@pytest.fixture(scope="session")
def catalogue():
    return cached_catalogue(str(ROOT / "data" / "mofi-catalog.json"))


class FakeLLM:
    """Scripted stand-in for ClaudeLLM: records calls, answers from a fixed plan."""

    def __init__(self, plan=None, rewrite="", answer="", refuse_answer=False):
        self.plan = plan or {}
        self.rewrite = rewrite
        self.answer = answer
        self.refuse_answer = refuse_answer
        self.calls = []

    def json(self, system, prompt, schema, *, max_tokens=2000):
        self.calls.append(("json", prompt))
        return {"in_scope": True, "search_query": "", "module_ids": [], "level": None, "language": None,
                "semester": None, "programme": None, **self.plan}

    def text(self, system, prompt, *, max_tokens=2000):
        self.calls.append(("text", prompt))
        if system.startswith("A catalogue search"):
            return self.rewrite
        if self.refuse_answer:
            from btu_rag.llm import LLMRefusal
            raise LLMRefusal("declined")
        return self.answer
