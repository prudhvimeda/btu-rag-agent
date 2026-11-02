from types import SimpleNamespace

import pytest

from btu_rag.config import Settings
from btu_rag.llm import ClaudeLLM, LLMRefusal


class RecordingClient:
    """Captures the request ClaudeLLM sends; returns a canned Messages API response."""

    def __init__(self, text="{}", stop_reason="end_turn"):
        self.requests = []
        reply = SimpleNamespace(stop_reason=stop_reason,
                                content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)])
        create = lambda **kw: self.requests.append(kw) or reply  # noqa: E731
        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def settings(**overrides):
    return Settings(**{**Settings.from_env().__dict__, "use_fallbacks": True, "effort": "low", **overrides})


def test_json_uses_structured_output_and_fallbacks():
    client = RecordingClient(text='{"in_scope": true}')
    result = ClaudeLLM(settings(), client=client).json("sys", "q", {"type": "object"})
    request = client.requests[0]
    assert result == {"in_scope": True}
    assert request["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert request["fallbacks"] == "default" and request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["model"] == "claude-opus-5-5"


def test_fallbacks_can_be_switched_off():
    client = RecordingClient(text="hi")
    assert ClaudeLLM(settings(use_fallbacks=False), client=client).text("sys", "q") == "hi"
    assert "fallbacks" not in client.requests[0]


def test_refusal_raises():
    with pytest.raises(LLMRefusal):
        ClaudeLLM(settings(), client=RecordingClient(stop_reason="refusal")).text("sys", "q")
