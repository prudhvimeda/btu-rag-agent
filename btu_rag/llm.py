"""Thin wrapper over the Anthropic SDK, plus the protocol the graph and judge depend on.

Tests and key-free runs pass their own object with the same two methods.
"""

import json
from typing import Any, Protocol

import anthropic

from .config import Settings


class LLMRefusal(RuntimeError):
    """The model (and its fallback chain) declined the request."""


class LLM(Protocol):
    def text(self, system: str, prompt: str, *, max_tokens: int = 2000) -> str: ...

    def json(self, system: str, prompt: str, schema: dict, *, max_tokens: int = 2000) -> dict: ...


class ClaudeLLM:
    def __init__(self, settings: Settings, model: str | None = None, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.model = model or settings.model
        self.client = client or anthropic.Anthropic()

    def _create(self, system: str, prompt: str, max_tokens: int, output_config: dict) -> Any:
        request = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.settings.effort, **output_config},
        )
        if self.settings.use_fallbacks:
            # On a safety decline the API re-runs the request on a fallback model in the same call.
            response = self.client.beta.messages.create(
                **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        else:
            response = self.client.messages.create(**request)
        if response.stop_reason == "refusal":
            raise LLMRefusal("The model declined to answer this request.")
        return response

    @staticmethod
    def _text_of(response: Any) -> str:
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def text(self, system: str, prompt: str, *, max_tokens: int = 2000) -> str:
        return self._text_of(self._create(system, prompt, max_tokens, {}))

    def json(self, system: str, prompt: str, schema: dict, *, max_tokens: int = 2000) -> dict:
        response = self._create(
            system, prompt, max_tokens, {"format": {"type": "json_schema", "schema": schema}}
        )
        return json.loads(self._text_of(response))
