"""HTTP API. Run with: uvicorn btu_rag.api:app --reload"""

import time
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .catalogue import cached_catalogue
from .config import Settings, has_credentials
from .graph import Agent
from .llm import ClaudeLLM
from .retriever import Filters


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


class Source(BaseModel):
    module_id: str
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    mode: Literal["generated", "extractive", "no_evidence", "out_of_scope"]
    citations: list[str]
    sources: list[Source]
    trace: list[str]
    latency_ms: int


def create_app(agent: Agent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is not None:
            app.state.agent = agent
        else:
            settings = Settings.from_env()
            llm = ClaudeLLM(settings) if has_credentials() else None
            app.state.agent = Agent(cached_catalogue(str(settings.catalogue_path)), llm, settings.top_k)
        yield

    app = FastAPI(title="BTU RAG agent", version="0.1.0", lifespan=lifespan)

    def get_agent(request: Request) -> Agent:
        return request.app.state.agent

    @app.get("/health")
    def health(a: Agent = Depends(get_agent)) -> dict:
        return {
            "status": "ok",
            "modules": len(a.catalogue.modules),
            "catalogue_generated_at": a.catalogue.generated_at,
            "answer_mode": "generated" if a.llm else "extractive (no API key)",
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, a: Agent = Depends(get_agent)) -> AskResponse:
        started = time.perf_counter()
        try:
            state = a.ask(body.question)
        except anthropic.RateLimitError:
            raise HTTPException(429, "The model is rate limited. Try again shortly.")
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"Model request failed ({e.status_code}).")
        except anthropic.APIConnectionError:
            raise HTTPException(502, "Could not reach the model provider.")
        return AskResponse(
            answer=state["answer"],
            mode=state["mode"],
            citations=state["citations"],
            sources=[Source(module_id=h.module.id, title=h.module.title_en, url=h.module.url)
                     for h in state["hits"]],
            trace=state.get("trace", []),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

    @app.get("/modules/{module_id}")
    def module(module_id: str, a: Agent = Depends(get_agent)) -> dict:
        found = a.catalogue.by_id.get(module_id)
        if found is None:
            raise HTTPException(404, f"No module {module_id} in the catalogue.")
        return found.evidence()

    @app.get("/search")
    def search(
        q: str = Query("", max_length=200),
        level: Literal["bachelor", "master"] | None = None,
        language: Literal["English", "German"] | None = None,
        semester: Literal["winter", "summer"] | None = None,
        k: int = Query(10, ge=1, le=50),
        a: Agent = Depends(get_agent),
    ) -> list[dict]:
        hits = a.retriever.search(q, Filters(level=level, language=language, semester=semester), k)
        return [{"score": None if h.exact_id else round(h.score, 3), "coverage": round(h.coverage, 3),
                 **h.module.evidence()} for h in hits]

    return app


app = create_app()
