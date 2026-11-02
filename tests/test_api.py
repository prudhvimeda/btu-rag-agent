import pytest
from fastapi.testclient import TestClient

from btu_rag.api import create_app
from btu_rag.graph import Agent


@pytest.fixture(scope="module")
def client(catalogue):
    with TestClient(create_app(Agent(catalogue, llm=None))) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["modules"] > 3000


def test_ask_returns_answer_with_sources(client):
    body = client.post("/ask", json={"question": "Is IT Security Law taught in English?"}).json()
    assert body["mode"] == "extractive"
    assert body["sources"][0] == {"module_id": "11891", "title": "IT Security Law",
                                  "url": "https://btu-mofi.com/modules/11891/"}
    assert body["latency_ms"] >= 0


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_module_lookup(client):
    assert client.get("/modules/11101").json()["credits_ects"] == 8
    assert client.get("/modules/00000").status_code == 404


def test_search_applies_filters(client):
    rows = client.get("/search", params={"q": "security", "language": "English", "level": "master"}).json()
    assert rows and all(r["teaching_language"] == "English" and "master" in r["levels"] for r in rows)
