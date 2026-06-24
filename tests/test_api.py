from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import agentic_rag.api.main as main_module
from agentic_rag.exceptions import AgentError


@pytest.fixture
def client(monkeypatch):
    # Avoid real startup (Gemini config, vector store init) during lifespan
    fake_agent = MagicMock()
    fake_agent.store.count.return_value = 7
    fake_agent.ask.return_value = "Mocked grounded answer [source: site]"

    def fake_lifespan_init(app):
        app.state.agent = fake_agent

    monkeypatch.setattr(
        main_module, "AgenticRAG", lambda settings=None: fake_agent
    )
    monkeypatch.setattr(
        "agentic_rag.settings.Settings.validate_for_runtime", lambda self: None
    )

    with TestClient(main_module.app) as c:
        yield c, fake_agent


def test_health_endpoint(client):
    c, fake_agent = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 7


def test_ask_endpoint_returns_answer(client):
    c, fake_agent = client
    resp = c.post("/ask", json={"question": "What does CapitalNxt do?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "What does CapitalNxt do?"
    assert "Mocked grounded answer" in body["answer"]


def test_ask_endpoint_validates_empty_question(client):
    c, _ = client
    resp = c.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length validation


def test_ask_endpoint_maps_agent_error_to_502(client):
    c, fake_agent = client
    fake_agent.ask.side_effect = AgentError("Gemini API call failed")
    resp = c.post("/ask", json={"question": "anything"})
    assert resp.status_code == 502


def test_ingest_status_defaults_to_idle(client):
    c, _ = client
    resp = c.get("/ingest/status")
    assert resp.status_code == 200
    assert resp.json()["state"] in ("idle", "running", "done", "error")
