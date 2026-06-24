from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.agent import AgenticRAG
from agentic_rag.vector_store import RetrievedChunk


def _make_part(function_call=None, text=None):
    part = MagicMock()
    part.function_call = function_call
    part.text = text
    return part


def _make_response(parts, text=None):
    resp = MagicMock()
    candidate = MagicMock()
    candidate.content.parts = parts
    resp.candidates = [candidate]
    resp.text = text
    return resp


@patch("agentic_rag.agent.genai")
def test_agent_answers_without_tool_call(mock_genai, settings):
    final_response = _make_response(
        [_make_part(function_call=None, text="Direct answer")], text="Direct answer"
    )

    mock_chat = MagicMock()
    mock_chat.send_message.return_value = final_response
    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai.GenerativeModel.return_value = mock_model

    store = MagicMock()
    agent = AgenticRAG(settings=settings, store=store)

    answer = agent.ask("What is 2+2?")
    assert answer == "Direct answer"
    store.query.assert_not_called()


@patch("agentic_rag.agent.genai")
def test_agent_calls_search_tool_then_answers(mock_genai, settings):
    fc = MagicMock()
    fc.name = "search_knowledge_base"
    fc.args = {"query": "services offered"}

    tool_call_response = _make_response([_make_part(function_call=fc)])
    final_response = _make_response(
        [_make_part(function_call=None, text="Grounded answer")], text="Grounded answer"
    )

    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = [tool_call_response, final_response]
    mock_model = MagicMock()
    mock_model.start_chat.return_value = mock_chat
    mock_genai.GenerativeModel.return_value = mock_model
    mock_genai.protos.Content.return_value = MagicMock()
    mock_genai.protos.Part.return_value = MagicMock()
    mock_genai.protos.FunctionResponse.return_value = MagicMock()

    store = MagicMock()
    store.query.return_value = [
        RetrievedChunk(
            text="We offer X, Y, Z.", metadata={"source": "site", "title": "Services"}, distance=0.1
        )
    ]

    agent = AgenticRAG(settings=settings, store=store)
    answer = agent.ask("What services are offered?")

    assert answer == "Grounded answer"
    store.query.assert_called_once()
    assert mock_chat.send_message.call_count == 2


def test_ask_rejects_empty_question(settings):
    with patch("agentic_rag.agent.genai"):
        agent = AgenticRAG(settings=settings, store=MagicMock())
        with pytest.raises(ValueError):
            agent.ask("   ")
