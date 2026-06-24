import pytest

from agentic_rag.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("hello world", chunk_size=100, overlap=10)
    assert chunks == ["hello world"]


def test_long_text_is_split_with_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # consecutive chunks should overlap
    assert chunks[0][-50:] == text[250:300]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=10, overlap=-1)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)
