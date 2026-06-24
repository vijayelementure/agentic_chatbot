import shutil

import pytest

from agentic_rag.vector_store import GeminiEmbedder, TextChunk, VectorStore


class FakeEmbedder(GeminiEmbedder):
    def __init__(self):
        pass  # skip genai.configure

    def embed(self, texts, task_type="retrieval_document"):
        # deterministic fake embeddings based on text length
        return [[float(len(t) % 7), 0.0, 1.0] for t in texts]


@pytest.fixture
def vector_store(settings, tmp_path):
    settings.chroma_db_dir = str(tmp_path / "chroma")
    store = VectorStore(settings=settings, embedder=FakeEmbedder())
    yield store
    shutil.rmtree(settings.chroma_db_dir, ignore_errors=True)


def test_add_and_query_chunks(vector_store):
    chunks = [
        TextChunk(id="1", text="hello world", metadata={"source": "a", "title": "A"}),
        TextChunk(id="2", text="goodbye world", metadata={"source": "b", "title": "B"}),
    ]
    written = vector_store.add_chunks(chunks)
    assert written == 2
    assert vector_store.count() == 2

    results = vector_store.query("hello", n_results=2)
    assert len(results) == 2
    sources = {r.metadata["source"] for r in results}
    assert sources == {"a", "b"}


def test_upsert_overwrites_existing_id(vector_store):
    vector_store.add_chunks([TextChunk(id="1", text="v1", metadata={"source": "a", "title": "A"})])
    vector_store.add_chunks([TextChunk(id="1", text="v2", metadata={"source": "a", "title": "A"})])
    assert vector_store.count() == 1
