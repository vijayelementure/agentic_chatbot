from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import google.genai as genai
from google.genai.types import EmbedContentConfig
from tenacity import retry, stop_after_attempt, wait_exponential

from agentic_rag.exceptions import EmbeddingError, VectorStoreError
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, str]
    distance: float


class GeminiEmbedder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.client = genai.Client(api_key=self.settings.GEMINI_API_KEY)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _embed_one(self, text: str, task_type: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.settings.GEMINI_EMBED_MODEL,
            contents=text,
            config=EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values

    def embed(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        try:
            return [self._embed_one(t, task_type) for t in texts]
        except Exception as e:
            raise EmbeddingError(f"Gemini embedding request failed: {e}") from e


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[TextChunk]) -> int:
        pass

    @abstractmethod
    def query(self, query_text: str, n_results: int | None = None) -> list[RetrievedChunk]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass


class PineconeVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings | None = None, embedder: GeminiEmbedder | None = None):
        self.settings = settings or get_settings()
        self.embedder = embedder or GeminiEmbedder(self.settings)
        try:
            from pinecone import Pinecone, ServerlessSpec
            self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
            if self.settings.PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
                # Get sample embedding to determine correct dimension
                sample_embedding = self.embedder.embed(["test"])[0]
                self.pc.create_index(
                    name=self.settings.PINECONE_INDEX_NAME,
                    dimension=len(sample_embedding),
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.settings.PINECONE_ENVIRONMENT,
                    ),
                )
            self.index = self.pc.Index(self.settings.PINECONE_INDEX_NAME)
        except ImportError:
            raise VectorStoreError("Pinecone not installed: install with `pip install pinecone`")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone: {e}") from e

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        batch_size = self.settings.EMBEDDING_BATCH_SIZE
        written = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self.embedder.embed(texts, task_type="retrieval_document")
            vectors = [
                {"id": c.id, "values": e, "metadata": {**c.metadata, "text": c.text}}
                for c, e in zip(batch, embeddings)
            ]
            try:
                self.index.upsert(vectors=vectors)
            except Exception as e:
                raise VectorStoreError(f"Failed to upsert batch: {e}") from e
            written += len(batch)
            logger.info("Embedded + upserted %d/%d chunks", written, len(chunks))
        return written

    def query(self, query_text: str, n_results: int | None = None) -> list[RetrievedChunk]:
        n_results = n_results or self.settings.RETRIEVAL_TOP_K
        try:
            q_embedding = self.embedder.embed([query_text], task_type="retrieval_query")[0]
            results = self.index.query(
                vector=q_embedding,
                top_k=n_results,
                include_metadata=True,
            )
        except EmbeddingError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Query failed: {e}") from e
        return [
            RetrievedChunk(
                text=m["metadata"]["text"],
                metadata={k: v for k, v in m["metadata"].items() if k != "text"},
                distance=1 - m["score"],
            )
            for m in results["matches"]
        ]

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]


def get_vector_store(settings: Settings | None = None) -> BaseVectorStore:
    settings = settings or get_settings()
    return PineconeVectorStore(settings)

VectorStore = PineconeVectorStore
