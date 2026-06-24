from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agentic_rag.chunking import chunk_text
from agentic_rag.drive_reader import DriveReader
from agentic_rag.exceptions import AgenticRAGError
from agentic_rag.logging_config import get_logger
from agentic_rag.settings import Settings, get_settings
from agentic_rag.vector_store import TextChunk, get_vector_store, BaseVectorStore
from agentic_rag.web_crawler import WebsiteCrawler

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    drive_documents: int
    website_pages: int
    chunks_written: int
    total_chunks_in_store: int


def make_id(source: str, idx: int) -> str:
    h = hashlib.md5(f"{source}-{idx}".encode()).hexdigest()[:10]
    return f"{h}-{idx}"


def run_ingestion(settings: Settings | None = None, store: BaseVectorStore | None = None) -> IngestionResult:
    settings = settings or get_settings()
    settings.validate_for_runtime()
    store = store or get_vector_store(settings)

    all_chunks: list[TextChunk] = []
    drive_doc_count = 0
    page_count = 0

    if settings.GOOGLE_DRIVE_FOLDER_ID:
        logger.info("Reading documents from Google Drive...")
        try:
            docs = DriveReader(settings).fetch_documents()
            drive_doc_count = len(docs)
            for doc in docs:
                for i, chunk in enumerate(chunk_text(doc.text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
                    all_chunks.append(
                        TextChunk(
                            id=make_id(doc.source, i),
                            text=chunk,
                            metadata={"source": doc.source, "title": doc.name, "origin": "gdrive"},
                        )
                    )
        except AgenticRAGError as e:
            logger.warning("Skipping Google Drive ingestion: %s", e)
    else:
        logger.info("No GOOGLE_DRIVE_FOLDER_ID set — skipping Google Drive ingestion.")

    logger.info("Crawling website: %s (max %d pages)...", settings.WEBSITE_URL, settings.MAX_CRAWL_PAGES)
    pages = WebsiteCrawler(settings).crawl_sync()
    page_count = len(pages)
    for page in pages:
        for i, chunk in enumerate(chunk_text(page.text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
            all_chunks.append(
                TextChunk(
                    id=make_id(page.source, i),
                    text=chunk,
                    metadata={"source": page.source, "title": page.url, "origin": "website"},
                )
            )

    logger.info("Total chunks to embed: %d", len(all_chunks))
    chunks_written = 0
    if all_chunks:
        logger.info("Embedding and storing in vector DB (calls the Gemini API)...")
        chunks_written = store.add_chunks(all_chunks)
    else:
        logger.warning("Nothing to ingest.")

    total = store.count()
    logger.info("Ingestion complete. Vector store now has %d chunks total.", total)
    return IngestionResult(
        drive_documents=drive_doc_count,
        website_pages=page_count,
        chunks_written=chunks_written,
        total_chunks_in_store=total,
    )


def main() -> None:
    result = run_ingestion()
    print(
        f"Drive documents: {result.drive_documents} | "
        f"Website pages: {result.website_pages} | "
        f"Chunks written: {result.chunks_written} | "
        f"Total chunks in store: {result.total_chunks_in_store}"
    )


if __name__ == "__main__":
    main()
