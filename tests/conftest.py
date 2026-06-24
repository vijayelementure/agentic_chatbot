import pytest

from agentic_rag.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        gemini_api_key="test-key",
        google_drive_folder_id="test-folder",
        website_url="https://example.com/",
        chroma_db_dir="./data/test_chroma_db",
    )
