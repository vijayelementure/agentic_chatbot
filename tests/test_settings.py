import pytest
from pydantic import ValidationError

from agentic_rag.settings import Settings


def test_defaults_load():
    s = Settings(gemini_api_key="x")
    assert s.website_url == "https://capitalnxt.co.in/"
    assert s.gemini_chat_model == "gemini-2.0-flash"
    assert s.max_crawl_pages == 50


def test_validate_for_runtime_raises_without_api_key():
    s = Settings(gemini_api_key="")
    with pytest.raises(RuntimeError):
        s.validate_for_runtime()


def test_validate_for_runtime_passes_with_api_key():
    s = Settings(gemini_api_key="real-key")
    s.validate_for_runtime()  # should not raise


@pytest.mark.parametrize("field", ["max_crawl_pages", "chunk_size", "embedding_batch_size"])
def test_positive_int_validation(field):
    with pytest.raises(ValidationError):
        Settings(gemini_api_key="x", **{field: 0})
