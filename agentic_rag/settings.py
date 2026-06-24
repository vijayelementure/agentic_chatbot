
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    GEMINI_API_KEY: str = ""
    GEMINI_CHAT_MODEL: str = "models/gemini-3.1-flash-lite"
    GEMINI_EMBED_MODEL: str = "models/gemini-embedding-001"
    GOOGLE_SERVICE_ACCOUNT_FILE: str = "credentials.json"
    GOOGLE_DRIVE_FOLDER_ID: str = ""
    WEBSITE_URL: str = "https://capitalnxt.co.in/"
    MAX_CRAWL_PAGES: int = 50
    CRAWL_REQUEST_TIMEOUT_SECONDS: int = 60
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 400
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "agentic-rag"
    EMBEDDING_BATCH_SIZE: int = 50
    MAX_TOOL_CALLS: int = 5
    RETRIEVAL_TOP_K: int = 5
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ALLOW_ORIGINS: str = "*"
    API_KEY: str | None = None
    UVICORN_WORKERS: int = 4
    UVICORN_RELOAD: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    QUERY_CACHE_SIZE: int = 1000

    @property
    def cors_origins_list(self) -> list[str]:
        if self.CORS_ALLOW_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ALLOW_ORIGINS.split(",") if origin.strip()]

    @field_validator(
        "MAX_CRAWL_PAGES", "CHUNK_SIZE", "EMBEDDING_BATCH_SIZE", "MAX_TOOL_CALLS", "RETRIEVAL_TOP_K"
    )
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    def validate_for_runtime(self) -> None:
        missing = []
        if not self.GEMINI_API_KEY or self.GEMINI_API_KEY == "your_gemini_api_key_here":
            missing.append("GEMINI_API_KEY")
        if not self.PINECONE_API_KEY or self.PINECONE_API_KEY == "your_pinecone_api_key_here":
            missing.append("PINECONE_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required configuration: {', '.join(missing)}. "
                f"Copy .env.example to .env and fill in the values."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
