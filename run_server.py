#!/usr/bin/env python3
import uvicorn
from agentic_rag.settings import get_settings
from agentic_rag.logging_config import configure_logging


def main():
    settings = get_settings()
    configure_logging()
    uvicorn.run(
        "agentic_rag.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.UVICORN_WORKERS,
        reload=settings.UVICORN_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
