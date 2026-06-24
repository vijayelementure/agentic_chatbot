import logging
import sys

from agentic_rag.settings import get_settings

configured = False


def configure_logging() -> None:
    global configured
    if configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    if settings.LOG_JSON:
        fmt = (
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    for noisy in (
        "urllib3", "google", "googleapiclient", "chromadb", "httpx", "pinecone"
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("agentic_rag.agent").setLevel(logging.WARNING)

    configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
