from __future__ import annotations
from agentic_rag.agent import AgenticRAG
from agentic_rag.exceptions import AgentError
from agentic_rag.logging_config import configure_logging
from agentic_rag.settings import get_settings


def main() -> None:
    configure_logging()
    settings = get_settings()
    settings.validate_for_runtime()

    agent = AgenticRAG(settings=settings)
    print(f"Agentic RAG ready. Knowledge base has {agent.store.count()} chunks.")
    print("Type your question (or 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        try:
            answer = agent.ask(question)
            print(f"\nAgent: {answer}\n")
        except AgentError as e:
            print(f"\n[error] {e}\n")


if __name__ == "__main__":
    main()
