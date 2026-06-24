
class AgenticRAGError(Exception):
    pass


class ConfigurationError(AgenticRAGError):
    pass


class DriveAccessError(AgenticRAGError):
    pass


class CrawlError(AgenticRAGError):
    pass


class EmbeddingError(AgenticRAGError):
   pass


class VectorStoreError(AgenticRAGError):
    pass


class AgentError(AgenticRAGError):
   pass
