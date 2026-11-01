from typing import Optional
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5-mini"
    openai_parser_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout: float = 60.0
    openai_max_retries: int = 2
    retrieval_enabled: bool = False

    # Qdrant Vector Database
    qdrant_url: Optional[str] = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: Optional[str] = None

    # Retrieval & Embedding
    retrieval_top_k: int = 5
    retrieval_dense_limit: int = 20
    retrieval_sparse_limit: int = 20

    # Credentials belong to the AI service, independent of process cwd.
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
