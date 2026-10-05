from typing import Optional
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: Optional[str] = None
    gemini_model_name: str = "gemini-3.1-flash-lite"
    gemini_parser_model_list: str = "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash-lite"
    gemini_evaluator_model_name: str = "gemini-3.1-flash-lite"
    gemini_evaluator_embeddings_model_name: str = "models/gemini-embedding-2"
    gemini_temperature: float = 0.0

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334

    # Retrieval & Embedding
    gemini_embedding_model: str = "models/gemini-embedding-2"
    retrieval_top_k: int = 5
    retrieval_dense_limit: int = 20
    retrieval_sparse_limit: int = 20

    # Cấu hình đọc từ file .env của backend hoặc file .env cục bộ của python
    model_config = SettingsConfigDict(
        env_file=(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "mora-backend", ".env"),
            ".env"
        ),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
