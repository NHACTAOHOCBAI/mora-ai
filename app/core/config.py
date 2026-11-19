import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Provider configuration is populated only by app.core.runtime_config from
    # the Admin control plane.  The aliases deliberately prevent OPENAI_* and
    # BENCHMARK_* values in .env from becoming an emergency fallback.
    openai_api_key: str | None = Field(default=None, validation_alias="MORA_RUNTIME_OPENAI_KEY")
    benchmark_judge_api_key: str | None = Field(default=None, validation_alias="MORA_RUNTIME_BENCHMARK_KEY")
    openai_model: str | None = Field(default=None, validation_alias="MORA_RUNTIME_CHAT_MODEL")
    openai_parser_model: str | None = Field(default=None, validation_alias="MORA_RUNTIME_PARSER_MODEL")
    openai_embedding_model: str | None = Field(default=None, validation_alias="MORA_RUNTIME_EMBEDDING_MODEL")
    benchmark_judge_model: str | None = Field(default=None, validation_alias="MORA_RUNTIME_BENCHMARK_JUDGE_MODEL")
    benchmark_judge_embedding_model: str | None = Field(default=None, validation_alias="MORA_RUNTIME_BENCHMARK_EMBEDDING_MODEL")
    openai_timeout: float = 60.0
    openai_max_retries: int = 2
    backend_url: str = "http://localhost:8080"
    ai_service_internal_key: Optional[str] = None
    runtime_model_prices: dict[str, dict[str, float]] = Field(default_factory=dict, exclude=True)
    retrieval_enabled: bool = False

    benchmark_judge_timeout: float = 45.0
    benchmark_judge_max_retries: int = 1

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

    # Operational analytics. Rates are USD per million tokens and are deliberately
    # configuration, rather than hard-coded provider pricing.
    ai_usage_vnd_per_usd: float | None = Field(default=None, validation_alias="MORA_RUNTIME_VND_PER_USD")
    ai_usage_pricing_version: str | None = Field(default=None, validation_alias="MORA_RUNTIME_PRICING_VERSION")

    def model_prices(self):
        return self.runtime_model_prices

    # Credentials belong to the AI service, independent of process cwd.
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
