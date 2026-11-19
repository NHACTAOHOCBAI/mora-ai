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

    benchmark_judge_api_key: Optional[str] = None
    benchmark_judge_model: str = "gpt-5-mini"
    benchmark_judge_embedding_model: str = "text-embedding-3-small"
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
    ai_usage_vnd_per_usd: float = 26000
    ai_usage_pricing_version: str = "2026-11"
    ai_usage_default_input_per_million_usd: float = 0
    ai_usage_default_cached_input_per_million_usd: float = 0
    ai_usage_default_output_per_million_usd: float = 0
    ai_usage_gpt_5_mini_input_per_million_usd: float = 0.25
    ai_usage_gpt_5_mini_cached_input_per_million_usd: float = 0.025
    ai_usage_gpt_5_mini_output_per_million_usd: float = 2.0
    ai_usage_text_embedding_3_small_input_per_million_usd: float = 0.02

    def model_prices(self):
        return {
            "default": {"input": self.ai_usage_default_input_per_million_usd,
                        "cached_input": self.ai_usage_default_cached_input_per_million_usd,
                        "output": self.ai_usage_default_output_per_million_usd},
            "gpt-5-mini": {"input": self.ai_usage_gpt_5_mini_input_per_million_usd,
                            "cached_input": self.ai_usage_gpt_5_mini_cached_input_per_million_usd,
                            "output": self.ai_usage_gpt_5_mini_output_per_million_usd},
            "text-embedding-3-small": {"input": self.ai_usage_text_embedding_3_small_input_per_million_usd,
                                         "cached_input": 0, "output": 0},
        }

    # Credentials belong to the AI service, independent of process cwd.
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "..", "..", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
