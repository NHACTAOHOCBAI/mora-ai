import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model_name: str = "gemini-3.1-flash-lite"
    gemini_evaluator_model_name: str = "gemini-3.1-flash-lite"
    gemini_evaluator_embeddings_model_name: str = "models/gemini-embedding-2"
    gemini_temperature: float = 0.0

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
