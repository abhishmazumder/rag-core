from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure_ai_foundry_response_endpoint: str | None = None
    azure_ai_foundry_response_model: str | None = None
    azure_ai_foundry_embedding_endpoint: str | None = None
    azure_ai_foundry_embedding_model: str | None = None
    azure_ai_foundry_embedding_dimensions: int | None = Field(default=None, gt=0)
    azure_search_endpoint: str | None = None
    azure_search_index_name: str | None = None
