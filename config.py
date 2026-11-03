"""
Application configuration — loaded from .env via pydantic-settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str = "gemini-flash-lite-latest"

    # HuggingFace embedding model (must be a sentence-transformers model)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Directory where FAISS index files are persisted (one sub-folder per collection)
    faiss_persist_dir: str = "./faiss_store"

    # Text splitting: overlap keeps a sentence that straddles two chunks retrievable from either
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Number of source chunks to retrieve
    retriever_top_k: int = 4

    # Uploads larger than this are rejected
    max_upload_mb: int = 20

    # Set to false in corporate networks with self-signed proxy certificates
    ssl_verify: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
