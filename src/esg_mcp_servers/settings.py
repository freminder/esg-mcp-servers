import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── PostgreSQL ──────────────────────────────────────────────────────────
    POSTGRES_DSN: str = "postgresql://esg_user:password@localhost:5432/esg_platform"

    # ── MongoDB ─────────────────────────────────────────────────────────────
    MONGODB_URI: str = "mongodb://localhost:27017/"
    MONGODB_DB: str = "esg_platform"

    # ── Anthropic / Claude ──────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    CLAUDE_MAX_TOKENS: int = 4096
    CLAUDE_TEMPERATURE: float = 0.1

    # ── HuggingFace ────────────────────────────────────────────────────────
    HF_TOKEN: str = ""

    # ── Embeddings ──────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    EMBEDDING_DIMENSIONS: int = 1024
    EMBEDDING_DEVICE: str = "auto"
    EMBEDDING_BATCH_SIZE: int = 128
    MODEL_CACHE_DIR: str = "./model_cache"

    # ── Crawler ─────────────────────────────────────────────────────────────
    SEARCH_LIMIT: int = 20
    DOWNLOAD_FOLDER: str = "./downloads"
    CRAWLER_DELAY_SECONDS: int = 3
    CRAWLER_MAX_WORKERS: int = 5

    # ── PDF Processing ───────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    PDF_VERIFIER_MODEL_PATH: str = "./model_artifacts/Model.pkl"
    PDF_VERIFIER_VECTORIZER_PATH: str = "./model_artifacts/Vecteur.pkl"

    # ── Query Cache ──────────────────────────────────────────────────────────
    CACHE_TTL_DAYS: int = 7
    CACHE_SIMILARITY_THRESHOLD: float = 0.85

    # ── Vector Search ────────────────────────────────────────────────────────
    SIMILARITY_THRESHOLD: float = 0.5
    DEFAULT_TOP_K: int = 5

    # ── Regulations ──────────────────────────────────────────────────────────
    EURLEX_BASE_URL: str = "https://eur-lex.europa.eu"
    REGULATION_CHUNK_SIZE: int = 1024
    REGULATION_CHUNK_OVERLAP: int = 128


settings = Settings()
os.environ.setdefault("HF_TOKEN", settings.HF_TOKEN)
