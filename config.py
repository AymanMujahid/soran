import os
import secrets
from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from pathlib import Path

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "NOXUS Core"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = Field(default=False)
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080
    ALGORITHM: str = "HS256"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:3000"])
    ALLOWED_METHODS: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"])
    ALLOWED_HEADERS: List[str] = Field(default=["*"])
    
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_RETRY_ON_TIMEOUT: bool = True
    
    # File Storage
    FILE_STORAGE_PATH: Path = Field(default=Path("data/files"))
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: List[str] = Field(default=[".pdf", ".docx", ".txt", ".xlsx", ".xls"])
    
    # Vector Database
    VECTOR_DB_PATH: Path = Field(default=Path("data/vectors"))
    FAISS_INDEX_FACTORY: str = "Flat"
    EMBEDDING_DIMENSION: int = 768
    
    # AI Models
    BGE_MODEL: str = "BAAI/bge-small-en-v1.5"
    E5_MODEL: str = "intfloat/e5-small-v2"
    LLM_MODEL: str = "mistral-7b-instruct"
    CLASSIFICATION_MODEL: str = "microsoft/DialoGPT-medium"
    
    # OCR/ASR
    TESSERACT_CMD: str = "tesseract"
    WHISPER_MODEL: str = "openai/whisper-base"
    
    # TTS
    TTS_MODEL: str = "tts_models/en/ljspeech/tacotron2-DDC"
    TTS_OUTPUT_PATH: Path = Field(default=Path("data/audio"))
    
    # Performance
    WORKER_PROCESSES: int = 1
    MAX_REQUESTS: int = 1000
    MAX_REQUESTS_JITTER: int = 50
    TIMEOUT: int = 300
    
    # Monitoring
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    SENTRY_DSN: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    @validator("FILE_STORAGE_PATH", "VECTOR_DB_PATH", "TTS_OUTPUT_PATH")
    def create_directories(cls, v):
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        if not v.startswith(("postgresql://", "sqlite:///")):
            raise ValueError("DATABASE_URL must be postgresql:// or sqlite:///")
        return v
    
    @validator("ALLOWED_FILE_TYPES")
    def validate_file_types(cls, v):
        return [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in v]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True

# Global settings instance
settings = Settings()

# Security configurations
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
}

# Database configurations
DATABASE_CONFIG = {
    "pool_size": settings.DATABASE_POOL_SIZE,
    "max_overflow": settings.DATABASE_MAX_OVERFLOW,
    "pool_timeout": settings.DATABASE_POOL_TIMEOUT,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}

# Redis configurations
REDIS_CONFIG = {
    "max_connections": settings.REDIS_MAX_CONNECTIONS,
    "retry_on_timeout": settings.REDIS_RETRY_ON_TIMEOUT,
    "health_check_interval": 30,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": settings.LOG_FORMAT
        },
    },
    "handlers": {
        "default": {
            "level": settings.LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": settings.LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "logs/app.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": settings.LOG_LEVEL,
            "propagate": False
        }
    }
}