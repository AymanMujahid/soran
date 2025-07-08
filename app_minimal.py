import time
import uuid
import logging
import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Basic configuration
class Settings:
    APP_NAME: str = "NOXUS Core"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    SECRET_KEY: str = "your-development-secret-key-change-in-production"
    DATABASE_URL: str = "sqlite:///./noxus.db"
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

settings = Settings()

# Basic database setup
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NOXUS Core minimal application...")
    yield
    logger.info("Shutting down NOXUS Core...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NOXUS Core - Minimal Test Version",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception {error_id}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "message": "NOXUS Core is running successfully!"
    }

# Basic info endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to NOXUS Core",
        "version": settings.VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "status": "operational"
    }

# Test endpoint
@app.get(f"{settings.API_PREFIX}/test")
async def test_endpoint():
    return {
        "message": "API is working perfectly!",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected" if os.path.exists("./noxus.db") else "ready to create"
    }

# Simple authentication test (placeholder)
@app.post(f"{settings.API_PREFIX}/auth/test")
async def test_auth():
    return {
        "message": "Authentication system ready",
        "note": "This is a test endpoint. Full auth will be implemented next."
    }

if __name__ == "__main__":
    print("🚀 Starting NOXUS Core...")
    print(f"📖 API Documentation: http://localhost:8000{settings.API_PREFIX}/docs")
    print(f"🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "app_minimal:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )