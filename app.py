import time
import uuid
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from config import settings, SECURITY_HEADERS
from database import get_db, init_database, database_health_check, get_connection_stats
from auth import (
    UserManager, SecurityManager, get_current_user, require_active_user,
    require_bot_ownership, log_api_usage, rate_limiter, validate_input_length
)
from models import User, Bot
import logging.config
from config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NOXUS Core application...")
    
    if not init_database():
        logger.error("Failed to initialize database")
        raise RuntimeError("Database initialization failed")
    
    logger.info("NOXUS Core started successfully")
    yield
    
    logger.info("Shutting down NOXUS Core...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NOXUS Core - Advanced AI Bot Platform",
    openapi_url=f"{settings.API_PREFIX}/openapi.json" if settings.DEBUG else None,
    docs_url=f"{settings.API_PREFIX}/docs" if settings.DEBUG else None,
    redoc_url=f"{settings.API_PREFIX}/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else ["localhost", "127.0.0.1"]
)

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    start_time = time.time()
    
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    
    if not rate_limiter.is_allowed(client_ip, settings.RATE_LIMIT_PER_MINUTE, 60):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded"}
        )
    
    response = await call_next(request)
    
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s - "
        f"IP: {client_ip}"
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

# Health and Monitoring Endpoints
@app.get("/health")
async def health_check():
    db_healthy = await database_health_check()
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "database": "healthy" if db_healthy else "unhealthy"
    }

@app.get("/metrics")
async def get_metrics():
    return {
        "database": get_connection_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }

# Authentication Endpoints
@app.post(f"{settings.API_PREFIX}/auth/register")
async def register(
    email: str,
    username: str,
    password: str,
    full_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        email = validate_input_length(email, 255)
        username = validate_input_length(username, 100)
        password = validate_input_length(password, 128)
        
        user_manager = UserManager(db)
        user = user_manager.create_user(email, username, password, full_name)
        
        access_token = SecurityManager.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        refresh_token = SecurityManager.create_refresh_token(
            data={"sub": str(user.id)}
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post(f"{settings.API_PREFIX}/auth/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    try:
        user_manager = UserManager(db)
        user = user_manager.authenticate_user(form_data.username, form_data.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = SecurityManager.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        refresh_token = SecurityManager.create_refresh_token(
            data={"sub": str(user.id)}
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.post(f"{settings.API_PREFIX}/auth/refresh")
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    try:
        payload = SecurityManager.verify_token(refresh_token, "refresh")
        user_id = payload.get("sub")
        
        user_manager = UserManager(db)
        user = user_manager.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        access_token = SecurityManager.create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

@app.get(f"{settings.API_PREFIX}/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "username": current_user.username,
        "full_name": current_user.full_name,
        "is_active": current_user.is_active,
        "subscription_tier": current_user.subscription_tier,
        "created_at": current_user.created_at.isoformat()
    }

# Bot Management Endpoints
@app.post(f"{settings.API_PREFIX}/bots")
async def create_bot(
    name: str,
    description: Optional[str] = None,
    personality: Optional[str] = None,
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db)
):
    try:
        name = validate_input_length(name, 255)
        description = validate_input_length(description, 1000) if description else None
        personality = validate_input_length(personality, 2000) if personality else None
        
        bot = Bot(
            name=name,
            description=description,
            personality=personality,
            owner_id=current_user.id
        )
        
        db.add(bot)
        db.commit()
        db.refresh(bot)
        
        logger.info(f"Bot created: {bot.name} by user {current_user.email}")
        
        return {
            "id": str(bot.id),
            "name": bot.name,
            "description": bot.description,
            "status": bot.status,
            "created_at": bot.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Bot creation error: {e}")
        raise HTTPException(status_code=500, detail="Bot creation failed")

@app.get(f"{settings.API_PREFIX}/bots")
async def get_user_bots(
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db)
):
    try:
        bots = db.query(Bot).filter(Bot.owner_id == current_user.id).all()
        
        return {
            "bots": [
                {
                    "id": str(bot.id),
                    "name": bot.name,
                    "description": bot.description,
                    "status": bot.status,
                    "created_at": bot.created_at.isoformat(),
                    "usage_count": bot.usage_count
                }
                for bot in bots
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching user bots: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bots")

@app.get(f"{settings.API_PREFIX}/bots/{{bot_id}}")
async def get_bot(
    bot_id: str,
    current_user: User = Depends(require_active_user),
    db: Session = Depends(get_db)
):
    try:
        require_bot_ownership(bot_id, current_user, db)
        
        bot = db.query(Bot).filter(Bot.id == bot_id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        return {
            "id": str(bot.id),
            "name": bot.name,
            "description": bot.description,
            "personality": bot.personality,
            "status": bot.status,
            "model_name": bot.model_name,
            "temperature": bot.temperature,
            "max_tokens": bot.max_tokens,
            "created_at": bot.created_at.isoformat(),
            "updated_at": bot.updated_at.isoformat(),
            "usage_count": bot.usage_count,
            "training_data_count": bot.training_data_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching bot {bot_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bot")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=settings.WORKER_PROCESSES,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        reload=settings.DEBUG
    )