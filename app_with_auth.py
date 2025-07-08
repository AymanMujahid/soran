import time
import uuid
import logging
import os
import secrets
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from email_validator import validate_email, EmailNotValidError

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    APP_NAME: str = "NOXUS Core"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    SECRET_KEY: str = "your-development-secret-key-change-in-production"
    DATABASE_URL: str = "sqlite:///./noxus.db"
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"]

settings = Settings()

# Database Models
Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime)
    api_key = Column(String(255), unique=True, index=True)
    
    bots = relationship("Bot", back_populates="owner", cascade="all, delete-orphan")

class Bot(Base):
    __tablename__ = "bots"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="active", nullable=False)
    personality = Column(Text)
    system_prompt = Column(Text)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1000)
    model_name = Column(String(100), default="mistral-7b")
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    usage_count = Column(Integer, default=0)
    settings = Column(JSON)
    
    owner = relationship("User", back_populates="bots")

# Database setup
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"api/v1/auth/login")

# Security functions
class SecurityManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=1440)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Rate limiting
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int = 60, window: int = 60) -> bool:
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window)
        
        if key not in self.requests:
            self.requests[key] = []
        
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        if len(self.requests[key]) >= limit:
            return False
        
        self.requests[key].append(now)
        return True

rate_limiter = RateLimiter()

# Authentication dependency
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = SecurityManager.verify_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class BotCreate(BaseModel):
    name: str
    description: Optional[str] = None
    personality: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class BotResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    usage_count: int

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

# Validation functions
def validate_email_format(email: str) -> str:
    try:
        valid = validate_email(email)
        return valid.email
    except EmailNotValidError:
        raise ValueError("Invalid email format")

def validate_password_strength(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not any(c.islower() for c in password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain at least one number")
    return password

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting NOXUS Core with Authentication...")
    logger.info("📊 Database tables created successfully")
    yield
    logger.info("🛑 Shutting down NOXUS Core...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NOXUS Core - AI Bot Platform with Authentication",
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
    client_ip = request.client.host
    
    # Rate limiting
    if not rate_limiter.is_allowed(client_ip, limit=100):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s - IP: {client_ip}"
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

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to NOXUS Core - AI Bot Platform",
        "version": settings.VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "status": "operational",
        "features": ["Authentication", "User Management", "Bot Creation", "Database Integration"]
    }

@app.get("/health")
async def health_check():
    db_status = "connected" if os.path.exists("./noxus.db") else "ready"
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "database": db_status,
        "components": {
            "authentication": "active",
            "database": "active",
            "api": "active"
        }
    }

# Authentication endpoints
@app.post(f"{settings.API_PREFIX}/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        # Validate email and password
        validated_email = validate_email_format(user_data.email)
        validated_password = validate_password_strength(user_data.password)
        
        # Check if user already exists
        if db.query(User).filter(User.email == validated_email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        if db.query(User).filter(User.username == user_data.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")
        
        # Create user
        hashed_password = SecurityManager.hash_password(validated_password)
        api_key = f"nxs_{secrets.token_urlsafe(32)}"
        
        user = User(
            email=validated_email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            api_key=api_key
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create access token
        access_token = SecurityManager.create_access_token(
            data={"sub": user.id, "email": user.email}
        )
        
        user_response = UserResponse.from_orm(user)
        
        logger.info(f"✅ New user registered: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post(f"{settings.API_PREFIX}/auth/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        # Find user by email or username
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user:
            user = db.query(User).filter(User.username == form_data.username).first()
        
        # Verify user and password
        if not user or not SecurityManager.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email/username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        access_token = SecurityManager.create_access_token(
            data={"sub": user.id, "email": user.email}
        )
        
        user_response = UserResponse.from_orm(user)
        
        logger.info(f"✅ User logged in: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=user_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get(f"{settings.API_PREFIX}/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return UserResponse.from_orm(current_user)

# Bot management endpoints
@app.post(f"{settings.API_PREFIX}/bots", response_model=BotResponse)
async def create_bot(
    bot_data: BotCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        bot = Bot(
            owner_id=current_user.id,
            name=bot_data.name,
            description=bot_data.description,
            personality=bot_data.personality,
            temperature=bot_data.temperature,
            max_tokens=bot_data.max_tokens
        )
        
        db.add(bot)
        db.commit()
        db.refresh(bot)
        
        logger.info(f"✅ Bot created: {bot.name} by user {current_user.email}")
        
        return BotResponse.from_orm(bot)
        
    except Exception as e:
        logger.error(f"Bot creation error: {e}")
        raise HTTPException(status_code=500, detail="Bot creation failed")

@app.get(f"{settings.API_PREFIX}/bots")
async def get_user_bots(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        bots = db.query(Bot).filter(Bot.owner_id == current_user.id).all()
        
        return {
            "bots": [BotResponse.from_orm(bot) for bot in bots],
            "total": len(bots)
        }
        
    except Exception as e:
        logger.error(f"Error fetching user bots: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bots")

if __name__ == "__main__":
    print("🚀 Starting NOXUS Core with Full Authentication...")
    print(f"📖 API Documentation: http://localhost:8000{settings.API_PREFIX}/docs")
    print(f"🔍 Health Check: http://localhost:8000/health")
    print(f"🔐 Authentication: Registration and Login available")
    print(f"🤖 Bot Management: Create and manage your AI bots")
    
    uvicorn.run(
        "app_with_auth:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )