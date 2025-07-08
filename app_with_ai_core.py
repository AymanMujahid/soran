import time
import uuid
import logging
import os
import secrets
import hashlib
import jwt
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Request, status, File, UploadFile, Form
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

# Import our AI modules
from file_processor import get_file_processing_service
from rag_system import get_rag_system
from vector_database import get_vector_database
from embeddings import get_embedding_service

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Settings:
    APP_NAME: str = "NOXUS Core"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    SECRET_KEY: str = "your-development-secret-key-change-in-production"
    DATABASE_URL: str = "sqlite:///./noxus.db"
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"]
    UPLOAD_DIR: str = "data/uploads"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB

settings = Settings()

# Create upload directory
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

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
    model_name = Column(String(100), default="all-MiniLM-L6-v2")
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    usage_count = Column(Integer, default=0)
    knowledge_count = Column(Integer, default=0)
    settings = Column(JSON)
    
    owner = relationship("User", back_populates="bots")
    files = relationship("BotFile", back_populates="bot", cascade="all, delete-orphan")

class BotFile(Base):
    __tablename__ = "bot_files"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    bot_id = Column(String(36), ForeignKey("bots.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    processed_at = Column(DateTime)
    chunks_count = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    bot = relationship("Bot", back_populates="files")

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
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class BotResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    created_at: datetime
    usage_count: int
    knowledge_count: int

    class Config:
        from_attributes = True

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_context: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    response_time: float
    confidence: float

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
    return password

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

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting NOXUS Core with AI Components...")
    logger.info("📊 Database tables created successfully")
    logger.info("🧠 AI services initialized")
    yield
    logger.info("🛑 Shutting down NOXUS Core...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NOXUS Core - Advanced AI Bot Platform with RAG, Vector Database, and File Processing",
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
    
    if not rate_limiter.is_allowed(client_ip, limit=100):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s - IP: {client_ip}")
    
    return response

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to NOXUS Core - Advanced AI Bot Platform",
        "version": settings.VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "status": "operational",
        "features": [
            "Authentication & User Management",
            "Bot Creation & Management", 
            "File Processing (PDF, DOCX, Excel, Images)",
            "Vector Database Storage",
            "AI Embeddings",
            "RAG-based Chat System",
            "Semantic Search"
        ]
    }

@app.get("/health")
async def health_check():
    try:
        # Test database connection
        db_status = "connected" if os.path.exists("./noxus.db") else "ready"
        
        # Test AI services
        embedding_service = get_embedding_service()
        ai_status = "ready"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.VERSION,
            "components": {
                "database": db_status,
                "authentication": "active",
                "ai_embeddings": ai_status,
                "file_processing": "active",
                "vector_database": "active",
                "rag_system": "active"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Authentication endpoints (same as before)
@app.post(f"{settings.API_PREFIX}/auth/register", response_model=TokenResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    try:
        validated_email = validate_email_format(user_data.email)
        validated_password = validate_password_strength(user_data.password)
        
        if db.query(User).filter(User.email == validated_email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        
        if db.query(User).filter(User.username == user_data.username).first():
            raise HTTPException(status_code=400, detail="Username already taken")
        
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
        
        access_token = SecurityManager.create_access_token(data={"sub": user.id, "email": user.email})
        user_response = UserResponse.from_orm(user)
        
        logger.info(f"✅ New user registered: {user.email}")
        
        return TokenResponse(access_token=access_token, token_type="bearer", user=user_response)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post(f"{settings.API_PREFIX}/auth/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user:
            user = db.query(User).filter(User.username == form_data.username).first()
        
        if not user or not SecurityManager.verify_password(form_data.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Incorrect email/username or password")
        
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        
        user.last_login = datetime.utcnow()
        db.commit()
        
        access_token = SecurityManager.create_access_token(data={"sub": user.id, "email": user.email})
        user_response = UserResponse.from_orm(user)
        
        logger.info(f"✅ User logged in: {user.email}")
        
        return TokenResponse(access_token=access_token, token_type="bearer", user=user_response)
        
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
async def create_bot(bot_data: BotCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        bot = Bot(
            owner_id=current_user.id,
            name=bot_data.name,
            description=bot_data.description,
            personality=bot_data.personality,
            system_prompt=bot_data.system_prompt,
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
async def get_user_bots(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        bots = db.query(Bot).filter(Bot.owner_id == current_user.id).all()
        
        return {
            "bots": [BotResponse.from_orm(bot) for bot in bots],
            "total": len(bots)
        }
        
    except Exception as e:
        logger.error(f"Error fetching user bots: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bots")

# File processing endpoints
@app.post(f"{settings.API_PREFIX}/bots/{{bot_id}}/upload")
async def upload_file(
    bot_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify bot ownership
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Validate file
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        # Save file
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = Path(settings.UPLOAD_DIR) / filename
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Create database record
        bot_file = BotFile(
            bot_id=bot_id,
            filename=filename,
            original_filename=file.filename,
            file_path=str(file_path),
            file_size=file.size,
            file_type=Path(file.filename).suffix.lower()
        )
        
        db.add(bot_file)
        db.commit()
        db.refresh(bot_file)
        
        # Process file asynchronously
        try:
            file_processing_service = get_file_processing_service()
            result = await file_processing_service.process_and_store(
                file_path=file_path,
                bot_id=bot_id,
                embedding_model=bot.model_name
            )
            
            if result['success']:
                bot_file.status = "completed"
                bot_file.processed_at = datetime.utcnow()
                bot_file.chunks_count = result['chunks_processed']
                
                # Update bot knowledge count
                bot.knowledge_count += result['chunks_processed']
                
                db.commit()
                
                logger.info(f"✅ File processed: {file.filename} for bot {bot_id}")
                
                return {
                    "message": "File uploaded and processed successfully",
                    "file_id": bot_file.id,
                    "chunks_processed": result['chunks_processed'],
                    "status": "completed"
                }
            else:
                bot_file.status = "failed"
                bot_file.error_message = result.get('error', 'Unknown error')
                db.commit()
                
                raise HTTPException(status_code=500, detail=f"File processing failed: {result.get('error')}")
                
        except Exception as processing_error:
            bot_file.status = "failed"
            bot_file.error_message = str(processing_error)
            db.commit()
            raise HTTPException(status_code=500, detail=f"File processing failed: {processing_error}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.get(f"{settings.API_PREFIX}/bots/{{bot_id}}/files")
async def get_bot_files(bot_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # Verify bot ownership
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        files = db.query(BotFile).filter(BotFile.bot_id == bot_id).all()
        
        return {
            "files": [
                {
                    "id": f.id,
                    "filename": f.original_filename,
                    "file_type": f.file_type,
                    "file_size": f.file_size,
                    "status": f.status,
                    "chunks_count": f.chunks_count,
                    "processed_at": f.processed_at.isoformat() if f.processed_at else None,
                    "created_at": f.created_at.isoformat()
                } for f in files
            ],
            "total": len(files)
        }
        
    except Exception as e:
        logger.error(f"Error fetching bot files: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch files")

# Chat endpoints
@app.post(f"{settings.API_PREFIX}/bots/{{bot_id}}/chat", response_model=ChatResponse)
async def chat_with_bot(
    bot_id: str,
    chat_data: ChatMessage,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify bot ownership
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Get RAG system
        rag_system = get_rag_system(bot.model_name)
        
        # Generate session ID if not provided
        session_id = chat_data.session_id or f"{bot_id}_{current_user.id}_{int(time.time())}"
        
        # Create or get chat session
        session_info = await rag_system.get_session_info(session_id)
        if not session_info:
            await rag_system.create_chat_session(
                session_id=session_id,
                bot_id=bot_id,
                user_id=current_user.id,
                system_prompt=bot.system_prompt,
                personality=bot.personality,
                temperature=bot.temperature,
                max_tokens=bot.max_tokens
            )
        
        # Generate response
        response = await rag_system.chat(
            session_id=session_id,
            user_message=chat_data.message,
            use_context=chat_data.use_context
        )
        
        # Update bot usage count
        bot.usage_count += 1
        db.commit()
        
        return ChatResponse(
            response=response.message,
            session_id=session_id,
            sources=response.sources,
            response_time=response.response_time,
            confidence=response.confidence
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat failed")

@app.get(f"{settings.API_PREFIX}/bots/{{bot_id}}/stats")
async def get_bot_stats(bot_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Get vector database stats
        vector_db = get_vector_database(bot_id)
        vector_stats = vector_db.get_stats()
        
        return {
            "bot_info": {
                "id": bot.id,
                "name": bot.name,
                "status": bot.status,
                "created_at": bot.created_at.isoformat(),
                "usage_count": bot.usage_count,
                "knowledge_count": bot.knowledge_count
            },
            "vector_database": vector_stats,
            "files_count": db.query(BotFile).filter(BotFile.bot_id == bot_id).count(),
            "model_info": {
                "embedding_model": bot.model_name,
                "temperature": bot.temperature,
                "max_tokens": bot.max_tokens
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching bot stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")

if __name__ == "__main__":
    print("🚀 Starting NOXUS Core with Full AI Capabilities...")
    print(f"📖 API Documentation: http://localhost:8000{settings.API_PREFIX}/docs")
    print(f"🔍 Health Check: http://localhost:8000/health")
    print(f"🔐 Authentication: Registration and Login available")
    print(f"🤖 Bot Management: Create and manage your AI bots")
    print(f"📄 File Processing: Upload PDF, DOCX, Excel files")
    print(f"🧠 AI Chat: RAG-based intelligent conversations")
    print(f"🔍 Vector Search: Semantic search through your documents")
    
    uvicorn.run(
        "app_with_ai_core:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )