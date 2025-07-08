import time
import uuid
import logging
import os
import secrets
import hashlib
import jwt
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
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
    VERSION: str = "1.5.0-Minimal"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    SECRET_KEY: str = "your-development-secret-key-change-in-production"
    DATABASE_URL: str = "sqlite:///./noxus.db"
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"]

settings = Settings()

# Simple Knowledge Storage (without AI for now)
class SimpleKnowledgeStore:
    def __init__(self):
        self.storage_path = Path("data/knowledge")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.bot_knowledge = {}
    
    def add_text(self, bot_id: str, text: str, metadata: Dict = None):
        if bot_id not in self.bot_knowledge:
            self.bot_knowledge[bot_id] = []
        
        knowledge_item = {
            'id': str(uuid.uuid4()),
            'text': text,
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat()
        }
        
        self.bot_knowledge[bot_id].append(knowledge_item)
        self._save_to_file(bot_id)
        return True
    
    def search_text(self, bot_id: str, query: str, limit: int = 3):
        if bot_id not in self.bot_knowledge:
            return []
        
        # Simple keyword search
        query_words = query.lower().split()
        results = []
        
        for item in self.bot_knowledge[bot_id]:
            text_lower = item['text'].lower()
            score = 0
            
            for word in query_words:
                if word in text_lower:
                    score += text_lower.count(word)
            
            if score > 0:
                results.append({
                    'text': item['text'],
                    'score': score,
                    'metadata': item['metadata']
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def get_stats(self, bot_id: str):
        count = len(self.bot_knowledge.get(bot_id, []))
        return {
            'total_items': count,
            'bot_id': bot_id
        }
    
    def _save_to_file(self, bot_id: str):
        try:
            file_path = self.storage_path / f"{bot_id}.json"
            with open(file_path, 'w') as f:
                json.dump(self.bot_knowledge[bot_id], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
    
    def _load_from_file(self, bot_id: str):
        try:
            file_path = self.storage_path / f"{bot_id}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    self.bot_knowledge[bot_id] = json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")

# Global knowledge store
knowledge_store = SimpleKnowledgeStore()

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
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    last_login = Column(DateTime)
    
    bots = relationship("Bot", back_populates="owner", cascade="all, delete-orphan")

class Bot(Base):
    __tablename__ = "bots"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="active", nullable=False)
    personality = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    usage_count = Column(Integer, default=0)
    knowledge_count = Column(Integer, default=0)
    
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
    if user is None or not user.is_active:
        raise credentials_exception
    
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

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting NOXUS Core Minimal Version...")
    logger.info("📊 Database initialized successfully")
    yield
    logger.info("🛑 Shutting down NOXUS Core...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="NOXUS Core - Minimal Working Version (No Dependencies Issues)",
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

# Basic endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to NOXUS Core - Minimal Working Version",
        "version": settings.VERSION,
        "docs": f"{settings.API_PREFIX}/docs",
        "status": "operational",
        "features": [
            "✅ Authentication & User Management (Working)",
            "✅ Bot Creation & Management (Working)",
            "✅ Simple Knowledge Storage (Working)",
            "✅ Basic Chat System (Working)",
            "✅ No Dependency Conflicts (Working)",
            "⏳ AI Embeddings (Coming Next)",
            "⏳ Vector Database (Coming Next)"
        ],
        "note": "This minimal version avoids all dependency conflicts and provides core functionality"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "database": "connected" if os.path.exists("./noxus.db") else "ready",
        "dependencies": "minimal (no conflicts)",
        "components": {
            "authentication": "active",
            "bot_management": "active",
            "knowledge_storage": "active (simple)",
            "chat_system": "active (basic)"
        }
    }

# Authentication endpoints
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
        
        user = User(
            email=validated_email,
            username=user_data.username,
            hashed_password=hashed_password,
            full_name=user_data.full_name
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        access_token = SecurityManager.create_access_token(data={"sub": user.id, "email": user.email})
        user_response = UserResponse.from_orm(user)
        
        logger.info(f"✅ User registered: {user.email}")
        
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

# Bot management
@app.post(f"{settings.API_PREFIX}/bots", response_model=BotResponse)
async def create_bot(bot_data: BotCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        bot = Bot(
            owner_id=current_user.id,
            name=bot_data.name,
            description=bot_data.description,
            personality=bot_data.personality
        )
        
        db.add(bot)
        db.commit()
        db.refresh(bot)
        
        # Initialize knowledge store for this bot
        knowledge_store._load_from_file(bot.id)
        
        logger.info(f"✅ Bot created: {bot.name} by {current_user.email}")
        
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
        logger.error(f"Error fetching bots: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bots")

# Knowledge management
@app.post(f"{settings.API_PREFIX}/bots/{{bot_id}}/knowledge")
async def add_knowledge(
    bot_id: str,
    text: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify bot ownership
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Add to knowledge store
        success = knowledge_store.add_text(
            bot_id=bot_id,
            text=text,
            metadata={'added_by': current_user.username, 'source': 'manual'}
        )
        
        if success:
            # Update bot knowledge count
            bot.knowledge_count += 1
            db.commit()
            
            logger.info(f"✅ Knowledge added to bot {bot_id}")
            
            return {
                "message": "Knowledge added successfully",
                "bot_id": bot_id,
                "text_length": len(text)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add knowledge")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding knowledge: {e}")
        raise HTTPException(status_code=500, detail="Failed to add knowledge")

# Simple chat
@app.post(f"{settings.API_PREFIX}/bots/{{bot_id}}/chat")
async def chat_with_bot(
    bot_id: str,
    message: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify bot ownership
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        # Search knowledge
        results = knowledge_store.search_text(bot_id, message)
        
        # Generate simple response
        if results:
            context = results[0]['text'][:200] + "..." if len(results[0]['text']) > 200 else results[0]['text']
            response = f"Based on what I know: {context}"
            
            if len(results) > 1:
                response += f"\n\nI also found {len(results)-1} other relevant pieces of information."
        else:
            response = f"I don't have specific information about '{message}' yet. Please add some knowledge to help me learn!"
        
        # Update usage count
        bot.usage_count += 1
        db.commit()
        
        logger.info(f"✅ Chat response generated for bot {bot_id}")
        
        return {
            "response": response,
            "sources_found": len(results),
            "bot_name": bot.name,
            "personality": bot.personality
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat failed")

# Bot stats
@app.get(f"{settings.API_PREFIX}/bots/{{bot_id}}/stats")
async def get_bot_stats(bot_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        bot = db.query(Bot).filter(Bot.id == bot_id, Bot.owner_id == current_user.id).first()
        if not bot:
            raise HTTPException(status_code=404, detail="Bot not found")
        
        knowledge_stats = knowledge_store.get_stats(bot_id)
        
        return {
            "bot_info": {
                "id": bot.id,
                "name": bot.name,
                "status": bot.status,
                "created_at": bot.created_at.isoformat(),
                "usage_count": bot.usage_count,
                "knowledge_count": bot.knowledge_count
            },
            "knowledge_storage": knowledge_stats,
            "version": "minimal (no AI dependencies)"
        }
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch stats")

if __name__ == "__main__":
    print("🚀 Starting NOXUS Core Minimal Version...")
    print(f"📖 API Documentation: http://localhost:8000{settings.API_PREFIX}/docs")
    print(f"🔍 Health Check: http://localhost:8000/health")
    print(f"✅ No Dependency Conflicts - Works Out of the Box!")
    print(f"🔐 Authentication: Registration and Login")
    print(f"🤖 Bot Management: Create and manage bots")
    print(f"📚 Knowledge Storage: Add and search text")
    print(f"💬 Basic Chat: Simple keyword-based responses")
    print(f"🎯 Ready to add AI features step by step!")
    
    uvicorn.run(
        "app_minimal_working:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )