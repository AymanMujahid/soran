from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    """Generate a UUID string"""
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
    
    # Relationships
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
    
    # Relationships
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
    error_message = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    bot = relationship("Bot", back_populates="files")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    bot_id = Column(String(36), ForeignKey("bots.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"))
    session_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    ended_at = Column(DateTime)
    message_count = Column(Integer, default=0)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(36), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    response_time_ms = Column(Integer)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

# Database utility functions
class DatabaseService:
    def __init__(self, db_session):
        self.db = db_session
    
    def create_user(self, email: str, username: str, hashed_password: str, full_name: str = None, api_key: str = None) -> User:
        """Create a new user in the database"""
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=full_name,
            api_key=api_key
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def get_user_by_email(self, email: str) -> User:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_id(self, user_id: str) -> User:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> User:
        """Get user by username"""
        return self.db.query(User).filter(User.username == username).first()
    
    def create_bot(self, owner_id: str, name: str, description: str = None, **kwargs) -> Bot:
        """Create a new bot"""
        bot = Bot(
            owner_id=owner_id,
            name=name,
            description=description,
            **kwargs
        )
        self.db.add(bot)
        self.db.commit()
        self.db.refresh(bot)
        return bot
    
    def get_user_bots(self, user_id: str) -> list:
        """Get all bots owned by a user"""
        return self.db.query(Bot).filter(Bot.owner_id == user_id).all()
    
    def get_bot_by_id(self, bot_id: str) -> Bot:
        """Get bot by ID"""
        return self.db.query(Bot).filter(Bot.id == bot_id).first()
    
    def update_user_login(self, user_id: str):
        """Update user's last login time"""
        user = self.get_user_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()