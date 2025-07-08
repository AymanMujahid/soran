from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

Base = declarative_base()

class BotStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"

class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime)
    subscription_tier = Column(String(50), default="free")
    api_key = Column(String(255), unique=True, index=True)
    
    bots = relationship("Bot", back_populates="owner", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

class Bot(Base):
    __tablename__ = "bots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default=BotStatus.ACTIVE, nullable=False)
    personality = Column(Text)
    system_prompt = Column(Text)
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=1000)
    model_name = Column(String(100), default="mistral-7b-instruct")
    embedding_model = Column(String(100), default="bge-small")
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_trained = Column(DateTime)
    training_data_count = Column(Integer, default=0)
    usage_count = Column(Integer, default=0)
    settings = Column(JSON)
    
    owner = relationship("User", back_populates="bots")
    files = relationship("BotFile", back_populates="bot", cascade="all, delete-orphan")
    sessions = relationship("ChatSession", back_populates="bot", cascade="all, delete-orphan")
    knowledge_base = relationship("KnowledgeChunk", back_populates="bot", cascade="all, delete-orphan")
    analytics = relationship("BotAnalytics", back_populates="bot", cascade="all, delete-orphan")

class BotFile(Base):
    __tablename__ = "bot_files"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    mime_type = Column(String(100))
    status = Column(String(20), default=FileStatus.PENDING, nullable=False)
    processed_at = Column(DateTime)
    error_message = Column(Text)
    extracted_text_length = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    bot = relationship("Bot", back_populates="files")
    chunks = relationship("KnowledgeChunk", back_populates="source_file", cascade="all, delete-orphan")

class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)
    source_file_id = Column(UUID(as_uuid=True), ForeignKey("bot_files.id"))
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    embedding_vector = Column(LargeBinary)
    embedding_model = Column(String(100), nullable=False)
    tags = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    
    bot = relationship("Bot", back_populates="knowledge_base")
    source_file = relationship("BotFile", back_populates="chunks")

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    session_name = Column(String(255))
    is_active = Column(Boolean, default=True, nullable=False)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    ended_at = Column(DateTime)
    message_count = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    session_metadata = Column(JSON)
    
    bot = relationship("Bot", back_populates="sessions")
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    tokens_used = Column(Integer, default=0)
    response_time_ms = Column(Integer)
    context_chunks_used = Column(JSON)
    feedback_score = Column(Integer)
    feedback_comment = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    metadata = Column(JSON)
    
    session = relationship("ChatSession", back_populates="messages")

class BotAnalytics(Base):
    __tablename__ = "bot_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    total_messages = Column(Integer, default=0)
    total_sessions = Column(Integer, default=0)
    total_users = Column(Integer, default=0)
    avg_response_time = Column(Float, default=0.0)
    avg_feedback_score = Column(Float)
    total_tokens_used = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    metrics = Column(JSON)
    
    bot = relationship("Bot", back_populates="analytics")

class APIUsage(Base):
    __tablename__ = "api_usage"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer)
    tokens_used = Column(Integer, default=0)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    metadata = Column(JSON)

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"), nullable=False)
    job_type = Column(String(50), nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime)
    progress_percentage = Column(Float, default=0.0)
    logs = Column(Text)
    error_message = Column(Text)
    parameters = Column(JSON)
    results = Column(JSON)

class SystemAlert(Base):
    __tablename__ = "system_alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    is_resolved = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    resolved_at = Column(DateTime)
    metadata = Column(JSON)

class ScheduledTask(Base):
    __tablename__ = "scheduled_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bot_id = Column(UUID(as_uuid=True), ForeignKey("bots.id"))
    task_name = Column(String(255), nullable=False)
    task_type = Column(String(50), nullable=False)
    schedule_expression = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_run = Column(DateTime)
    next_run = Column(DateTime, nullable=False)
    run_count = Column(Integer, default=0)
    parameters = Column(JSON)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)