import secrets
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from email_validator import validate_email, EmailNotValidError
from models import User, APIUsage
from database import get_db
from config import settings
import logging

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/token")
bearer_scheme = HTTPBearer()

class AuthenticationError(Exception):
    pass

class AuthorizationError(Exception):
    pass

class SecurityManager:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_api_key() -> str:
        return f"nxs_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            if payload.get("type") != token_type:
                raise jwt.InvalidTokenError("Invalid token type")
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

class UserManager:
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, email: str, username: str, password: str, full_name: Optional[str] = None) -> User:
        if self.get_user_by_email(email):
            raise ValueError("Email already registered")
        
        if self.get_user_by_username(username):
            raise ValueError("Username already taken")
        
        try:
            valid = validate_email(email)
            email = valid.email
        except EmailNotValidError:
            raise ValueError("Invalid email format")
        
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        hashed_password = SecurityManager.hash_password(password)
        api_key = SecurityManager.generate_api_key()
        
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=full_name,
            api_key=SecurityManager.hash_api_key(api_key)
        )
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        logger.info(f"New user created: {user.email}")
        return user
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not SecurityManager.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"User authenticated: {user.email}")
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        hashed_key = SecurityManager.hash_api_key(api_key)
        user = self.db.query(User).filter(User.api_key == hashed_key).first()
        if not user or not user.is_active:
            return None
        return user
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        return self.db.query(User).filter(User.id == user_id).first()
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        user = self.get_user_by_id(user_id)
        if not user:
            return False
        
        if not SecurityManager.verify_password(old_password, user.hashed_password):
            return False
        
        if len(new_password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        user.hashed_password = SecurityManager.hash_password(new_password)
        self.db.commit()
        
        logger.info(f"Password changed for user: {user.email}")
        return True
    
    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        user = self.get_user_by_id(user_id)
        if not user:
            return None
        
        new_api_key = SecurityManager.generate_api_key()
        user.api_key = SecurityManager.hash_api_key(new_api_key)
        self.db.commit()
        
        logger.info(f"API key regenerated for user: {user.email}")
        return new_api_key

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
    except AuthenticationError:
        raise credentials_exception
    
    user_manager = UserManager(db)
    user = user_manager.get_user_by_id(user_id)
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

def get_current_user_api_key(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: Session = Depends(get_db)) -> User:
    api_key = credentials.credentials
    if not api_key.startswith("nxs_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )
    
    user_manager = UserManager(db)
    user = user_manager.authenticate_api_key(api_key)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return user

def get_current_user_flexible(token: Optional[str] = Depends(oauth2_scheme), credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme), db: Session = Depends(get_db)) -> User:
    if token:
        return get_current_user(token, db)
    elif credentials:
        return get_current_user_api_key(credentials, db)
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

def require_active_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_superuser(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

def check_bot_ownership(bot_id: str, current_user: User, db: Session) -> bool:
    from models import Bot
    bot = db.query(Bot).filter(Bot.id == bot_id).first()
    if not bot:
        return False
    return bot.owner_id == current_user.id or current_user.is_superuser

def require_bot_ownership(bot_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not check_bot_ownership(bot_id, current_user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this bot"
        )
    return True

class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
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

def log_api_usage(endpoint: str, method: str, status_code: int, response_time_ms: int, user_id: str, db: Session, **kwargs):
    try:
        usage = APIUsage(
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            **kwargs
        )
        db.add(usage)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log API usage: {e}")

def validate_input_length(text: str, max_length: int = 10000) -> str:
    if len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Input too long. Maximum {max_length} characters allowed."
        )
    return text

def sanitize_filename(filename: str) -> str:
    import re
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    filename = filename[:255]
    return filename