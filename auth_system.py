import secrets
import hashlib
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from email_validator import validate_email, EmailNotValidError
import logging

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-super-secret-key-change-in-production-use-env-file"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")

class AuthenticationError(Exception):
    pass

class SecurityManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key"""
        return f"nxs_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

class UserService:
    """Service for user management operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def validate_email(self, email: str) -> str:
        """Validate email format"""
        try:
            valid = validate_email(email)
            return valid.email
        except EmailNotValidError:
            raise ValueError("Invalid email format")
    
    def validate_password(self, password: str) -> str:
        """Validate password requirements"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one number")
        
        return password
    
    def create_user(self, email: str, username: str, password: str, full_name: Optional[str] = None) -> Dict:
        """Create a new user (placeholder for database integration)"""
        # This will be connected to actual database models later
        email = self.validate_email(email)
        password = self.validate_password(password)
        
        # Check if user exists (placeholder)
        # In real implementation, check database
        
        hashed_password = SecurityManager.hash_password(password)
        api_key = SecurityManager.generate_api_key()
        
        user_data = {
            "id": secrets.token_urlsafe(16),
            "email": email,
            "username": username,
            "full_name": full_name,
            "hashed_password": hashed_password,
            "api_key": api_key,
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        logger.info(f"User created successfully: {email}")
        return user_data
    
    def authenticate_user(self, email: str, password: str) -> Optional[Dict]:
        """Authenticate user with email and password"""
        # Placeholder for database lookup
        # In real implementation, query database for user
        
        # For testing, create a dummy user
        if email == "test@noxus.com" and password == "TestPassword123":
            return {
                "id": "test_user_id",
                "email": email,
                "username": "testuser",
                "full_name": "Test User",
                "is_active": True
            }
        
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Get user by ID (placeholder)"""
        # Placeholder for database lookup
        if user_id == "test_user_id":
            return {
                "id": user_id,
                "email": "test@noxus.com",
                "username": "testuser",
                "full_name": "Test User",
                "is_active": True
            }
        return None

# Dependency functions for FastAPI
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = None) -> Dict:
    """Get current user from JWT token"""
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
    
    # For now, use the service without database
    user_service = UserService(db)
    user = user_service.get_user_by_id(user_id)
    
    if user is None:
        raise credentials_exception
    
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return user

def require_active_user(current_user: Dict = Depends(get_current_user)) -> Dict:
    """Require an active user"""
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

# Rate limiting (simple in-memory implementation)
class SimpleRateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int = 60, window: int = 60) -> bool:
        """Check if request is allowed based on rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > window_start]
        
        if len(self.requests[key]) >= limit:
            return False
        
        self.requests[key].append(now)
        return True

# Global rate limiter instance
rate_limiter = SimpleRateLimiter()

def check_rate_limit(request_id: str, limit: int = 60) -> bool:
    """Check rate limit for a request"""
    if not rate_limiter.is_allowed(request_id, limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    return True