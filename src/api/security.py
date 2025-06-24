"""
Security utilities for API authentication and authorization.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.config import settings


class APIKeyAuth:
    """Simple API key authentication."""
    
    def __init__(self, api_keys: Optional[list] = None):
        self.api_keys = api_keys or []
    
    def authenticate(self, api_key: str) -> bool:
        """Authenticate API key."""
        if not self.api_keys:  # No API keys configured, allow all
            return True
        return api_key in self.api_keys


class JWTAuth:
    """JWT token authentication."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> dict:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
