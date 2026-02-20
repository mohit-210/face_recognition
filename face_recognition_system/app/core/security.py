import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import jwt

from app.core.config import get_settings


def _normalized_password_bytes(password: str) -> bytes:
    # bcrypt accepts max 72 bytes; normalize to fixed-length bytes first.
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    hashed = hashed_password.encode("utf-8")
    normalized = _normalized_password_bytes(plain_password)

    try:
        if bcrypt.checkpw(normalized, hashed):
            return True
    except ValueError:
        pass

    # Backward compatibility for legacy bcrypt hashes created from raw passwords.
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed)
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    normalized = _normalized_password_bytes(password)
    return bcrypt.hashpw(normalized, bcrypt.gensalt()).decode("utf-8")


def create_access_token(subject: str, company_id: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "company_id": company_id, "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str, company_id: int) -> str:
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.refresh_token_expire_minutes)
    payload: dict[str, Any] = {"sub": subject, "company_id": company_id, "exp": expire, "type": "refresh"}
    return jwt.encode(payload, settings.jwt_refresh_secret_key, algorithm=settings.jwt_algorithm)
