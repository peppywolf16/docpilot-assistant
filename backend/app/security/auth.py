import time
from typing import Literal
import jwt
from pydantic import BaseModel

# Secret key for signing JWTs (in production, stored securely in .env)
JWT_SECRET = "docpilot-super-secret-enterprise-key-2026"
JWT_ALGORITHM = "HS256"


class UserContext(BaseModel):
    user_id: str
    tenant: str                                    # e.g. "engineering", "finance"
    role: Literal["admin", "engineer", "intern"]   # RBAC role


def create_access_token(user_id: str, tenant: str, role: str, expires_in_sec: int = 3600) -> str:
    """Creates a signed, tamper-proof JWT token containing tenant and role permissions."""
    payload = {
        "sub": user_id,
        "tenant": tenant,
        "role": role,
        "exp": int(time.time()) + expires_in_sec,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_access_token(token: str) -> UserContext:
    """
    Decodes and verifies the JWT signature.
    Raises jwt.PyJWTError if the token has expired, been tampered with, or is invalid.
    """
    payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    return UserContext(
        user_id=payload["sub"],
        tenant=payload["tenant"],
        role=payload["role"],
    )