"""Primitivas de seguridad: hash de contrasenas y tokens JWT.

Se implementa el hash con `hashlib.pbkdf2_hmac` (el mismo algoritmo que usa
Django) para no depender de librerias nativas y mantener el proyecto liviano
en entornos serverless.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt

from app.core.config import get_settings

PBKDF2_ALGORITHM = "sha256"
PBKDF2_ITERATIONS = 260_000
PBKDF2_PREFIX = "pbkdf2_sha256"
_SALT_BYTES = 16


class InvalidTokenError(Exception):
    """El token JWT no es valido o ya expiro."""


# ---------------------------------------------------------------------------
# Contrasenas
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Devuelve el hash `pbkdf2_sha256$iteraciones$salt$hash`."""
    salt = secrets.token_hex(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM, password.encode("utf-8"), salt.encode("utf-8"), PBKDF2_ITERATIONS
    )
    return f"{PBKDF2_PREFIX}${PBKDF2_ITERATIONS}${salt}${base64.b64encode(digest).decode()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compara una contrasena en claro contra su hash almacenado."""
    try:
        prefix, raw_iterations, salt, encoded = hashed_password.split("$")
        if prefix != PBKDF2_PREFIX:
            return False
        iterations = int(raw_iterations)
    except (ValueError, AttributeError):
        return False

    digest = hashlib.pbkdf2_hmac(
        PBKDF2_ALGORITHM, plain_password.encode("utf-8"), salt.encode("utf-8"), iterations
    )
    return hmac.compare_digest(digest, base64.b64decode(encoded))


# ---------------------------------------------------------------------------
# Tokens JWT
# ---------------------------------------------------------------------------
def utcnow() -> datetime:
    """Fecha/hora actual en UTC con zona horaria."""
    return datetime.now(timezone.utc)


def create_access_token(
    subject: str,
    *,
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> tuple[str, int]:
    """Crea un JWT firmado.

    Returns:
        Tupla ``(token, segundos_de_vigencia)``.
    """
    settings = get_settings()
    expires_delta = expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    issued_at = utcnow()
    expires_at = issued_at + expires_delta

    claims: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": settings.app_name,
    }
    if extra_claims:
        claims.update(extra_claims)

    token = jwt.encode(claims, settings.secret_key, algorithm=settings.algorithm)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str) -> dict[str, Any]:
    """Decodifica y valida un JWT. Lanza :class:`InvalidTokenError` si falla."""
    settings = get_settings()
    try:
        return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
    except JWTError as exc:  # firma invalida, expirado, malformado...
        raise InvalidTokenError(str(exc)) from exc


__all__ = [
    "InvalidTokenError",
    "create_access_token",
    "decode_access_token",
    "hash_password",
    "utcnow",
    "verify_password",
]
