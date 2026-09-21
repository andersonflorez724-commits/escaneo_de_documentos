"""Dependencias compartidas de la API (autenticacion, paginacion, etc.)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings
from app.core.security import InvalidTokenError, decode_access_token
from app.services.users import UserRecord, UserRepository, get_user_repository

settings = get_settings()

# tokenUrl apunta al endpoint que acepta `application/x-www-form-urlencoded`,
# que es el que usa el boton "Authorize" de Swagger UI.
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/auth/token",
    auto_error=False,
    description="Token JWT obtenido en `/api/v1/auth/login` o `/api/v1/auth/token`.",
)

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_UNAUTHORIZED_HEADERS,
    )


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    repository: Annotated[UserRepository, Depends(get_user_repository)],
) -> UserRecord:
    """Resuelve el usuario autenticado a partir del JWT."""
    if not token:
        raise _unauthorized("No se proporciono el token de acceso.")

    try:
        payload = decode_access_token(token)
    except InvalidTokenError as exc:
        raise _unauthorized("El token es invalido o ha expirado.") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise _unauthorized("El token no contiene el sujeto (`sub`).")

    user = repository.get_by_id(str(user_id))
    if user is None or not user.is_active:
        raise _unauthorized("El usuario del token ya no esta disponible.")

    return user


CurrentUser = Annotated[UserRecord, Depends(get_current_user)]
UserRepo = Annotated[UserRepository, Depends(get_user_repository)]

__all__ = ["CurrentUser", "UserRepo", "get_current_user", "oauth2_scheme"]
