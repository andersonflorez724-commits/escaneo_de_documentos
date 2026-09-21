"""Endpoints de autenticacion (registro, login y perfil)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import CurrentUser, UserRepo
from app.core.security import create_access_token
from app.schemas.auth import LoginRequest, Token, UserCreate, UserRead
from app.services.users import EmailAlreadyRegisteredError, UserRecord, UserRepository

router = APIRouter(prefix="/auth", tags=["Autenticacion"])

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def _issue_token(user: UserRecord) -> Token:
    """Genera el token de acceso y lo empaqueta con los datos del usuario."""
    token, expires_in = create_access_token(
        subject=user.id,
        extra_claims={"email": user.email},
    )
    return Token(
        access_token=token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserRead.model_validate(user),
    )


def _authenticate_or_401(repository: UserRepository, email: str, password: str) -> UserRecord:
    user = repository.authenticate(email, password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas.",
            headers=_UNAUTHORIZED_HEADERS,
        )
    return user


REGISTER_EXAMPLES = {
    "nuevo_usuario": {
        "summary": "Registro tipico",
        "value": {
            "email": "ana@escaneo.com",
            "full_name": "Ana Maria Gomez",
            "password": "Segura123",
        },
    }
}

LOGIN_EXAMPLES = {
    "usuario_semilla": {
        "summary": "Cuenta de demostracion",
        "value": {"email": "admin@escaneo.com", "password": "Admin123*"},
    }
}


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar un usuario",
    description=(
        "Crea una cuenta nueva. El correo electronico es el identificador de "
        "acceso y la contrasena debe tener al menos 8 caracteres, con letras y "
        "numeros."
    ),
    responses={
        409: {"description": "El correo ya esta registrado."},
        422: {"description": "Datos invalidos (correo mal formado o contrasena debil)."},
    },
)
async def register(
    payload: Annotated[UserCreate, Body(openapi_examples=REGISTER_EXAMPLES)],
    repository: UserRepo,
) -> UserRead:
    try:
        user = repository.create(
            email=str(payload.email),
            password=payload.password,
            full_name=payload.full_name,
        )
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe una cuenta registrada con ese correo electronico.",
        ) from exc

    return UserRead.model_validate(user)


@router.post(
    "/login",
    response_model=Token,
    summary="Iniciar sesion (JSON)",
    description=(
        "Autentica con correo y contrasena y devuelve un **JWT**. "
        "Es el endpoint que consume el cliente Django."
    ),
    responses={401: {"description": "Credenciales incorrectas."}},
)
async def login(
    payload: Annotated[LoginRequest, Body(openapi_examples=LOGIN_EXAMPLES)],
    repository: UserRepo,
) -> Token:
    user = _authenticate_or_401(repository, str(payload.email), payload.password)
    return _issue_token(user)


@router.post(
    "/token",
    response_model=Token,
    summary="Iniciar sesion (OAuth2 / Swagger)",
    description=(
        "Variante `application/x-www-form-urlencoded` que habilita el boton "
        "**Authorize** de Swagger UI. Usa el email en el campo *username*."
    ),
    responses={401: {"description": "Credenciales incorrectas."}},
)
async def login_oauth2(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    repository: UserRepo,
) -> Token:
    user = _authenticate_or_401(repository, form_data.username, form_data.password)
    return _issue_token(user)


@router.get(
    "/me",
    response_model=UserRead,
    summary="Perfil del usuario autenticado",
    description="Devuelve los datos del usuario dueno del token JWT enviado.",
    responses={401: {"description": "Token ausente, invalido o expirado."}},
)
async def read_current_user(current_user: CurrentUser) -> UserRead:
    return UserRead.model_validate(current_user)
