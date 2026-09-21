"""Esquemas Pydantic del modulo de autenticacion."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

MIN_PASSWORD_LENGTH = 8


class UserBase(BaseModel):
    """Campos comunes de un usuario."""

    email: EmailStr = Field(description="Correo electronico (identificador de acceso).", examples=["ana@escaneo.com"])
    full_name: str = Field(
        min_length=2,
        max_length=120,
        description="Nombre completo del usuario.",
        examples=["Ana Maria Gomez"],
    )


class UserCreate(UserBase):
    """Carga util para registrar un usuario nuevo."""

    password: str = Field(
        min_length=MIN_PASSWORD_LENGTH,
        max_length=128,
        description="Contrasena (minimo 8 caracteres, con letras y numeros).",
        examples=["Segura123"],
    )

    @field_validator("password")
    @classmethod
    def _strong_enough(cls, value: str) -> str:
        if not any(char.isalpha() for char in value):
            raise ValueError("La contrasena debe incluir al menos una letra.")
        if not any(char.isdigit() for char in value):
            raise ValueError("La contrasena debe incluir al menos un numero.")
        return value

    @field_validator("full_name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return " ".join(value.split())


class UserRead(UserBase):
    """Representacion publica de un usuario (nunca expone el hash)."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Identificador unico del usuario.")
    is_active: bool = Field(default=True, description="Indica si la cuenta esta habilitada.")
    created_at: datetime = Field(description="Fecha de creacion de la cuenta.")


class LoginRequest(BaseModel):
    """Credenciales enviadas desde el cliente Django (JSON)."""

    email: EmailStr = Field(description="Correo electronico registrado.", examples=["admin@escaneo.com"])
    password: str = Field(description="Contrasena de la cuenta.", examples=["Admin123*"])


class Token(BaseModel):
    """Respuesta del inicio de sesion."""

    access_token: str = Field(description="JWT que debe enviarse en `Authorization: Bearer <token>`.")
    token_type: str = Field(default="bearer", description="Tipo de token.", examples=["bearer"])
    expires_in: int = Field(description="Segundos de vigencia del token.", examples=[7200])
    user: UserRead = Field(description="Datos del usuario autenticado.")


class TokenPayload(BaseModel):
    """Contenido decodificado del JWT."""

    sub: str = Field(description="Identificador del usuario (`subject`).")
    iat: int = Field(description="Emision (epoch).")
    exp: int = Field(description="Expiracion (epoch).")
    email: str | None = Field(default=None, description="Correo del usuario.")
    iss: str | None = Field(default=None, description="Emisor del token.")


__all__ = [
    "LoginRequest",
    "MIN_PASSWORD_LENGTH",
    "Token",
    "TokenPayload",
    "UserBase",
    "UserCreate",
    "UserRead",
]
