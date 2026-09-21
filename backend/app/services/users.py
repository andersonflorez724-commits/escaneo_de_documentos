"""Repositorio de usuarios del backend.

Se mantiene en memoria (con un usuario semilla) para que el taller funcione
sin infraestructura adicional. La interfaz es la misma que tendria un
repositorio respaldado por base de datos, de modo que el reemplazo no afecta
a los endpoints.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.core.config import get_settings
from app.core.security import hash_password, utcnow, verify_password


@dataclass(slots=True)
class UserRecord:
    """Usuario almacenado en el repositorio."""

    id: str
    email: str
    full_name: str
    hashed_password: str
    is_active: bool = True
    created_at: datetime = field(default_factory=utcnow)


class EmailAlreadyRegisteredError(ValueError):
    """Ya existe una cuenta con ese correo."""


class UserRepository:
    """Almacen de usuarios seguro para hilos."""

    def __init__(self) -> None:
        self._by_id: dict[str, UserRecord] = {}
        self._by_email: dict[str, UserRecord] = {}
        self._lock = threading.Lock()

    # ----------------------------- utilidades -----------------------------
    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    # -------------------------------- lectura ------------------------------
    def get_by_email(self, email: str) -> UserRecord | None:
        return self._by_email.get(self.normalize_email(email))

    def get_by_id(self, user_id: str) -> UserRecord | None:
        return self._by_id.get(str(user_id))

    def count(self) -> int:
        return len(self._by_id)

    # -------------------------------- escritura ----------------------------
    def create(self, *, email: str, password: str, full_name: str) -> UserRecord:
        normalized = self.normalize_email(email)
        with self._lock:
            if normalized in self._by_email:
                raise EmailAlreadyRegisteredError(normalized)

            user = UserRecord(
                id=str(uuid.uuid4()),
                email=normalized,
                full_name=" ".join(full_name.split()),
                hashed_password=hash_password(password),
            )
            self._by_id[user.id] = user
            self._by_email[normalized] = user
            return user

    def authenticate(self, email: str, password: str) -> UserRecord | None:
        """Devuelve el usuario si las credenciales son validas."""
        user = self.get_by_email(email)
        if user is None or not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def ensure_seed_user(self) -> UserRecord:
        """Crea el usuario demo definido en el entorno (idempotente)."""
        settings = get_settings()
        existing = self.get_by_email(settings.seed_user_email)
        if existing is not None:
            return existing
        return self.create(
            email=settings.seed_user_email,
            password=settings.seed_user_password,
            full_name=settings.seed_user_name,
        )


_repository = UserRepository()


def get_user_repository() -> UserRepository:
    """Dependencia de FastAPI que entrega el repositorio de usuarios."""
    return _repository


__all__ = [
    "EmailAlreadyRegisteredError",
    "UserRecord",
    "UserRepository",
    "get_user_repository",
]
