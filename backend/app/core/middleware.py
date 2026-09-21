"""Middleware transversal: identificador de peticion y medicion de tiempos.

El identificador viaja en la cabecera ``X-Request-ID`` y se incluye en las
respuestas de error, de modo que un usuario pueda reportar un fallo y en los
registros se encuentre exactamente esa peticion.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"
PROCESS_TIME_HEADER = "X-Process-Time-Ms"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Anota cada peticion con un identificador y su duracion."""

    def __init__(self, app: ASGIApp, *, log_requests: bool = True) -> None:
        super().__init__(app)
        self.log_requests = log_requests

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "request_id=%s %s %s fallo tras %.0f ms",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{elapsed_ms:.1f}"

        if self.log_requests:
            logger.info(
                "request_id=%s %s %s -> %s en %.0f ms",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )

        return response


__all__ = ["PROCESS_TIME_HEADER", "REQUEST_ID_HEADER", "RequestContextMiddleware"]
